import math
import time
from subprocess import Popen

import cv2
import ffmpeg
import numpy as np

from silver_disco.frame import Frame


class FrameProcessor:
    output_filename: str
    output_writer: Popen | None = None
    cumulative_time: int = 0

    def __init__(self, output_filename: str = ""):
        self.output_filename = output_filename

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        if self.output_writer:
            self.output_writer.stdin.close()
            self.output_writer.wait()
            self.output_writer = None

    def _frame_to_write(self, frame: Frame) -> cv2.typing.MatLike:
        return frame.raw

    def write_frame(self, frame: Frame) -> None:
        if self.output_filename:
            if not self.output_writer:
                self.output_writer = (
                    ffmpeg.input(
                        "pipe:",
                        format="rawvideo",
                        pix_fmt="rgb24",
                        s=f"{frame.width}x{frame.height}",
                    )
                    .output(self.output_filename, pix_fmt="yuv420p")
                    .overwrite_output()
                    .run_async(
                        pipe_stdin=True,
                        quiet=True,
                    )
                )
            self.output_writer.stdin.write(
                self._frame_to_write(frame).astype(np.uint8).tobytes()
            )

    def handle_time(self, frame: Frame) -> Frame:
        start = time.time()
        frame_output = self.handle(frame)
        end = time.time()
        self.cumulative_time += end - start
        return frame_output

    def handle(self, frame: Frame) -> Frame:
        return frame


class BackgroundSubtractorCV2Processor(FrameProcessor):
    _background_subtractor: cv2.BackgroundSubtractor | None = None

    def _frame_to_write(self, frame: Frame) -> cv2.typing.MatLike:
        return cv2.cvtColor(frame.background_subtracted, cv2.COLOR_GRAY2BGR)

    def handle(self, frame: Frame) -> Frame:
        frame.background_subtracted = self.background_subtractor.apply(frame.raw)
        frame.background = self.background_subtractor.getBackgroundImage()

        # frigate uses bluring and thresholding
        # opencv tutorial uses morphology kernels
        kernel = np.ones((7, 7), np.uint8)
        frame.background_subtracted = cv2.morphologyEx(
            frame.background_subtracted, cv2.MORPH_OPEN, kernel
        )
        frame.background_subtracted = cv2.morphologyEx(
            frame.background_subtracted, cv2.MORPH_CLOSE, kernel
        )
        return super().handle(frame)


class BackgroundSubtractorKNNProcessor(BackgroundSubtractorCV2Processor):
    @property
    def background_subtractor(self) -> cv2.BackgroundSubtractor:
        if not self._background_subtractor:
            self._background_subtractor = cv2.createBackgroundSubtractorKNN(
                detectShadows=False
            )
        return self._background_subtractor


class BackgroundSubtractorMOG2Processor(BackgroundSubtractorCV2Processor):
    @property
    def background_subtractor(self) -> cv2.BackgroundSubtractor:
        if not self._background_subtractor:
            self._background_subtractor = cv2.createBackgroundSubtractorMOG2(
                detectShadows=False
            )
        return self._background_subtractor


class ContourProcessor(FrameProcessor):
    def _frame_to_write(self, frame: Frame) -> cv2.typing.MatLike:
        contours_img = frame.raw.copy()
        if frame.contours:
            cv2.drawContours(
                contours_img, frame.contours, -1, (255, 0, 0, 128), thickness=cv2.FILLED
            )
            for contour in frame.contours:
                contour_bbox = cv2.boundingRect(contour)
                cv2.rectangle(
                    contours_img,
                    (int(contour_bbox[0]), int(contour_bbox[1])),
                    (
                        int(contour_bbox[0] + contour_bbox[2]),
                        int(contour_bbox[1] + contour_bbox[3]),
                    ),
                    (0, 255, 0),
                )
        return contours_img

    def _get_contour_min_size(
        self, contour_centre_y: int, width: int, height: int
    ) -> int:
        # 0 at midline (horizon), 1 at edge
        contour_proportion_halfway = abs(contour_centre_y - (height / 2)) / (height / 2)

        sizeabs = 0.015
        sizescale = contour_proportion_halfway * 0.15

        contour_min_size = ((sizescale + sizeabs) * width) * (
            (sizescale + sizeabs) * height
        )
        return contour_min_size

    def _filter_contour(
        self, contour: cv2.typing.MatLike, width: int, height: int
    ) -> bool:
        area = cv2.contourArea(contour)
        # size based on position in frame - smaller on horizon
        contour_bbox = cv2.boundingRect(contour)
        contour_centre = (
            contour_bbox[0] + (contour_bbox[2] / 2),
            contour_bbox[1] + (contour_bbox[3] / 2),
        )
        contour_centre_y = contour_bbox[1] + (contour_bbox[3] / 2)

        contour_min_size = self._get_contour_min_size(contour_centre_y, width, height)
        contour_max_size = (0.90 * width) * (0.90 * height)
        if area < contour_min_size:
            return False
        if area > contour_max_size:
            return False
        return True

    def handle(self, frame: Frame) -> Frame:
        frame.contours, _ = cv2.findContours(
            frame.background_subtracted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        frame.contours = [
            contour
            for contour in frame.contours
            if self._filter_contour(contour, frame.width, frame.height)
        ]
        return super().handle(frame)


def get_rect_centre(bbox: cv2.typing.Rect) -> tuple[float, float]:
    return (bbox[0] + (bbox[2] / 2)), (bbox[1] + (bbox[3] / 2))


def get_rect_overlap(bbox1: cv2.typing.Rect, bbox2: cv2.typing.Rect) -> float:
    # order l->r,b->t
    if (bbox1[0], bbox1[1]) > (bbox2[0], bbox2[1]):
        return get_rect_overlap(bbox2, bbox1)

    x1, y1, w1, h1 = bbox1
    l1, r1, b1, t1 = x1, x1 + w1, y1, y1 + h1

    x2, y2, w2, h2 = bbox2
    l2, r2, b2, t2 = x2, x2 + w2, y2, y2 + h2

    if l2 > r1 and b2 > t1:
        return 0.0

    inner_left = max(l1, l2)
    inner_right = min(r1, r2)
    inner_bottom = max(b1, b2)
    inner_top = min(t1, t2)
    inner_area = (inner_right - inner_left) * (inner_top - inner_bottom)
    total_area = (bbox1[2] * bbox1[3]) + (bbox2[2] * bbox2[3]) - inner_area

    return inner_area / total_area


def enlarge_rect(
    rect: cv2.typing.Rect, max_width: int, max_height: int, scale: float = 1.0
) -> cv2.typing.Rect:
    x, y, w, h = rect
    x = max(0, x - w * scale)
    y = max(0, y - h * scale)
    w = min(max_width - x, w * (1 + 2 * scale))
    h = min(max_height - y, h * (1 + 2 * scale))
    return (int(x), int(y), int(w), int(h))


class TrackingProcessor(FrameProcessor):
    trackers: list[cv2.Tracker]
    trackers_rect: list[cv2.typing.Rect]
    trackers_age: list[int]

    def __init__(self, output_filename: str = ""):
        super().__init__(output_filename)
        self.trackers = []
        self.trackers_rect = []
        self.trackers_age = []

    def _find_tracker(
        self, bbox: cv2.typing.Rect, age_max: int = 25, overlap_min: float = 00.0
    ) -> cv2.Tracker | None:
        # TODO faster data structure!
        closest_tracker = None
        closest_dist = 0.0
        for tracker, tracker_rect, tracker_age in zip(
            self.trackers, self.trackers_rect, self.trackers_age, strict=False
        ):
            if (
                tracker_age < age_max
                and get_rect_overlap(bbox, tracker_rect) > overlap_min
            ):
                bbox_centre = get_rect_centre(bbox)
                tracker_centre = get_rect_centre(tracker_rect)
                dist = math.sqrt(
                    (bbox_centre[0] - tracker_centre[0]) ** 2
                    + (bbox_centre[1] - tracker_centre[1]) ** 2
                )
                if closest_tracker is None or dist < closest_dist:
                    closest_dist = dist
                    closest_tracker = tracker
        return closest_tracker

    def _create_tracker(self):
        # tracker picked using https://broutonlab.com/blog/opencv-object-tracking/
        tracker = cv2.TrackerKCF.create()
        # tracker = cv2.TrackerCSRT.create()
        # tracker = cv2.legacy.TrackerMedianFlow.create()
        # tracker = cv2.legacy.TrackerBoosting.create()
        # tracker = cv2.TrackerDaSiamRPN.create()
        # tracker = cv2.legacy.TrackerMOSSE_create()
        return tracker

    def _update_existing_trackers(self, frame: Frame) -> None:
        for i, tracker in enumerate(self.trackers):
            if self.trackers_age[i]:
                # this tracker has previously lost target, grow slightly
                self.trackers_age[i] += 1
                if self.trackers_age[i] < 25:
                    self.trackers_rect[i] = enlarge_rect(
                        self.trackers_rect[i], frame.width, frame.height, 0.05
                    )
            else:
                ok, bbox = tracker.update(frame.raw)
                if ok:
                    # still in frame, update
                    self.trackers_rect[i] = [int(n) for n in bbox]
                    self.trackers_age[i] = 0
                else:
                    # gone missing
                    self.trackers_age[i] = 1

    def _update_from_contours(self, frame: Frame) -> None:
        for contour in frame.contours:
            bbox = cv2.boundingRect(contour)  # x, y, w, h
            # has an existing tracker?
            if tracker := self._find_tracker(bbox):
                i = self.trackers.index(tracker)
                self.trackers_age[i] = 0
            else:
                tracker = self._create_tracker()
                bbox = enlarge_rect(bbox, frame.width, frame.height, 0.5)
                tracker.init(frame.raw, bbox)
                self.trackers.append(tracker)
                self.trackers_rect.append(bbox)
                self.trackers_age.append(0)

    def handle(self, frame: Frame) -> Frame:
        # update trackers with new frame
        self._update_existing_trackers(frame)
        # find new trackers to add
        self._update_from_contours(frame)

        # store for stats later
        for tracker, tracker_rect, tracker_age in zip(
            self.trackers, self.trackers_rect, self.trackers_age, strict=False
        ):
            frame.tracked[id(tracker)] = tracker_rect

        return super().handle(frame)

    def _frame_to_write(self, frame: Frame) -> cv2.typing.MatLike:
        contours_img = frame.raw.copy()
        for tracker, tracker_rect, tracker_age in zip(
            self.trackers, self.trackers_rect, self.trackers_age, strict=False
        ):
            cv2.rectangle(
                contours_img,
                (int(tracker_rect[0]), int(tracker_rect[1])),
                (
                    int(tracker_rect[0] + tracker_rect[2]),
                    int(tracker_rect[1] + tracker_rect[3]),
                ),
                (255, 0, 0) if tracker_age else (0, 255, 0),
            )
            cv2.putText(
                contours_img,
                f"{id(tracker)}:{tracker_age}",
                (int(tracker_rect[0]), int(tracker_rect[1] + tracker_rect[3])),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 0) if tracker_age else (0, 255, 0),
            )

        return contours_img


class TrackingSimpleProcessor(FrameProcessor):
    trackers: list[str]
    trackers_rect: list[cv2.typing.Rect]
    trackers_age: list[int]

    def __init__(self, output_filename: str = ""):
        super().__init__(output_filename)
        self.trackers = []
        self.trackers_rect = []
        self.trackers_age = []

    def _find_tracker(
        self, bbox: cv2.typing.Rect, age_max: int = -1, overlap_min: float = 00.0
    ) -> str | None:
        # TODO faster data structure!
        closest_tracker = None
        closest_dist = 0.0
        for tracker, tracker_rect, tracker_age in zip(
            self.trackers, self.trackers_rect, self.trackers_age, strict=False
        ):
            if (age_max < 0 or tracker_age < age_max) and get_rect_overlap(
                bbox, tracker_rect
            ) > overlap_min:
                bbox_centre = get_rect_centre(bbox)
                tracker_centre = get_rect_centre(tracker_rect)
                dist = math.sqrt(
                    (bbox_centre[0] - tracker_centre[0]) ** 2
                    + (bbox_centre[1] - tracker_centre[1]) ** 2
                )
                if closest_tracker is None or dist < closest_dist:
                    closest_dist = dist
                    closest_tracker = tracker
        return closest_tracker

    def _update_from_contours(self, frame: Frame) -> None:
        for i, _ in enumerate(self.trackers):
            self.trackers_age[i] += 1

        for contour in frame.contours:
            bbox = cv2.boundingRect(contour)  # x, y, w, h
            # has an existing tracker?
            if tracker := self._find_tracker(bbox, 50):
                i = self.trackers.index(tracker)
                self.trackers_rect[i] = bbox
                self.trackers_age[i] = 0
            else:
                self.trackers.append(str(len(self.trackers)))
                self.trackers_rect.append(bbox)
                self.trackers_age.append(0)

        for i, _ in enumerate(self.trackers):
            if self.trackers_age[i] == 1:
                # this tracker has lost target, grow slightly
                self.trackers_rect[i] = enlarge_rect(
                    self.trackers_rect[i], frame.width, frame.height, 0.5
                )

    def handle(self, frame: Frame) -> Frame:
        self._update_from_contours(frame)

        # store for stats later
        for tracker, tracker_rect, tracker_age in zip(
            self.trackers, self.trackers_rect, self.trackers_age, strict=False
        ):
            frame.tracked[id(tracker)] = tracker_rect

        return super().handle(frame)

    def _frame_to_write(self, frame: Frame) -> cv2.typing.MatLike:
        contours_img = frame.raw.copy()
        for tracker, tracker_rect, tracker_age in zip(
            self.trackers, self.trackers_rect, self.trackers_age, strict=False
        ):
            cv2.rectangle(
                contours_img,
                (int(tracker_rect[0]), int(tracker_rect[1])),
                (
                    int(tracker_rect[0] + tracker_rect[2]),
                    int(tracker_rect[1] + tracker_rect[3]),
                ),
                (255, 0, 0) if tracker_age else (0, 255, 0),
            )
            cv2.putText(
                contours_img,
                f"{tracker}:{tracker_age}",
                (int(tracker_rect[0]), int(tracker_rect[1] + tracker_rect[3])),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 0) if tracker_age else (0, 255, 0),
            )

        return contours_img
