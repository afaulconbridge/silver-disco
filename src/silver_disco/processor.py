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


class BackgroundSubtractorKNNProcessor(FrameProcessor):
    _background_subtractor: cv2.BackgroundSubtractor | None = None

    @property
    def background_subtractor(self) -> cv2.BackgroundSubtractor:
        if not self._background_subtractor:
            self._background_subtractor = cv2.createBackgroundSubtractorKNN()
        return self._background_subtractor

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
        _, frame.background_subtracted = cv2.threshold(
            frame.background_subtracted, 100, 255, cv2.THRESH_BINARY
        )

        return super().handle(frame)


class ContourProcessor(FrameProcessor):
    def _frame_to_write(self, frame: Frame) -> cv2.typing.MatLike:
        contours_img = frame.raw.copy()
        if frame.contours:
            cv2.drawContours(contours_img, frame.contours, -1, (255, 0, 0))
        return contours_img

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

        # 0 at midline (horizon), 1 at edge
        contour_proportion_halfway = abs(
            (contour_bbox[1] + (contour_bbox[3] / 2)) - (height / 2)
        ) / (height / 2)

        sizeabs = 0.02
        sizescale = contour_proportion_halfway * 0.25

        contour_min_size = ((sizescale + sizeabs) * width) * (
            (sizescale + sizeabs) * height
        )
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


class TrackingProcessor(FrameProcessor):
    trackers: list[cv2.Tracker]
    trackers_rect: list[cv2.Tracker]

    def __init__(self, output_filename: str = ""):
        super().__init__(output_filename)
        self.trackers = []
        self.trackers_rect = []

    @staticmethod
    def _get_overlap(bbox1: cv2.typing.Rect, bbox2: cv2.typing.Rect) -> float:
        # order l->r,b->t
        if (bbox1[0], bbox1[1]) > (bbox2[0], bbox2[1]):
            bbox2, bbox1 = bbox1, bbox2
        inner_left = max(bbox1[0], bbox2[0])
        inner_right = max(bbox1[0] + bbox1[2], bbox2[0] + bbox2[2])
        inner_bottom = max(bbox1[1], bbox2[1])
        inner_top = max(bbox1[1] + bbox1[3], bbox2[1] + bbox2[3])
        inner_area = (inner_right - inner_left) * (inner_top - inner_bottom)
        total_area = (bbox1[2] * bbox1[3]) + (bbox2[2] * bbox2[3]) - inner_area

        return inner_area / total_area

    def _find_tracker(self, bbox: cv2.typing.Rect) -> cv2.Tracker | None:
        # TODO faster data structure!
        for tracker, tracker_rect in zip(
            self.trackers, self.trackers_rect, strict=False
        ):
            if self._get_overlap(bbox, tracker_rect) > 0.01:
                return tracker
        return None

    def _update_tracker_rect(self, tracker: cv2.Tracker, rect: cv2.typing.Rect) -> None:
        i = self.trackers.index(tracker)
        self.trackers_rect[i] = [int(n) for n in rect]

    def _stop_tracker(self, tracker: cv2.Tracker) -> None:
        i = self.trackers.index(tracker)
        self.trackers_rect.pop(i)
        self.trackers.pop(i)

    def handle(self, frame: Frame) -> Frame:
        # update trackers with new frame
        for tracker in self.trackers:
            ok, bbox = tracker.update(frame.raw)
            if ok:
                # still in frame, update
                self._update_tracker_rect(tracker, bbox)
            else:
                # gone missing
                # TODO remember for a while and try to rediscover
                self._stop_tracker(tracker)

        # find new trackers to add
        for contour in frame.contours:
            bbox = cv2.boundingRect(contour)  # x, y, w, h
            # has an existing tracker?
            if tracker := self._find_tracker(bbox):
                pass
            else:
                # tracker picked using https://broutonlab.com/blog/opencv-object-tracking/
                # tracker = cv2.TrackerKCF.create()
                tracker = cv2.TrackerCSRT.create()
                # tracker = cv2.legacy.TrackerMedianFlow.create()
                # tracker = cv2.legacy.TrackerBoosting.create()
                tracker.init(frame.raw, bbox)
                self.trackers.append(tracker)
                self.trackers_rect.append(bbox)

        return super().handle(frame)

    def _frame_to_write(self, frame: Frame) -> cv2.typing.MatLike:
        contours_img = frame.raw.copy()
        for tracker_rect in self.trackers_rect:
            cv2.rectangle(
                contours_img,
                (int(tracker_rect[0]), int(tracker_rect[1])),
                (
                    int(tracker_rect[0] + tracker_rect[2]),
                    int(tracker_rect[1] + tracker_rect[3]),
                ),
                (255, 0, 0),
            )
        return contours_img
