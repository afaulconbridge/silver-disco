from subprocess import Popen

import cv2
import ffmpeg
import numpy as np

from silver_disco.frame import Frame


class FrameProcessor:
    output_filename: str
    output_writer: Popen | None = None

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

    def handle(self, frame: Frame) -> Frame:
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

        return frame


class BackgroundSubtractorProcessor(FrameProcessor):
    _background_subtractor: cv2.BackgroundSubtractor | None = None

    @property
    def background_subtractor(self):
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
        kernel = np.ones((5, 5), np.uint8)
        frame.background_subtracted = cv2.morphologyEx(
            frame.background_subtracted, cv2.MORPH_OPEN, kernel
        )
        frame.background_subtracted = cv2.morphologyEx(
            frame.background_subtracted, cv2.MORPH_CLOSE, kernel
        )
        _, frame.background_subtracted = cv2.threshold(
            frame.background_subtracted, 100, 255, cv2.THRESH_BINARY
        )

        frame = super().handle(frame)
        return frame


class ContourProcessor(FrameProcessor):
    def _frame_to_write(self, frame: Frame) -> cv2.typing.MatLike:
        contours_img = frame.raw.copy()
        if frame.contours:
            cv2.drawContours(contours_img, frame.contours, -1, (255, 0, 0))
        return contours_img

    def handle(self, frame: Frame) -> Frame:
        frame.contours, hierarchy = cv2.findContours(
            frame.background_subtracted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # TODO size based on position in frame - smaller on horizon
        contour_min_size = (0.02 * frame.width) * (0.02 * frame.height)
        contour_max_size = (0.90 * frame.width) * (0.90 * frame.height)
        frame.contours = [
            contour
            for contour in frame.contours
            if cv2.contourArea(contour) > contour_min_size
            and cv2.contourArea(contour) < contour_max_size
        ]
        frame = super().handle(frame)
        return frame


class TrackingProcessor(FrameProcessor):
    trackers: list[cv2.Tracker]
    tracker_rects: list[cv2.Tracker]

    def __init__(self, output_filename: str = ""):
        super().__init__(output_filename)
        self.trackers = []
        self.tracker_rects = []

    def _find_tracker(self, x: int, y: int) -> cv2.Tracker | None:
        for tracker, tracker_rect in zip(
            self.trackers, self.tracker_rects, strict=False
        ):
            print(f"{x},{y} vs {tracker_rect}")

            if (
                x > tracker_rect[0]
                and x < tracker_rect[0] + tracker_rect[2]
                and y > tracker_rect[1]
                and y < tracker_rect[1] + tracker_rect[3]
            ):
                return tracker
        return None

    def _update_tracker_rect(self, tracker: cv2.Tracker, rect: cv2.typing.Rect) -> None:
        i = self.trackers.index(tracker)
        self.tracker_rects[i] = [int(n) for n in rect]

    def _stop_tracker(self, tracker) -> None:
        i = self.trackers.index(tracker)
        self.tracker_rects.pop(i)
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
            centre_x, centre_y = bbox[0] + (bbox[2] / 2), bbox[1] + (bbox[3] / 2)
            # has an existing tracker?
            # TODO faster data structure!
            # TODO based on bbox overlap, not centre containment
            if tracker := self._find_tracker(centre_x, centre_y):
                pass
            else:
                # tracker picked using https://broutonlab.com/blog/opencv-object-tracking/
                tracker = cv2.legacy.TrackerKCF.create()
                tracker = cv2.legacy.TrackerMedianFlow.create()
                tracker.init(frame.raw, bbox)
                self.trackers.append(tracker)
                self.tracker_rects.append(bbox)

        frame = super().handle(frame)
        return frame

    def _frame_to_write(self, frame: Frame) -> cv2.typing.MatLike:
        contours_img = frame.raw.copy()
        for tracker_rect in self.tracker_rects:
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
