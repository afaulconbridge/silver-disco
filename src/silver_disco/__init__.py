import contextlib
import sys
from collections.abc import Iterable

import cv2
import numpy as np

from silver_disco.frame import FileFrameSourceFFMPEG, Frame, FrameSource, FrameStats
from silver_disco.processor import (
    BackgroundSubtractorKNNProcessor,
    BackgroundSubtractorMOG2Processor,
    ContourProcessor,
    FrameProcessor,
    TrackingProcessor,
    TrackingSimpleProcessor,
)
from silver_disco.statsplot import plot_stats


class Pipeline:
    source: FrameSource
    processors: tuple[FrameProcessor, ...]
    frame_stats = list[FrameStats]

    def __init__(self, source: FrameSource, processors: Iterable[FrameProcessor]):
        self.source = source
        self.processors = tuple(processors)
        self.frame_stats = []

    def _get_stats(self, frame: Frame) -> FrameStats:
        assert frame.background_subtracted is not None
        motion_white_count = np.count_nonzero(frame.background_subtracted)
        motion_proportion = motion_white_count / (frame.width * frame.height)

        contours = [tuple(cv2.boundingRect(contour)) for contour in frame.contours]

        trackers = frame.tracked

        return FrameStats(motion_proportion, contours, trackers)

    def run(self) -> None:
        with contextlib.ExitStack() as stack:
            [stack.enter_context(processor) for processor in self.processors]
            for frame in self.source:
                frame_output = frame
                for processor in self.processors:
                    frame_input = frame_output
                    frame_output = processor.handle_time(frame_input)
                    processor.write_frame(frame_output)
                frame_stats = self._get_stats(frame_output)
                self.frame_stats.append(frame_stats)
                # replace with plotting at end
                sys.stdout.write(repr(frame_stats))
                sys.stdout.write("\n")

        sys.stdout.write(f"Time (source) = {self.source.cumulative_time:.1f}s\n")
        for processor in self.processors:
            sys.stdout.write(
                f"Time ({processor.__class__.__name__}) = {processor.cumulative_time:.1f}s\n"
            )

        plot_stats(self.frame_stats, "stats.png")


def main() -> int:
    # filename = "cam1_2024_04_28_20_55__2024_04_28_20_58_edited.mp4"
    # filename = "clip_cam1_1715799284.060785-1715799460.771761.mp4"
    filename = "clip_cam1_1714246175.211109-1714246190.397412.mp4"
    source = FileFrameSourceFFMPEG(filename)
    processors: list[FrameProcessor] = [
        FrameProcessor("out_raw.mp4"),
        BackgroundSubtractorKNNProcessor("out_background_subtracted.mp4"),
        # BackgroundSubtractorMOG2Processor("out_background_subtracted.mp4"),
        ContourProcessor("out_contours.mp4"),
        # TrackingProcessor("out_tracking.mp4"),
        TrackingSimpleProcessor("out_tracking.mp4"),
    ]

    pipeline = Pipeline(source, processors)
    pipeline.run()

    return 0
