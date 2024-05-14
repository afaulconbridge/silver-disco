import contextlib
import sys
from collections.abc import Iterable

from silver_disco.frame import FileFrameSourceFFMPEG, FrameSource
from silver_disco.processor import (
    BackgroundSubtractorKNNProcessor,
    ContourProcessor,
    FrameProcessor,
    TrackingProcessor,
)


class Pipeline:
    source: FrameSource
    processors: tuple[FrameProcessor, ...]

    def __init__(self, source: FrameSource, processors: Iterable[FrameProcessor]):
        self.source = source
        self.processors = tuple(processors)

    def run(self) -> None:
        with contextlib.ExitStack() as stack:
            [stack.enter_context(processor) for processor in self.processors]
            for frame in self.source:
                frame_output = frame
                for processor in self.processors:
                    frame_input = frame_output
                    frame_output = processor.handle_time(frame_input)
                    processor.write_frame(frame_output)

        sys.stdout.write(f"Time (source) = {self.source.cumulative_time:.1f}s\n")
        for processor in self.processors:
            sys.stdout.write(
                f"Time ({processor.__class__.__name__}) = {processor.cumulative_time:.1f}s\n"
            )


def main() -> int:
    source = FileFrameSourceFFMPEG("clip_cam1_1714246175.211109-1714246190.397412.mp4")
    processors: list[FrameProcessor] = [
        FrameProcessor("out_raw.mp4"),
        BackgroundSubtractorKNNProcessor("out_background_subtracted.mp4"),
        ContourProcessor("out_contours.mp4"),
        TrackingProcessor("out_tracking.mp4"),
    ]

    pipeline = Pipeline(source, processors)
    pipeline.run()

    return 0
