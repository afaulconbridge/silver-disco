import contextlib

from silver_disco.frame import FileFrameSourceFFMPEG
from silver_disco.processor import (
    BackgroundSubtractorProcessor,
    ContourProcessor,
    FrameProcessor,
    TrackingProcessor,
)


def main() -> int:
    source = FileFrameSourceFFMPEG("clip_cam1_1714246175.211109-1714246190.397412.mp4")
    processors = [
        FrameProcessor("out_raw.mp4"),
        BackgroundSubtractorProcessor("out_background_subtracted.mp4"),
        ContourProcessor("out_contours.mp4"),
        TrackingProcessor("out_tracking.mp4"),
    ]
    with contextlib.ExitStack() as stack:
        [stack.enter_context(processor) for processor in processors]
        for frame in source:
            for processor in processors:
                frame = processor.handle(frame)

    return 0
