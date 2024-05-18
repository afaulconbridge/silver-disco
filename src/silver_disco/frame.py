import time
from dataclasses import dataclass, field
from subprocess import Popen

import cv2
import ffmpeg
import numpy as np


@dataclass
class Frame:
    raw: cv2.typing.MatLike
    background: cv2.typing.MatLike | None = None
    background_subtracted: cv2.typing.MatLike | None = None
    contours: list[cv2.typing.MatLike] = field(default_factory=list)
    tracked: dict[int, tuple[int, int, int, int]] = field(default_factory=dict)

    @property
    def width(self) -> float:
        return self.raw.shape[1]

    @property
    def height(self) -> float:
        return self.raw.shape[0]


class FrameSource:
    def __init__(self):
        pass

    def __iter__(self):
        return self

    def __next__(self) -> Frame:
        raise NotImplementedError


class FileFrameSourceCV2(FrameSource):
    filename: str
    cap: cv2.VideoCapture | None = None

    def __init__(self, filename: str):
        super()
        self.filename = filename

    def __next__(self) -> Frame:
        if not self.cap:
            self.cap = cv2.VideoCapture(self.filename)

        ret, frame = self.cap.read()
        if ret:
            return Frame(raw=frame)
        else:
            self.cap.release()
            self.cap = None
            raise StopIteration


class FileFrameSourceFFMPEG(FrameSource):
    filename: str
    reader: Popen | None = None
    width: int | None = None
    height: int | None = None
    cumulative_time: int = 0

    def __init__(self, filename: str):
        super()
        self.filename = filename

    def _detect_width_height(self) -> None:
        probe = ffmpeg.probe(self.filename)
        video_stream = next(
            (stream for stream in probe["streams"] if stream["codec_type"] == "video"),
            None,
        )
        self.width = int(video_stream["width"])
        self.height = int(video_stream["height"])

    def __next__(self) -> Frame:
        start = time.time()

        if not self.reader:
            self._detect_width_height()
            self.reader = (
                ffmpeg.input(self.filename)
                .output("pipe:", format="rawvideo", pix_fmt="rgb24")
                .run_async(pipe_stdout=True, quiet=True)
            )

        if self.reader.poll() is not None:
            raise StopIteration
        in_bytes = self.reader.stdout.read(self.width * self.height * 3)

        if not in_bytes:
            self.reader.wait()
            self.reader = None
            self.width = None
            self.height = None
            raise StopIteration

        in_frame = np.frombuffer(in_bytes, np.uint8).reshape(
            [self.height, self.width, 3]
        )
        frame = Frame(raw=in_frame)

        end = time.time()
        self.cumulative_time += end - start

        return frame


@dataclass
class FrameStats:
    motion: float
    contours: list[tuple[int, int, int, int]]
    trackers: dict[str, tuple[int, int, int, int]]
