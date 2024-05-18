from collections.abc import Iterable

import matplotlib.pyplot as plt

from silver_disco.frame import FrameStats


def plot_stats(frame_stats: Iterable[FrameStats], filename: str) -> None:
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

    ax1.plot([f.motion for f in frame_stats])
    ax2.plot([len(f.contours) for f in frame_stats])
    ax3.plot([len(f.trackers) for f in frame_stats])

    fig.savefig(filename)
