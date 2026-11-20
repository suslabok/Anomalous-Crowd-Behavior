import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from vad.data.common import segments_from_labels


def plot_score_curves(videos, thr: float, path, max_videos: int = 12) -> None:
    """videos need: id, score, labels, alarm. Grey = annotated anomaly, red line = first alarm per segment."""
    vs = videos[:max_videos]
    cols = 2 if len(vs) > 1 else 1
    rows = int(np.ceil(len(vs) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(6.5 * cols, 2.2 * rows), squeeze=False)
    for ax, v in zip(axes.ravel(), vs):
        ax.plot(v["score"], lw=1)
        ax.axhline(thr, color="orange", ls="--", lw=0.8)
        for s, e in segments_from_labels(v["labels"]):
            ax.axvspan(s, e, color="grey", alpha=0.25)
            hits = np.flatnonzero(np.asarray(v["alarm"])[s:e + 1])
            if len(hits):
                ax.axvline(s + hits[0], color="red", lw=1)
        ax.set_title(str(v["id"]), fontsize=8)
        ax.tick_params(labelsize=7)
    for ax in axes.ravel()[len(vs):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_tradeoff(rows, path) -> None:
    """Delay and detection rate versus number of false alarms, one line per (smoothing, persistence)."""
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    groups = {}
    for r in rows:
        groups.setdefault((r["smooth"], r["min_consecutive"]), []).append(r)
    for (s, m), rs in sorted(groups.items()):
        rs = sorted(rs, key=lambda r: (r["n_false_alarms"], r["threshold"]))
        x = [r["n_false_alarms"] for r in rs]
        ax[0].plot(x, [r["delay_frames_mean"] for r in rs], marker="o", ms=3, lw=0.8, label=f"smooth {s}, persist {m}")
        ax[1].plot(x, [r["detection_rate"] for r in rs], marker="o", ms=3, lw=0.8)
    ax[0].set_xlabel("false alarms on normal footage (count)"); ax[0].set_ylabel("mean delay (frames)")
    ax[1].set_xlabel("false alarms on normal footage (count)"); ax[1].set_ylabel("detection rate")
    ax[0].legend(fontsize=6, ncol=2)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)
