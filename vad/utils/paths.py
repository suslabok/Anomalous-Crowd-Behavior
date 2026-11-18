"""Colab-aware path management.

Persistent (Drive): checkpoints, logs, results, dataset archives (tar).
Ephemeral (local disk): raw + processed datasets, flow cache. Rebuilt each session
from the tar archives on Drive (see vad/data/prepare.py pack/unpack).
"""
import os
import sys
from dataclasses import dataclass
from pathlib import Path


def in_colab() -> bool:
    return "google.colab" in sys.modules or "COLAB_RELEASE_TAG" in os.environ


def mount_drive(mount_point: str = "/content/drive") -> None:
    if not os.path.ismount(mount_point):
        from google.colab import drive  # type: ignore
        drive.mount(mount_point)


@dataclass
class Paths:
    persist: Path       # survives session end
    local: Path         # fast, wiped on session end

    @property
    def checkpoints(self) -> Path: return self.persist / "checkpoints"
    @property
    def logs(self) -> Path: return self.persist / "logs"
    @property
    def results(self) -> Path: return self.persist / "results"
    @property
    def archives(self) -> Path: return self.persist / "archives"   # tar/zip datasets on Drive
    @property
    def raw_data(self) -> Path: return self.local / "raw"
    @property
    def processed(self) -> Path: return self.local / "processed"
    @property
    def flow(self) -> Path: return self.local / "flow"

    def make(self) -> "Paths":
        for p in (self.checkpoints, self.logs, self.results, self.archives,
                  self.raw_data, self.processed, self.flow):
            p.mkdir(parents=True, exist_ok=True)
        return self


def get_paths(cfg) -> Paths:
    if in_colab():
        mount_drive()
        persist, local = Path(cfg.paths.drive_root), Path(cfg.paths.local_root)
    else:
        persist, local = Path("outputs"), Path("outputs/local")
    return Paths(persist=persist, local=local).make()
