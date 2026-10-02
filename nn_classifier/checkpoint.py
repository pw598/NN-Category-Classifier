"""A training checkpoint that survives being killed halfway through a save.

`artifacts.save_bundle` is for a finished model. This is for an
unfinished one: everything `training.fit` needs to carry on from the
end of an epoch as though it had never stopped -- the weights, both
optimizers' moment buffers, the scheduler, the best weights seen so
far, the early-stopping counters and the random-number state.

Two constraints shape it, and both come from where it has to run.

**It has to outlive the cluster.** Training on CPU takes long enough
that a run is stopped in the evening and picked up the next day, on a
cluster that was terminated in between. So the checkpoint goes
somewhere durable, which on Databricks means the workspace mount --
and that mount refuses any single file over its size cap with

    OSError: [Errno 27] File too large

(`paths.scratch_dir` has the long version of that story). A sparse
first layer and its two SparseAdam moment buffers are each
`features x 512` floats, so one optimizer state is already past the
cap on a large vocabulary. Every part is therefore written in shards
of at most `max_file_mb`.

**It has to be all-or-nothing.** A checkpoint is only ever written
because the process might die, so it must tolerate dying during the
write. A new generation is written beside the old one under new file
names; `state.json` is then replaced in a single rename, and that
rename is the commit. Until it happens the previous generation is
intact and is what a resume will read; the superseded files are
removed only afterwards. A kill at any point leaves one complete
generation on disk.
"""

from __future__ import annotations

import io
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

STATE_NAME = "state.json"


class TrainingCheckpoint:
    """A directory holding one resumable training state.

    Parts are named blobs -- "model", "optim", "best" -- each pickled
    with `torch.save` and split into shards. A part can be carried
    forward unchanged from the previous generation, which is what keeps
    the best-weights copy from being rewritten every epoch when it only
    changes on the epochs that improve.
    """

    def __init__(self, directory, max_file_mb: int = 400):
        self.directory = Path(directory)
        self.max_bytes = int(max_file_mb) * 1024 * 1024
        if self.max_bytes <= 0:
            raise ValueError("max_file_mb must be positive.")

    # -- reading -----------------------------------------------------

    @property
    def state_path(self) -> Path:
        return self.directory / STATE_NAME

    def exists(self) -> bool:
        return self.state_path.is_file()

    def read_state(self) -> Dict[str, Any]:
        return json.loads(self.state_path.read_text(encoding="utf-8"))

    def load_part(self, name: str, map_location: Optional[str] = "cpu") -> Any:
        """One part of the committed generation, or None if it has none."""
        import torch

        entry = self.read_state()["parts"].get(name)
        if entry is None:
            return None
        buffer = io.BytesIO()
        for file_name in entry["files"]:
            path = self.directory / file_name
            if not path.is_file():
                raise FileNotFoundError(
                    f"Checkpoint {self.directory} is missing {file_name}, which "
                    f"its {STATE_NAME} lists. It cannot be resumed; delete the "
                    "directory to start that training run again."
                )
            buffer.write(path.read_bytes())
        if buffer.tell() != entry["bytes"]:
            raise OSError(
                f"Checkpoint part {name!r} in {self.directory} is "
                f"{buffer.tell():,} bytes, expected {entry['bytes']:,}. The "
                "files were truncated; delete the directory to start again."
            )
        buffer.seek(0)
        return torch.load(buffer, map_location=map_location, weights_only=False)

    # -- writing -----------------------------------------------------

    def _write_part(self, name: str, generation: int, obj: Any) -> Dict[str, Any]:
        import torch

        buffer = io.BytesIO()
        torch.save(obj, buffer)
        view = buffer.getbuffer()
        total = len(view)

        files = []
        n_shards = max(1, -(-total // self.max_bytes))
        for i in range(n_shards):
            file_name = f"{name}.g{generation:05d}.part{i:03d}"
            target = self.directory / file_name
            partial = self.directory / (file_name + ".tmp")
            with open(partial, "wb") as fh:
                fh.write(view[i * self.max_bytes:(i + 1) * self.max_bytes])
                fh.flush()
                try:
                    os.fsync(fh.fileno())
                except OSError:
                    # Some FUSE mounts do not implement fsync. The rename
                    # below is still the commit point for this shard.
                    pass
            os.replace(partial, target)
            files.append(file_name)
        del view
        return {"files": files, "bytes": total, "generation": generation}

    def save(
        self,
        state: Dict[str, Any],
        parts: Dict[str, Any],
        carry: Iterable[str] = (),
    ) -> Dict[str, Any]:
        """Write a new generation and commit it.

        `parts` are written fresh. `carry` names parts to keep from the
        committed generation without rewriting them. Anything in
        neither is dropped. Returns `{"bytes": ..., "seconds": ...}`
        for the caller to report -- on a slow mount the save can cost
        as much as an epoch, and that should be visible.
        """
        started = time.time()
        self.directory.mkdir(parents=True, exist_ok=True)

        previous = self.read_state() if self.exists() else {"generation": 0, "parts": {}}
        generation = int(previous.get("generation", 0)) + 1

        entries: Dict[str, Any] = {}
        for name in carry:
            if name in parts:
                continue
            if name in previous["parts"]:
                entries[name] = previous["parts"][name]
        written = 0
        for name, obj in parts.items():
            entries[name] = self._write_part(name, generation, obj)
            written += entries[name]["bytes"]

        record = dict(state)
        record["generation"] = generation
        record["parts"] = entries
        record["saved_at"] = time.time()

        partial = self.directory / (STATE_NAME + ".tmp")
        partial.write_text(json.dumps(record, indent=1), encoding="utf-8")
        os.replace(partial, self.state_path)          # <- the commit

        self._remove_unreferenced(entries)
        return {"bytes": written, "seconds": time.time() - started}

    def _remove_unreferenced(self, entries: Dict[str, Any]) -> None:
        """Delete shards no committed part points at.

        Covers the generation just superseded and anything a killed
        save left behind. Failures are ignored: a stray file costs
        disk space, and a checkpoint that raised here would have
        failed *after* successfully committing.
        """
        keep = {STATE_NAME}
        for entry in entries.values():
            keep.update(entry["files"])
        for path in self.directory.iterdir():
            if path.name in keep or not path.is_file():
                continue
            if ".part" in path.name or path.name.endswith(".tmp"):
                try:
                    path.unlink()
                except OSError:
                    pass

    def clear(self) -> None:
        """Remove the checkpoint entirely. A no-op if it is not there."""
        if not self.directory.is_dir():
            return
        for path in self.directory.iterdir():
            if path.is_file():
                try:
                    path.unlink()
                except OSError:
                    pass
        try:
            self.directory.rmdir()
        except OSError:
            pass
