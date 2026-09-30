"""Environment-aware default paths.

Data and outputs live next to the repo, on a cluster and on a laptop
alike, so nothing has to be edited by hand when the notebook moves.
Absolute cluster paths are deliberately not used: they look reasonable
until a workspace mount refuses the write, and that failure surfaces at
the end of a long run rather than the start.

Override either root with an environment variable if the defaults are
wrong for your setup:

    NN_CLASSIFIER_DATA_DIR
    NN_CLASSIFIER_OUTPUT_DIR
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Tuple

# <repo>/nn_classifier/paths.py -> <repo>
REPO_ROOT = Path(__file__).resolve().parent.parent


def on_databricks() -> bool:
    """True when running on a Databricks cluster with /dbfs mounted."""
    if os.environ.get("DATABRICKS_RUNTIME_VERSION"):
        return True
    return os.path.isdir("/dbfs")


def data_dir() -> Path:
    """Directory holding the golden set."""
    override = os.environ.get("NN_CLASSIFIER_DATA_DIR")
    if override:
        return Path(override)
    return REPO_ROOT / "data"


def output_dir() -> Path:
    """Directory to write run outputs into."""
    override = os.environ.get("NN_CLASSIFIER_OUTPUT_DIR")
    if override:
        return Path(override)
    return REPO_ROOT / "outputs"


def scratch_dir(name: str = "nn_classifier") -> Path:
    """Somewhere large, fast, and local to write multi-gigabyte files.

    `output_dir()` is the wrong place for them. On Databricks it is a
    `/Workspace` FUSE mount with a per-file size cap, and a write that
    exceeds it fails with

        OSError: [Errno 27] File too large

    which is EFBIG -- the filesystem refusing the file, not the disk
    running out. An out-of-fold probability array at 526,670 rows and
    3,473 categories is 7.3 GB and will not go there.

    `/local_disk0` is the cluster's ephemeral SSD: no such cap, and far
    faster. Ephemeral is the right trade for this content. An
    out-of-fold checkpoint exists to survive the Python process dying,
    which is what an OOM does; it does not need to survive the cluster
    terminating, and anything that does belongs in `output_dir()`
    where it is small enough to fit.

    Override with `NN_CLASSIFIER_SCRATCH_DIR`.

    Candidates are *probed*, not assumed. `/local_disk0` existing does
    not mean it is writable: on shared-access-mode and serverless
    clusters it is root-owned, and the mkdir fails with

        PermissionError: [Errno 13] Permission denied: '/local_disk0/tmp'

    This used to propagate out of `describe()` and kill the first cell
    of every notebook, including notebook 01, which has no use for
    scratch space at all. So each candidate is tried in turn -- created
    and written to -- and the first that works is returned.
    """
    import tempfile

    candidates: List[Tuple[Path, str]] = []
    override = os.environ.get("NN_CLASSIFIER_SCRATCH_DIR")
    if override:
        candidates.append((Path(override), "NN_CLASSIFIER_SCRATCH_DIR"))
    if os.path.isdir("/local_disk0"):
        candidates.append((Path("/local_disk0/tmp"), "cluster SSD"))
    candidates.append((Path(tempfile.gettempdir()), "system temp"))

    problems = []
    for base, source in candidates:
        target = base / name
        try:
            target.mkdir(parents=True, exist_ok=True)
            probe = target / ".write_probe"
            probe.touch()
            probe.unlink()
        except OSError as exc:
            problems.append(f"{base} ({source}): {type(exc).__name__}")
            continue

        if problems:
            # Say so. A run that quietly moved its multi-gigabyte
            # intermediates onto a small root volume, and a run that
            # got the SSD, look identical until one of them fills the
            # disk mid-fold.
            print(f"[paths] scratch: {'; '.join(problems)} -- using {target}")
        return target

    raise OSError(
        "No writable scratch directory. Tried: "
        + "; ".join(problems)
        + ". Set NN_CLASSIFIER_SCRATCH_DIR to somewhere writable."
    )


def free_bytes(path) -> Optional[int]:
    """Free space on the filesystem holding `path`, or None."""
    try:
        st = os.statvfs(str(path))
    except (OSError, AttributeError, ValueError):
        return None
    return st.f_bavail * st.f_frsize


def data_path(filename: str = "cleaned_data.txt") -> str:
    return str(data_dir() / filename)


def output_path(*parts: str) -> Path:
    p = output_dir().joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def torch_device(prefer_gpu: bool = True) -> str:
    """'cuda' when a GPU is visible and wanted, else 'cpu'."""
    if not prefer_gpu:
        return "cpu"
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def describe() -> str:
    """A status banner. Never raises.

    Every notebook opens with this, so anything it cannot determine
    has to degrade to a printed note rather than an exception.
    Scratch space is the case that matters: only notebook 05 writes
    there, but a PermissionError from probing it used to kill cell 1
    of notebook 01, which does not use it at all.
    """
    env = "Databricks" if on_databricks() else "local"

    try:
        scratch = scratch_dir()
        free = free_bytes(scratch)
        scratch_txt = str(scratch) + (
            "" if free is None else f"  ({free / 1024 ** 3:.0f} GB free)"
        )
    except OSError as exc:
        scratch_txt = (
            f"unavailable ({type(exc).__name__}) -- only notebook 05 needs it; "
            "set NN_CLASSIFIER_SCRATCH_DIR if you get there"
        )

    try:
        device = torch_device()
    except Exception as exc:                               # noqa: BLE001
        device = f"unknown ({type(exc).__name__})"

    return (
        f"environment: {env}\n"
        f"  data dir:   {data_dir()}\n"
        f"  output dir: {output_dir()}\n"
        f"  scratch:    {scratch_txt}\n"
        f"  device:     {device}"
    )
