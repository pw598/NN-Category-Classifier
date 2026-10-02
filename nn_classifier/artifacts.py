"""Saving a run so it can be loaded, audited, and reproduced.

A bundle is a directory, not a single pickle. The reason is that the
pieces have different lifetimes and different risks: torch weights
need `torch.load`, the cleaner and vectorizer are joblib pickles that
break across library versions, and the config and metrics are text
that should stay readable when everything else stops loading.

Every bundle carries a manifest with a fingerprint of what went into
it. `load_bundle` checks it. The failure this prevents is the
expensive one -- a vectorizer from one run loaded next to weights from
another, which does not raise, and produces confident nonsense.
"""

from __future__ import annotations

import json
import platform
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

MANIFEST_NAME = "manifest.json"
BUNDLE_VERSION = 1


def _jsonable(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return {k: _jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def _config_from_manifest(manifest: Dict[str, Any]) -> Any:
    """The saved config, rebuilt from the manifest's JSON.

    For bundles written before `config.joblib` existed. The sections
    (`data`, `cleaning`, ...) come back as attribute-access objects,
    which is all `predict` asks of a config; what is inside each
    section is left as plain JSON values, so `marker_columns` is still
    the dict of dicts the cleaning step takes.
    """
    from types import SimpleNamespace

    raw = manifest.get("config")
    if not isinstance(raw, dict):
        return None
    return SimpleNamespace(**{
        section: (SimpleNamespace(**values) if isinstance(values, dict) else values)
        for section, values in raw.items()
    })


def save_bundle(
    directory: str | Path,
    model: Any = None,
    model_kind: str = "torch",
    hierarchy: Any = None,
    cleaner: Any = None,
    vectorizer: Any = None,
    vocabulary: Any = None,
    word_vectors: Any = None,
    calibrators: Any = None,
    thresholds: Optional[Dict[int, float]] = None,
    per_level_thresholds: Optional[Dict[int, float]] = None,
    hierarchy_lookup: Optional[pd.DataFrame] = None,
    config: Any = None,
    history: Any = None,
    metrics: Optional[Dict[str, Any]] = None,
    n_training_rows: Optional[int] = None,
    notes: str = "",
    compress: int = 3,
) -> Path:
    """Write every piece a prediction needs into one directory.

    Only the pieces that exist are written, so the same function
    serves a torch run (weights + vectorizer or vocabulary) and an
    sklearn run (estimators + word vectors) without a branch at the
    call site.
    """
    import joblib

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    written: List[str] = []

    if model is not None:
        if model_kind == "torch":
            import torch

            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "class_name": type(model).__name__,
                    "init_kwargs": model.init_kwargs(),
                },
                directory / "model.pt",
            )
            written.append("model.pt")
        else:
            joblib.dump(model, directory / "model.joblib", compress=compress)
            written.append("model.joblib")

    for name, obj in [
        ("hierarchy", hierarchy),
        ("cleaner", cleaner),
        ("vectorizer", vectorizer),
        ("vocabulary", vocabulary),
        ("word_vectors", word_vectors),
        ("calibrators", calibrators),
    ]:
        if obj is not None:
            joblib.dump(obj, directory / f"{name}.joblib", compress=compress)
            written.append(f"{name}.joblib")

    # The config as an object, beside the readable copy in the manifest.
    # Scoring reads it to learn how the model has to be fed -- whether
    # the descriptions it was trained on carried a vendor marker or code
    # markers -- and must attach the same ones.
    if config is not None:
        joblib.dump(config, directory / "config.joblib", compress=compress)
        written.append("config.joblib")

    if hierarchy_lookup is not None:
        hierarchy_lookup.to_csv(directory / "hierarchy_lookup.csv", index=False)
        written.append("hierarchy_lookup.csv")

    if history is not None:
        frame = history.to_frame() if hasattr(history, "to_frame") else pd.DataFrame(history)
        frame.to_csv(directory / "history.csv", index=False)
        written.append("history.csv")

    manifest = {
        "bundle_version": BUNDLE_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_kind": model_kind,
        "model_class": type(model).__name__ if model is not None else None,
        "files": sorted(written),
        "level_columns": list(getattr(hierarchy, "level_columns", []) or []),
        "n_classes_per_level": [
            hierarchy.n_classes(i) for i in range(hierarchy.n_levels)
        ] if hierarchy is not None else [],
        "n_paths": int(getattr(hierarchy, "n_paths", 0) or 0),
        "n_training_rows": n_training_rows,
        # Two sets, because they are not interchangeable.
        #
        #   thresholds            joint prefix mass. Saturates on this
        #                         taxonomy -- depth 3 came back
        #                         unreachable on the vendor model --
        #                         so it is kept for reference only.
        #   per_level_thresholds  each level's own calibrated
        #                         confidence. The operative set:
        #                         98.8% coverage against the joint
        #                         version's holes.
        #
        # Scoring prefers per_level_thresholds when present. Bundles
        # saved before this existed have only the joint set and still
        # load; predict warns when it has to fall back.
        "thresholds": {str(k): (None if v == float("inf") else float(v))
                       for k, v in (thresholds or {}).items()},
        "per_level_thresholds": {str(k): (None if v == float("inf") else float(v))
                                 for k, v in (per_level_thresholds or {}).items()},
        "metrics": _jsonable(metrics or {}),
        "config": _jsonable(config) if config is not None else None,
        "notes": notes,
        "python": platform.python_version(),
    }
    (directory / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2))

    print(f"[artifacts] wrote {len(written) + 1} files to {directory}")
    return directory


def load_bundle(
    directory: str | Path,
    device: Optional[str] = None,
    verify: bool = True,
) -> Dict[str, Any]:
    """Load a bundle back, checking it against its own manifest.

    `verify=True` fails on a missing file rather than returning a
    half-loaded bundle. A missing vectorizer is not something to
    discover at the first prediction.
    """
    import joblib

    directory = Path(directory)
    manifest_path = directory / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"No {MANIFEST_NAME} in {directory}.")
    manifest = json.loads(manifest_path.read_text())

    if verify:
        missing = [f for f in manifest["files"] if not (directory / f).is_file()]
        if missing:
            raise FileNotFoundError(
                f"Bundle {directory} is incomplete; the manifest lists "
                f"{missing}, which are not present."
            )

    out: Dict[str, Any] = {"manifest": manifest, "directory": directory}

    for name in ("hierarchy", "cleaner", "vectorizer", "vocabulary",
                 "word_vectors", "calibrators"):
        path = directory / f"{name}.joblib"
        out[name] = joblib.load(path) if path.is_file() else None

    # The config the model was trained under. `predict` and the scoring
    # notebooks read `bundle["config"]` to decide whether markers must be
    # attached before scoring. This key used to be absent altogether, so
    # that check always came back "no markers" and a marker-trained
    # model was scored on descriptions missing the tokens it relies on:
    # no error, and a worse answer on every row.
    #
    # A pickle that will not load -- the config classes have moved on
    # since it was written -- falls back to the manifest rather than to
    # None, because None is exactly the silent failure described above.
    config_path = directory / "config.joblib"
    config = None
    if config_path.is_file():
        try:
            config = joblib.load(config_path)
        except Exception as exc:                           # noqa: BLE001
            print(f"[artifacts] config.joblib would not load "
                  f"({type(exc).__name__}); using the manifest's copy")
    out["config"] = config if config is not None else _config_from_manifest(manifest)

    lookup_path = directory / "hierarchy_lookup.csv"
    out["hierarchy_lookup"] = (
        pd.read_csv(lookup_path, dtype="str") if lookup_path.is_file() else None
    )

    def _thresholds(key):
        raw = manifest.get(key) or {}
        return {int(k): (float("inf") if v is None else float(v))
                for k, v in raw.items()}

    out["thresholds"] = _thresholds("thresholds")
    out["per_level_thresholds"] = _thresholds("per_level_thresholds")

    if (directory / "model.pt").is_file():
        import torch

        from .torch_models import MODEL_CLASSES

        payload = torch.load(
            directory / "model.pt", map_location=device or "cpu", weights_only=False
        )
        klass = MODEL_CLASSES[payload["class_name"]]
        model = klass(**payload["init_kwargs"])
        model.load_state_dict(payload["state_dict"])
        model.eval()
        out["model"] = model
    elif (directory / "model.joblib").is_file():
        out["model"] = joblib.load(directory / "model.joblib")
    else:
        out["model"] = None

    print(
        f"[artifacts] loaded {manifest.get('model_class')} from {directory} "
        f"(created {manifest.get('created_utc')})"
    )
    return out


def describe_bundle(directory: str | Path) -> str:
    """The manifest as readable text, without loading any weights."""
    directory = Path(directory)
    manifest = json.loads((directory / MANIFEST_NAME).read_text())
    lines = [
        f"bundle:        {directory}",
        f"created:       {manifest.get('created_utc')}",
        f"model:         {manifest.get('model_class')} ({manifest.get('model_kind')})",
        f"levels:        {manifest.get('level_columns')}",
        f"classes:       {manifest.get('n_classes_per_level')}",
        f"valid paths:   {manifest.get('n_paths'):,}",
        f"training rows: {manifest.get('n_training_rows')}",
        f"thresholds:    {manifest.get('thresholds')}  (joint prefix)",
        f"per-level:     {manifest.get('per_level_thresholds') or 'absent'}"
        + ("  <- operative" if manifest.get("per_level_thresholds") else ""),
    ]
    if manifest.get("notes"):
        lines.append(f"notes:         {manifest['notes']}")
    for name, value in (manifest.get("metrics") or {}).items():
        lines.append(f"  {name}: {value}")
    return "\n".join(lines)


def attach_names(
    frame: pd.DataFrame,
    lookup: pd.DataFrame,
    level_columns: Sequence[str],
    name_columns: Sequence[str],
) -> pd.DataFrame:
    """Put the category names back onto a frame of predicted IDs.

    A left join on the full path, not per level -- names are only
    unique within a parent, so joining level by level would attach the
    wrong name wherever one is reused elsewhere in the tree. That is
    the whole reason training happens on IDs.
    """
    level_columns = list(level_columns)
    name_columns = list(name_columns)
    pred_cols = [f"Predicted {c}" for c in level_columns]
    have = [c for c in pred_cols if c in frame.columns]
    if not have or lookup is None:
        return frame

    keys = [c.replace("Predicted ", "") for c in have]
    right = lookup[keys + [n for n in name_columns if n in lookup.columns]].drop_duplicates()
    renames = {k: f"Predicted {k}" for k in keys}
    renames.update({
        n: f"Predicted {n}" for n in name_columns if n in right.columns
    })
    right = right.rename(columns=renames)

    return frame.merge(right, on=have, how="left")
