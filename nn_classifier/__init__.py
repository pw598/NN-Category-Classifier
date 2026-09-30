"""Product category classification with neural networks and embeddings.

Same problem as the Naive Bayes repo -- predict a four-level category
path from a product description -- with three different model families
sharing one import, cleaning, and vectorization path:

  1. a multi-head network on BOW / TF-IDF vectors
  2. a multi-head network on learned dense embeddings (EmbeddingBag,
     optionally initialised from Word2Vec)
  3. classical estimators on pooled Word2Vec document vectors

Sharing the front half is the point. If each model cleaned its own
text, a difference in their scores would be unattributable, and the
comparison the repo exists to make would be worthless.

Typical use, from a notebook:

    from nn_classifier import RunConfig, data, cleaning, vectorization
    cfg = RunConfig()
    df = data.load_data(cfg.data)
    ...

Every submodule is imported lazily, on first attribute access. That is
not a micro-optimisation, it is about error messages. An eager
`from . import (...)` at the top of this file means one missing
third-party package -- scipy, torch, gensim -- breaks `import
nn_classifier` entirely, and breaks it with a *misleading* error:
CPython's import machinery swallows the ModuleNotFoundError raised
inside the subpackage and reports

    ImportError: cannot import name 'vectorization' from partially
    initialized module 'nn_classifier' (most likely due to a circular
    import)

which sends you looking for a cycle that does not exist. Loading on
demand means the real traceback reaches you, at the line that needed
the package. It also means the data-pull and sklearn paths work on a
machine with no torch installed.

If an import does fail, call `nn_classifier.diagnose()`: it imports
every submodule in turn and prints exactly which ones are broken and
why.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

__version__ = "0.1.0"

# Submodule -> the third-party packages it needs beyond numpy/pandas.
# Used only by diagnose(), to turn a failure into an instruction.
_SUBMODULE_REQUIREMENTS = {
    "ambiguity": (),
    "lookup": (),
    "artifacts": ("joblib",),
    "calibration": ("torch (temperature only)", "scikit-learn (isotonic only)"),
    "cleaning": ("nltk (leading-code stripping only)",),
    "config": (),
    "data": ("scikit-learn",),
    "deps": (),
    "evaluation": (),
    "hierarchy": (),
    "paths": (),
    "plotting": ("matplotlib",),
    "predict": ("joblib", "torch or scikit-learn, per bundle"),
    "sklearn_models": ("scikit-learn",),
    "torch_models": ("torch", "scipy"),
    "training": ("torch",),
    "vectorization": ("scikit-learn", "scipy", "gensim (Word2Vec only)"),
}

_SUBMODULES = frozenset(_SUBMODULE_REQUIREMENTS)

# Import name -> the name pip actually wants. Getting this wrong sends
# someone to `pip install sklearn`, which is a real but deprecated stub
# package that installs the wrong thing.
_PYPI_NAMES = {"sklearn": "scikit-learn", "cv2": "opencv-python", "PIL": "pillow"}

# Re-exported from .config, so `from nn_classifier import RunConfig` works
# without the caller knowing which module it lives in.
_CONFIG_EXPORTS = frozenset({
    "CalibrationConfig",
    "CleaningConfig",
    "DataConfig",
    "FilterConfig",
    "HierarchyConfig",
    "NNConfig",
    "RunConfig",
    "SklearnModelConfig",
    "SparseVectorizerConfig",
    "SplitConfig",
    "TokenVocabConfig",
    "Word2VecConfig",
    "load_config",
    "save_config",
})

_HIERARCHY_EXPORTS = frozenset({"LabelHierarchy", "predictions_frame"})

# Reachable as nnc.ensure_packages(...), because the first thing a
# notebook does on a fresh cluster should not require knowing which
# module the installer lives in.
_DEPS_EXPORTS = frozenset({"ensure_packages", "ensure_nltk_words", "is_available"})

__all__ = sorted(
    _SUBMODULES | _CONFIG_EXPORTS | _HIERARCHY_EXPORTS | _DEPS_EXPORTS | {"diagnose"}
)


def __getattr__(name: str):
    """PEP 562 lazy loading.

    Deliberately does not catch anything. A missing package should
    arrive as its own ModuleNotFoundError, naming the package, at the
    point of use.
    """
    if name in _SUBMODULES:
        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module

    if name in _CONFIG_EXPORTS:
        obj = getattr(importlib.import_module(".config", __name__), name)
        globals()[name] = obj
        return obj

    if name in _HIERARCHY_EXPORTS:
        obj = getattr(importlib.import_module(".hierarchy", __name__), name)
        globals()[name] = obj
        return obj

    if name in _DEPS_EXPORTS:
        obj = getattr(importlib.import_module(".deps", __name__), name)
        globals()[name] = obj
        return obj

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return __all__


def diagnose(verbose: bool = True):
    """Import every submodule and report which ones fail, and why.

    Written for the cluster, where the answer to 'why will this not
    import' is almost always one missing wheel and the default error
    points at the wrong thing.

        import nn_classifier
        nn_classifier.diagnose()

    Returns {submodule: None | exception}.
    """
    results = {}
    for name in sorted(_SUBMODULES):
        try:
            importlib.import_module(f".{name}", __name__)
            results[name] = None
        except Exception as exc:                        # noqa: BLE001
            results[name] = exc

    if verbose:
        width = max(len(n) for n in results)
        for name, exc in results.items():
            if exc is None:
                print(f"  ok    {name}")
            else:
                needs = _SUBMODULE_REQUIREMENTS.get(name, ())
                print(f"  FAIL  {name:<{width}}  {type(exc).__name__}: {exc}")
                if needs:
                    print(f"        {name} needs: {', '.join(needs)}")
        broken = [n for n, e in results.items() if e is not None]
        if broken:
            missing = sorted({
                _PYPI_NAMES.get(str(e).split("'")[1], str(e).split("'")[1])
                for e in results.values()
                if isinstance(e, ModuleNotFoundError) and "'" in str(e)
            })
            print(f"\n{len(broken)} of {len(results)} submodules failed: {broken}")
            if missing:
                print(f"Install the missing packages:  %pip install {' '.join(missing)}")
        else:
            print(f"\nall {len(results)} submodules import cleanly")
    return results


if TYPE_CHECKING:  # pragma: no cover - for editors and type checkers only
    from . import (  # noqa: F401
        artifacts,
        calibration,
        cleaning,
        config,
        data,
        deps,
        evaluation,
        hierarchy,
        paths,
        plotting,
        predict,
        sklearn_models,
        torch_models,
        training,
        vectorization,
    )
    from .config import (  # noqa: F401
        CalibrationConfig,
        CleaningConfig,
        DataConfig,
        FilterConfig,
        HierarchyConfig,
        NNConfig,
        RunConfig,
        SklearnModelConfig,
        SparseVectorizerConfig,
        SplitConfig,
        TokenVocabConfig,
        Word2VecConfig,
        load_config,
        save_config,
    )
    from .hierarchy import LabelHierarchy, predictions_frame  # noqa: F401
