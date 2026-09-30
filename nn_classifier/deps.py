"""Installing what a notebook needs, and only what it is missing.

Databricks runtimes vary in what they ship. A standard runtime has no
torch; an ML runtime does. nltk and gensim are on neither. So a
notebook that assumes any of them fails halfway through on some
clusters and not others, which is the worst kind of failure to debug.

`ensure_packages` checks first and installs only what is absent. The
check matters: pip on an already-satisfied requirement still resolves
the dependency graph, which on a cluster is tens of seconds of nothing,
and `%pip install` on Databricks additionally *restarts the Python
interpreter* — silently destroying every variable the notebook has
built up. Running it unconditionally at the top of a notebook is a
habit worth breaking.

This installs with `subprocess` rather than `%pip` deliberately. It
does not restart the interpreter, so it is safe to call from any cell
rather than only the first. The trade-off is that the install lands on
the driver only, which is correct here — every model in this repo is
trained on the driver, and the one thing that touches executors (the
Spark Word2Vec backend) needs its packages installed cluster-wide
anyway, via a cluster library rather than from a notebook.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import subprocess
import sys
from typing import Dict, List, Sequence

# Set this to anything truthy to stop the package installing on your
# behalf. Worth having: a shared cluster where someone else controls the
# environment is a place where a library quietly installing itself is
# rude at best, and `auto_install` is called from module scope in
# torch_models and training, where a surprise is least welcome.
NO_AUTO_INSTALL_ENV = "NN_CLASSIFIER_NO_AUTO_INSTALL"

# pip name -> import name, where they differ.
_IMPORT_NAMES: Dict[str, str] = {
    "scikit-learn": "sklearn",
    "pillow": "PIL",
    "opencv-python": "cv2",
    "pyyaml": "yaml",
}


def _import_name(requirement: str) -> str:
    """'scikit-learn>=1.3' -> 'sklearn'."""
    base = requirement
    for sep in ("[", "==", ">=", "<=", "~=", ">", "<", "!="):
        base = base.split(sep)[0]
    base = base.strip()
    return _IMPORT_NAMES.get(base, base.replace("-", "_"))


def is_available(requirement: str) -> bool:
    """True when the package can already be imported."""
    try:
        return importlib.util.find_spec(_import_name(requirement)) is not None
    except (ImportError, ValueError):
        return False


def ensure_packages(
    *requirements: str,
    quiet: bool = True,
    verbose: bool = True,
    index_url: str | None = None,
) -> List[str]:
    """Install only the requirements that are not already importable.

        nnc.ensure_packages("torch", "nltk")

    Returns the list of things it actually installed, which is empty on
    every run after the first. Accepts pip syntax, so
    `ensure_packages("torch>=2.0")` works — though note the check is on
    *presence*, not version: an existing but too-old torch will not be
    upgraded. That is deliberate. Silently upgrading a cluster's torch
    from under whatever else is running on it is not a thing a notebook
    should do on its own.

    `index_url` is there for torch specifically. The CPU-only wheel is
    a fraction of the size of the default one, and on a cluster with no
    GPU it is the one you want:

        nnc.ensure_packages(
            "torch", index_url="https://download.pytorch.org/whl/cpu")
    """
    missing = [r for r in requirements if not is_available(r)]
    present = [r for r in requirements if r not in missing]

    if verbose and present:
        print(f"[deps] already available: {', '.join(present)}")
    if not missing:
        return []

    if verbose:
        print(f"[deps] installing: {', '.join(missing)}")
    cmd = [sys.executable, "-m", "pip", "install"]
    if quiet:
        cmd.append("-q")
    if index_url:
        cmd += ["--index-url", index_url]
    cmd += list(missing)

    subprocess.check_call(cmd)

    # Without this, a package installed during this session can stay
    # invisible to import: the finders have already cached the state of
    # site-packages from before it existed.
    importlib.invalidate_caches()

    still_missing = [r for r in missing if not is_available(r)]
    if still_missing:
        raise ImportError(
            f"pip reported success but {still_missing} still cannot be "
            "imported. On Databricks this usually means the install went to "
            "a different interpreter than the notebook is using; try "
            f"`%pip install {' '.join(still_missing)}` instead, which will "
            "restart Python and clear the notebook's state."
        )

    if verbose:
        print(f"[deps] installed: {', '.join(missing)}")
    return missing


def auto_install(*requirements: str, purpose: str = "") -> List[str]:
    """Install a missing requirement at the point the code needs it.

    This is what the package calls itself, rather than asking every
    notebook to remember a setup cell. The check is cheap
    (`importlib.util.find_spec`), so the steady state is a no-op and
    nothing is printed; only a real install announces itself, because a
    cell that appears to hang for two minutes downloading torch should
    say why.

    Honours `NN_CLASSIFIER_NO_AUTO_INSTALL` for anyone who would rather
    manage the environment themselves. When set, the import simply fails
    with its own ModuleNotFoundError, which is the honest outcome.
    """
    if os.environ.get(NO_AUTO_INSTALL_ENV):
        return []

    missing = [r for r in requirements if not is_available(r)]
    if not missing:
        return []

    reason = f" (needed by {purpose})" if purpose else ""
    print(f"[deps] {', '.join(missing)} not installed{reason}; installing now")
    try:
        return ensure_packages(*missing, verbose=False)
    except Exception as exc:                            # noqa: BLE001
        # Never let a failed convenience install mask the real problem.
        # The import that follows raises ModuleNotFoundError, which is a
        # far clearer thing to read than a pip traceback.
        print(f"[deps] automatic install failed ({exc}); "
              f"install manually:  %pip install {' '.join(missing)}")
        return []


def ensure_nltk_words(verbose: bool = True) -> None:
    """Make sure the NLTK word corpus is present as well as the package.

    Two separate things, and the second is easy to forget: `pip install
    nltk` gives you the library but no data. The corpus download needs
    network access and a writable NLTK data directory, neither of which
    is guaranteed on a cluster — which is why `build_cleaner` lets you
    pass your own vocabulary set instead, and why the resolved word
    list is pickled into the cleaner so scoring never has to repeat
    this.
    """
    ensure_packages("nltk", verbose=verbose)
    import nltk

    try:
        nltk.data.find("corpora/words")
        if verbose:
            print("[deps] nltk 'words' corpus already available")
    except LookupError:
        if verbose:
            print("[deps] downloading the nltk 'words' corpus")
        nltk.download("words", quiet=True)


def report(requirements: Sequence[str] = ()) -> None:
    """Print what is and is not importable, without installing anything."""
    checks = list(requirements) or [
        "numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "joblib",
        "torch", "gensim", "nltk", "pyspark",
    ]
    width = max(len(c) for c in checks)
    for requirement in checks:
        mark = "ok     " if is_available(requirement) else "ABSENT "
        print(f"  {mark} {requirement:<{width}}")
