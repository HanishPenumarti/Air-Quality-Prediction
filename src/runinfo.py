"""Run manifest: ties every saved result to code version, environment, data and configuration."""
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import numpy as np

from config import ROOT

PACKAGES = ["numpy", "pandas", "pyarrow", "scikit-learn", "xgboost", "joblib", "matplotlib", "plotly",
            "streamlit"]


def file_sha256(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def git_commit():
    try:
        sha = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True,
                             timeout=10).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain"], capture_output=True, text=True,
                               timeout=10).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return None


def environment():
    try:
        import psutil
        ram = round(psutil.virtual_memory().total / 2 ** 30, 1)
    except Exception:
        ram = None
    versions = {}
    for p in PACKAGES:
        try:
            versions[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            versions[p] = None
    return dict(python=sys.version.split()[0], platform=platform.platform(), cpu_count=os.cpu_count(),
                ram_gb=ram, packages=versions)


def json_safe(d):
    out = {}
    for k, v in d.items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            out[k] = v
        elif isinstance(v, (np.integer, np.floating)):
            out[k] = v.item()
        elif isinstance(v, (list, tuple)) and all(isinstance(x, (str, int, float, bool)) for x in v):
            out[k] = list(v)
    return out


def write_manifest(path, **extra):
    m = dict(run_id=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"), created=datetime.now(timezone.utc).isoformat(),
             git_commit=git_commit(), environment=environment(), **extra)
    Path(path).write_text(json.dumps(m, indent=1, default=str))
    return m
