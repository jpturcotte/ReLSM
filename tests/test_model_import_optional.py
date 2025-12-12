import importlib
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch", reason="model import smoke test requires torch installed")


def test_model_imports_without_mamba(monkeypatch):
    """Importing model should not crash when mamba_ssm is absent."""

    original_find_spec = importlib.util.find_spec

    def fake_find_spec(name, *args, **kwargs):  # noqa: ANN001 - pytest helper
        if name.startswith("mamba_ssm"):
            raise ModuleNotFoundError("No module named 'mamba_ssm'")
        return original_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)

    # Ensure module-level import logic reruns with the patched finder.
    sys.modules.pop("model", None)

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    module = importlib.import_module("model")

    assert module is not None
