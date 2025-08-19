# sitecustomize.py – auto-imported by CPython on every startup
import importlib, types
try:
    av = importlib.import_module("av")
    if not hasattr(av, "logging"):
        av.logging = types.SimpleNamespace(
            ERROR=0,
            WARNING=1,
            INFO=2,
            DEBUG=3,
            set_level=lambda *_, **__: None,
        )
except ModuleNotFoundError:
    pass   # let Python raise the usual ImportError later
