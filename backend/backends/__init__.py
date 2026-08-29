from .base import GenerationBackend, ProgressFn

__all__ = ["GenerationBackend", "ProgressFn", "get_backend"]


def get_backend(name: str = "diffusers") -> GenerationBackend:
    """Resolve a backend by name. Add ComfyUI/remote engines here."""
    if name == "diffusers":
        from .diffusers_backend import DiffusersBackend

        return DiffusersBackend()
    raise ValueError(f"unknown backend: {name!r}")
