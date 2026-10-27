"""Fresh-process installed Kontext preprocessing audit; no pytest or conftest."""
from __future__ import annotations

import ast
import copy
import importlib.abc
import importlib.metadata
import json
from pathlib import Path
import sys
from types import SimpleNamespace


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


class NoModels(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"diffusers", "huggingface_hub", "pytest"} or fullname in {
            "backend.config", "backend.storage", "backend.main", "tests.conftest",
        }:
            raise AssertionError("preprocessing audit imported a model/application/test runtime: " + fullname)
        return None


sys.meta_path.insert(0, NoModels())

import numpy as np  # noqa: E402
import PIL  # noqa: E402
from PIL import Image  # noqa: E402
import torch  # noqa: E402

from backend import instruction_edit as edit  # noqa: E402

assert not torch.cuda.is_initialized(), "fresh CPU audit initialized CUDA during imports"
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
initialization_attempts = []


def forbid_cuda(*args, **kwargs):
    initialization_attempts.append(True)
    raise AssertionError("the CPU preprocessing audit attempted CUDA initialization")


torch.cuda._lazy_init = forbid_cuda
torch.cuda.init = forbid_cuda


def _tree(relative):
    distribution = importlib.metadata.distribution("diffusers")
    return ast.parse(Path(distribution.locate_file("diffusers") / relative).read_text(encoding="utf-8"))


def _class(tree, name):
    return next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name)


def _method(cls, name):
    return next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == name)


def audit(source_size):
    # Execute the complete installed PIL preprocessing path, never a pipeline
    # import, model, encoder, VAE or random-noise/device operation.
    processor_tree = _tree("image_processor.py")
    processor_class = _class(processor_tree, "VaeImageProcessor")
    methods = [copy.deepcopy(_method(processor_class, name)) for name in (
        "pil_to_numpy", "numpy_to_pt", "normalize", "resize", "get_default_height_width", "preprocess",
    )]
    valid_helpers = [copy.deepcopy(node) for node in processor_tree.body
                     if isinstance(node, ast.FunctionDef) and node.name in {"is_valid_image", "is_valid_image_imagelist"}]
    cls = ast.ClassDef(name="InstalledProcessor", bases=[], keywords=[], body=methods, decorator_list=[])
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
                              *valid_helpers, cls], type_ignores=[])
    namespace = {"np": np, "PIL": PIL, "torch": torch, "PIL_INTERPOLATION": {"lanczos": Image.Resampling.LANCZOS}}
    exec(compile(ast.fix_missing_locations(module), "installed-image-processing", "exec"), namespace)
    processor = namespace["InstalledProcessor"]()
    initializer = _method(processor_class, "__init__")
    names = [argument.arg for argument in initializer.args.args]
    defaults = dict(zip(names[-len(initializer.args.defaults):], initializer.args.defaults))
    config = {name: ast.literal_eval(value) for name, value in defaults.items()}
    assert config["do_resize"] and config["do_normalize"]
    assert not any(config[name] for name in ("do_binarize", "do_convert_rgb", "do_convert_grayscale"))
    assert config["resample"] == "lanczos" and config["reducing_gap"] is None
    config["vae_scale_factor"] = 16
    processor.config = SimpleNamespace(**config)
    pipeline = _class(_tree("pipelines/flux/pipeline_flux_kontext.py"), "FluxKontextPipeline")
    init = _method(pipeline, "__init__")
    image_processor_assignment = next(node for node in init.body if isinstance(node, ast.Assign)
                                      and ast.unparse(node.targets[0]) == "self.image_processor")
    assert ast.unparse(image_processor_assignment.value) == "VaeImageProcessor(vae_scale_factor=self.vae_scale_factor * 2)"
    branch = next(node for node in _method(pipeline, "__call__").body
                  if isinstance(node, ast.If) and "image is not None" in ast.unparse(node.test))
    original = Image.new("RGBA", source_size, (13, 77, 201, 0))
    original.getexif()[274] = 6
    prepared = edit.prepare_source(original)
    expected = np.asarray(prepared).astype(np.float32) / 255.0
    expected = 2.0 * expected.transpose(2, 0, 1)[None, ...] - 1.0
    environment = {"image": prepared, "torch": torch, "_auto_resize": False, "multiple_of": 16,
                   "self": SimpleNamespace(image_processor=processor, latent_channels=16)}
    exec(compile(ast.Module(body=[copy.deepcopy(branch)], type_ignores=[]), "installed-kontext-image-branch", "exec"), environment)
    tensor = environment["image"]
    assert tensor.device.type == "cpu" and tensor.dtype == torch.float32
    assert tensor.shape == (1, 3, prepared.height, prepared.width)
    assert np.array_equal(tensor.numpy(), expected)
    assert original.mode == "RGBA" and original.size == source_size
    assert not torch.cuda.is_initialized() and initialization_attempts == []
    assert "pytest" not in sys.modules and "diffusers" not in sys.modules
    print(json.dumps({"source_size": source_size, "tensor_shape": list(tensor.shape),
                      "cuda_initialized": False, "cuda_initialization_attempts": initialization_attempts}))


if __name__ == "__main__":
    audit((int(sys.argv[1]), int(sys.argv[2])))
