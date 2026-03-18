from __future__ import annotations

import inspect
from pathlib import Path
from typing import Iterable

from .types import OpCall


ATEN_GUESS_TABLE = {
    "matmul": "aten/src/ATen/native/LinearAlgebra.cpp",
    "mm": "aten/src/ATen/native/LinearAlgebra.cpp",
    "bmm": "aten/src/ATen/native/LinearAlgebra.cpp",
    "addmm": "aten/src/ATen/native/LinearAlgebra.cpp",
    "linear": "aten/src/ATen/native/Linear.cpp",
    "conv1d": "aten/src/ATen/native/Convolution.cpp",
    "conv2d": "aten/src/ATen/native/Convolution.cpp",
    "conv3d": "aten/src/ATen/native/Convolution.cpp",
    "layer_norm": "aten/src/ATen/native/layer_norm.cpp",
    "softmax": "aten/src/ATen/native/SoftMax.cpp",
    "gelu": "aten/src/ATen/native/Activation.cpp",
    "relu": "aten/src/ATen/native/Activation.cpp",
    "silu": "aten/src/ATen/native/Activation.cpp",
    "dropout": "aten/src/ATen/native/Dropout.cpp",
    "embedding": "aten/src/ATen/native/Embedding.cpp",
    "cat": "aten/src/ATen/native/TensorShape.cpp",
    "stack": "aten/src/ATen/native/TensorShape.cpp",
    "view": "aten/src/ATen/native/TensorShape.cpp",
    "reshape": "aten/src/ATen/native/TensorShape.cpp",
}


def map_ops_to_impl_paths(op_calls: Iterable[OpCall]) -> None:
    for op in op_calls:
        impl, note = _map_single(op.op_name)
        op.impl_path = impl
        op.mapper_note = note


def _map_single(op_name: str):
    if op_name.startswith("torch.nn.functional."):
        # 试图定位 Python functional 源码；如果定位不到，就 unknown
        attr = op_name.replace("torch.nn.functional.", "", 1)
        try:
            import torch.nn.functional as F

            obj = getattr(F, attr, None)
            if obj is not None:
                src = inspect.getsourcefile(obj)
                if src:
                    return str(Path(src).resolve()), "python functional source"
        except Exception:
            pass
        return "unknown", "functional op but source unresolved"

    if op_name.startswith("torch.nn."):
        # nn.Module 构造/forward wrapper 的 python 源码（不代表 kernel）
        attr = op_name.replace("torch.nn.", "", 1).split(".")[0]
        try:
            import torch.nn as nn

            obj = getattr(nn, attr, None)
            if obj is not None:
                src = inspect.getsourcefile(obj)
                if src:
                    return str(Path(src).resolve()), "nn module python source"
        except Exception:
            pass
        return "unknown", "nn op but source unresolved"

    if op_name.startswith("torch.ops.aten."):
        aten_name = op_name.replace("torch.ops.aten.", "", 1).split(".")[0]
        return _aten_impl_guess(aten_name), "from torch.ops.aten dispatch"

    if op_name.startswith("torch.cuda."):
        return "torch/cuda/*", "torch cuda runtime API (not ATen kernel)"

    if op_name.startswith("torch."):
        # torch tensor ops -> ATen dispatch（启发式）
        aten_name = op_name.replace("torch.", "", 1).split(".")[0]
        return _aten_impl_guess(aten_name), "likely dispatched to ATen native kernels"

    return "unknown", "unsupported op namespace"


def _aten_impl_guess(aten_name: str) -> str:
    if aten_name in ATEN_GUESS_TABLE:
        return ATEN_GUESS_TABLE[aten_name]
    return f"aten/src/ATen/native/* ({aten_name})"

