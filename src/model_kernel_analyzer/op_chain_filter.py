from __future__ import annotations

from pathlib import Path
from typing import FrozenSet, Optional

from .types import SymbolRef


_DEFAULT_SKIP_TOPLEVEL_MODULES: FrozenSet[str] = frozenset(
    {
        # 常见胶水代码
        "copy",
        "gc",
        "argparse",
        "os",
        "sys",
        "threading",
        "shutil",
        "platform",
        "readline",
        "json",
        "pickle",
        "logging",
        "subprocess",
        "inspect",
        "re",
        "weakref",
        "tempfile",
        "uuid",
        "datetime",
        "contextlib",
        "functools",
        "itertools",
        "collections",
        "warnings",
        "traceback",
        "linecache",
    }
)


def _norm_top(module: str) -> str:
    return (module or "").split(".")[0]


def _module_matches_prefix(module: str, pattern: str) -> bool:
    # pattern 支持 torch 或 transformers.trainer_utils 这种前缀
    if not pattern:
        return False
    if "." in pattern:
        return module == pattern or module.startswith(pattern + ".")
    return _norm_top(module) == pattern


def is_op_chain_node(
    callee_id: str,
    sym: Optional[SymbolRef],
    *,
    project_root: Optional[Path],
    extra_include_modules: FrozenSet[str],
    extra_exclude_modules: FrozenSet[str],
) -> bool:
    # external:xxx（默认只保留 torch/triton/functorch 等算子/执行相关）
    if callee_id.startswith("external:"):
        tail = callee_id[len("external:") :].lower()
        if ("torch" in tail) or ("triton" in tail) or ("functorch" in tail):
            return True
        # 只保留 transformers 的 forward
        if "transformers" in tail and sym is not None:
            q = (sym.qualname or "").lower()
            return q == "forward" or q.endswith(".forward")
        return False

    if sym is None:
        return False

    mod = sym.module or ""
    if any(_module_matches_prefix(mod, p) for p in extra_exclude_modules):
        return False
    if _norm_top(mod) in _DEFAULT_SKIP_TOPLEVEL_MODULES:
        return False

    if extra_include_modules:
        return any(_module_matches_prefix(mod, p) for p in extra_include_modules)

    if mod.startswith("torch") or mod.startswith("functorch"):
        return True
    if mod.startswith("triton"):
        return True

    # transformers：只保留 forward / *.forward
    if mod.startswith("transformers"):
        qn = (sym.qualname or "")
        return qn == "forward" or qn.endswith(".forward")

    # 工程内：仅保留 forward 相关（模块前向）
    if project_root and sym.file_path:
        try:
            proj_prefix = str(project_root.resolve()) + "/"
            fp = str(sym.file_path)
            if fp.startswith(proj_prefix):
                q = sym.qualname
                return q == "forward" or q.endswith(".forward")
        except Exception:
            pass

    return False

