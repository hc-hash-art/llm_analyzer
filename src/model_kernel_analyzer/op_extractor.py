from __future__ import annotations

import ast
from typing import Dict, List, Set

from .project_index import ProjectIndex
from .types import OpCall, SymbolRef


def extract_operator_calls(
    symbol_refs: Dict[str, SymbolRef],
    visited_nodes: Set[str],
    project_index: ProjectIndex,
) -> List[OpCall]:
    ops: List[OpCall] = []

    for sid in visited_nodes:
        sym = symbol_refs.get(sid)
        if not sym:
            continue
        fidx = project_index.by_file.get(sym.file_path)
        if not fidx or not fidx.tree:
            continue
        fn_node = _find_def_node(fidx.tree, sym)
        if not fn_node:
            continue

        for node in ast.walk(fn_node):
            if not isinstance(node, ast.Call):
                continue
            op_name = _to_op_name(node.func, fidx)
            if not op_name:
                continue
            expr = ast.unparse(node.func) if hasattr(ast, "unparse") else "<call>"
            ops.append(
                OpCall(
                    node_id=sid,
                    op_name=op_name,
                    call_expr=expr,
                    file_path=sym.file_path,
                    line=node.lineno,
                )
            )
    return ops


def _find_def_node(tree: ast.AST, sym: SymbolRef):
    want = sym.qualname.split(".")[-1]
    if sym.symbol_type not in ("function", "method"):
        return None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == want:
            if sym.symbol_type == "function":
                return node
            if sym.symbol_type == "method" and sym.class_name:
                cls = _find_enclosing_class(tree, node)
                if cls and cls.name == sym.class_name:
                    return node
    return None


def _find_enclosing_class(tree: ast.AST, target: ast.FunctionDef):
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and target in node.body:
            return node
    return None


def _attr_path(attr: ast.Attribute) -> str:
    parts: List[str] = []
    cur: ast.AST = attr
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
    return ".".join(reversed(parts))


def _to_op_name(func_node: ast.AST, fidx) -> str:
    # torch.xxx
    if isinstance(func_node, ast.Attribute):
        path = _attr_path(func_node)
        parts = path.split(".")
        if not parts:
            return ""
        root = parts[0]

        # 处理别名：import torch.nn.functional as F; F.relu(...)
        mapped_root = fidx.import_aliases.get(root, root)
        if mapped_root in ("torch", "torch.nn.functional", "torch.nn", "torch.cuda", "functorch"):
            return mapped_root + "." + ".".join(parts[1:]) if len(parts) > 1 else mapped_root

        # 直接识别 F / nn 常见别名
        if root in ("F", "nn"):
            alias_base = "torch.nn.functional" if root == "F" else "torch.nn"
            return alias_base + "." + ".".join(parts[1:]) if len(parts) > 1 else alias_base

        # 如果 root 在 from_imports 中映射到了 torch.xxx.xxx
        # 例如: from torch import softmax as S; S(...)
        return _to_op_name_from_from_imports(func_node, fidx)

    if isinstance(func_node, ast.Name):
        # from torch.nn import functional as F 的某些导入形式
        full = fidx.from_imports.get(func_node.id, "")
        if full.startswith("torch."):
            return full
    return ""


def _to_op_name_from_from_imports(func_node: ast.Attribute, fidx) -> str:
    # 仅做保守处理：如果 base/root 能在 from_imports 找到，返回 full+tail
    path = _attr_path(func_node)
    parts = path.split(".")
    if not parts:
        return ""
    root = parts[0]
    if root in fidx.from_imports:
        base = fidx.from_imports[root]
        return base + "." + ".".join(parts[1:]) if len(parts) > 1 else base
    return ""

