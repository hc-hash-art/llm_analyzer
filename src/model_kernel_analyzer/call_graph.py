from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from .project_index import FileIndex, ProjectIndex
from .types import CallEdge, ClassEdge, SymbolRef


@dataclass
class GraphBuildOutput:
    symbol_refs: Dict[str, SymbolRef]
    call_edges: List[CallEdge]
    class_edges: List[ClassEdge]
    visited_nodes: Set[str]
    entry_symbol_ids: List[str]
    entry_fallback_used: bool


def build_call_and_class_graph(
    project_index: ProjectIndex,
    entry_files: List[Path],
    *,
    max_depth: int = 20,
    follow_external_imports: bool = True,
    max_external_resolve: int = 200,
    entry_mode: str = "forward",
    forward_fallback: bool = True,
) -> GraphBuildOutput:
    symbol_refs: Dict[str, SymbolRef] = dict(project_index.symbol_by_id)
    call_edges: List[CallEdge] = []
    class_edges: List[ClassEdge] = []
    visited: Set[str] = set()
    external_count = 0

    entry_symbol_ids, entry_fallback_used = _select_entry_nodes(
        project_index,
        entry_files,
        entry_mode=entry_mode,
        forward_fallback=forward_fallback,
    )

    # 简化：用 DFS stack；visited 控制重复展开
    stack: List[Tuple[str, int]] = [(sid, 0) for sid in entry_symbol_ids]

    # 基于 __init__ 提取 self.xxx = SomeModule(...) 的类型信息（用于 self.xxx(...) -> SomeModule.forward）
    class_field_type_map = _build_class_field_type_map(project_index)

    while stack:
        node_id, depth = stack.pop()
        if node_id in visited or depth > max_depth:
            continue
        visited.add(node_id)

        sym = symbol_refs.get(node_id)
        if not sym:
            continue

        fidx = project_index.by_file.get(sym.file_path)
        if not fidx or not fidx.tree:
            continue

        fn_node = _find_def_node(fidx.tree, sym)
        if not fn_node:
            continue

        local_type_map = _collect_self_assignments(fn_node, fidx)
        local_var_type_map = _collect_local_var_types(fn_node, fidx)
        if sym.class_name:
            local_type_map.update(class_field_type_map.get((sym.module, sym.class_name), {}))

        for call_info in _iter_calls(fn_node):
            callee_id, resolved = _resolve_call_target(
                call_node=call_info["node"],
                current_sym=sym,
                fidx=fidx,
                project_index=project_index,
                local_type_map=local_type_map,
                local_var_type_map=local_var_type_map,
            )

            if not callee_id and follow_external_imports and external_count < max_external_resolve:
                callee_id = _resolve_external_symbol(call_info["node"], fidx)
                if callee_id:
                    resolved = True

            if callee_id:
                call_edges.append(
                    CallEdge(
                        caller_id=node_id,
                        callee_id=callee_id,
                        call_expr=call_info["expr"],
                        file_path=sym.file_path,
                        line=call_info["line"],
                        resolved=resolved,
                    )
                )
                stack.append((callee_id, depth + 1))

                if callee_id.startswith("external:"):
                    external_count += 1

    return GraphBuildOutput(
        symbol_refs=symbol_refs,
        call_edges=call_edges,
        class_edges=class_edges,
        visited_nodes=visited,
        entry_symbol_ids=entry_symbol_ids,
        entry_fallback_used=entry_fallback_used,
    )


def _select_entry_nodes(
    project_index: ProjectIndex,
    entry_files: List[Path],
    *,
    entry_mode: str,
    forward_fallback: bool,
) -> Tuple[List[str], bool]:
    sids: List[str] = []
    entry_fallback_used = False

    if entry_mode == "forward":
        sids = _entry_nodes_forward_only(project_index, entry_files)
        if not sids and forward_fallback:
            entry_fallback_used = True
            sids = _entry_nodes_all(project_index, entry_files)
        return sorted(set(sids)), entry_fallback_used

    if entry_mode == "all":
        return sorted(set(_entry_nodes_all(project_index, entry_files))), False

    raise ValueError(f"unknown entry_mode: {entry_mode}")


def _entry_nodes_forward_only(project_index: ProjectIndex, entry_files: List[Path]) -> List[str]:
    sids: List[str] = []
    for p in entry_files:
        fidx = project_index.by_file.get(str(p))
        if not fidx:
            continue
        # 顶层 forward 函数
        sid = project_index.resolve_symbol(fidx.module_name, "forward")
        if sid:
            sids.append(sid)
        # 文件内所有 *.forward 方法符号（method qualname endswith forward）
        for name, sym in fidx.symbols.items():
            if name.endswith(".forward"):
                sid2 = project_index.resolve_symbol(fidx.module_name, name)
                if sid2:
                    sids.append(sid2)
            # project_index 对 method 的 stored key 是 qualname（形如 Class.method）
            if sym.symbol_type == "method" and sym.qualname.endswith(".forward"):
                sids.append(f"{sym.module}:{sym.qualname}")
    return sids


def _entry_nodes_all(project_index: ProjectIndex, entry_files: List[Path]) -> List[str]:
    sids: List[str] = []
    for p in entry_files:
        fidx = project_index.by_file.get(str(p))
        if not fidx:
            continue
        for c in ("main", "run"):
            sid = project_index.resolve_symbol(fidx.module_name, c)
            if sid:
                sids.append(sid)
        for sym in fidx.symbols.values():
            if sym.symbol_type == "method" and sym.qualname.endswith(".forward"):
                sids.append(f"{sym.module}:{sym.qualname}")
    return sids


def _find_def_node(tree: ast.AST, sym: SymbolRef) -> Optional[ast.FunctionDef]:
    want = sym.qualname.split(".")[-1]
    if sym.symbol_type not in ("function", "method"):
        return None

    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name != want:
            continue
        if sym.symbol_type == "function":
            return node
        if sym.symbol_type == "method" and sym.class_name:
            cls = _find_enclosing_class(tree, node)
            if cls and cls.name == sym.class_name:
                return node
    return None


def _find_enclosing_class(tree: ast.AST, target: ast.FunctionDef) -> Optional[ast.ClassDef]:
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and target in node.body:
            return node
    return None


def _iter_calls(fn_node: ast.AST) -> List[Dict]:
    out: List[Dict] = []
    for n in ast.walk(fn_node):
        if isinstance(n, ast.Call):
            out.append(
                {
                    "node": n,
                    "line": n.lineno,
                    "expr": ast.unparse(n.func) if hasattr(ast, "unparse") else "<call>",
                }
            )
    return out


def _attr_path(attr: ast.Attribute) -> str:
    parts: List[str] = []
    cur: ast.AST = attr
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
    return ".".join(reversed(parts))


def _fq_from_call(call: ast.Call, fidx: FileIndex) -> Optional[str]:
    fn = call.func
    if isinstance(fn, ast.Name):
        if fn.id in fidx.from_imports:
            return fidx.from_imports[fn.id]
        return f"{fidx.module_name}.{fn.id}"
    if isinstance(fn, ast.Attribute):
        path = _attr_path(fn)
        parts = path.split(".")
        if not parts:
            return None
        root = parts[0]
        # 例如: nn.Linear -> alias root "nn" -> torch.nn (import alias)
        mapped_root = fidx.import_aliases.get(root, root)
        return mapped_root + "." + ".".join(parts[1:])
    return None


def _collect_self_assignments(fn_node: ast.AST, fidx: FileIndex) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for n in ast.walk(fn_node):
        if not isinstance(n, ast.Assign) or len(n.targets) != 1:
            continue
        t = n.targets[0]
        if (
            isinstance(t, ast.Attribute)
            and isinstance(t.value, ast.Name)
            and t.value.id == "self"
            and isinstance(n.value, ast.Call)
        ):
            fq = _fq_from_call(n.value, fidx)
            if fq:
                out[t.attr] = fq
    return out


def _collect_local_var_types(fn_node: ast.AST, fidx: FileIndex) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for n in ast.walk(fn_node):
        if not isinstance(n, ast.Assign) or len(n.targets) != 1:
            continue
        t = n.targets[0]
        if isinstance(t, ast.Name) and isinstance(n.value, ast.Call):
            fq = _fq_from_call(n.value, fidx)
            if fq:
                out[t.id] = fq
    return out


def _resolve_call_target(
    *,
    call_node: ast.Call,
    current_sym: SymbolRef,
    fidx: FileIndex,
    project_index: ProjectIndex,
    local_type_map: Dict[str, str],
    local_var_type_map: Dict[str, str],
) -> Tuple[Optional[str], bool]:
    fn = call_node.func

    # 直接函数调用: foo(...)
    if isinstance(fn, ast.Name):
        sid = project_index.resolve_symbol(fidx.module_name, fn.id)
        if sid:
            return sid, True

        # 如果 foo 是某个局部变量实例，则 foo(...) 触发其 __call__/forward
        if fn.id in local_var_type_map:
            fq = local_var_type_map[fn.id]  # module.Cls
            parts = fq.split(".")
            if len(parts) >= 2:
                module = ".".join(parts[:-1])
                cls = parts[-1]
                # 优先 forward
                fwd_sid = project_index.resolve_symbol(module, f"{cls}.forward")
                if fwd_sid:
                    return fwd_sid, True
    # self.xxx(...)
    elif isinstance(fn, ast.Attribute):
        if isinstance(fn.value, ast.Name) and fn.value.id == "self":
            # 先按 self.xxx 字段的类型（来自 __init__）推 forward
            fq_field = local_type_map.get(fn.attr)
            if fq_field:
                parts = fq_field.split(".")
                if len(parts) >= 2:
                    module = ".".join(parts[:-1])
                    cls = parts[-1]
                    fwd_sid = project_index.resolve_symbol(module, f"{cls}.forward")
                    if fwd_sid:
                        return fwd_sid, True

                    # 如果不是项目内实现：当作 external forward
                    if fn.attr != "forward":
                        return f"external:{module}.{cls}.forward", True

            # 否则当作类方法调用：self.method(...)
            if current_sym.class_name:
                sid = project_index.resolve_symbol(
                    current_sym.module, f"{current_sym.class_name}.{fn.attr}"
                )
                if sid:
                    return sid, True

    return None, False


def _resolve_external_symbol(call_node: ast.Call, fidx: FileIndex) -> Optional[str]:
    fn = call_node.func
    # set_seed(...)：可能是 from transformers.trainer_utils import set_seed
    if isinstance(fn, ast.Name):
        full = fidx.from_imports.get(fn.id)
        if full:
            return f"external:{full}"
        return None

    if isinstance(fn, ast.Attribute):
        # torch.cuda.is_available(...)
        path = _attr_path(fn)
        parts = path.split(".")
        if not parts:
            return None
        root = parts[0]
        mapped_root = fidx.import_aliases.get(root, root)
        full = mapped_root + ("." + ".".join(parts[1:]) if len(parts) > 1 else "")
        if mapped_root.startswith("torch") or mapped_root.startswith("functorch") or mapped_root.startswith("triton"):
            return f"external:{full}"
        if "transformers" in full:
            return f"external:{full}"
    return None


def _build_class_field_type_map(
    project_index: ProjectIndex,
) -> Dict[Tuple[str, str], Dict[str, str]]:
    out: Dict[Tuple[str, str], Dict[str, str]] = {}
    # 只在本项目内 build：找到 *. __init__ method 的 AST，然后解析 self.xxx = SomeModule(...)
    for sid, sym in project_index.symbol_by_id.items():
        if sym.symbol_type != "method" or not sym.qualname.endswith(".__init__"):
            continue
        if not sym.class_name:
            continue
        fidx = project_index.by_file.get(sym.file_path)
        if not fidx or not fidx.tree:
            continue
        fn_node = _find_def_node(fidx.tree, sym)
        if not fn_node:
            continue
        out[(sym.module, sym.class_name)] = _collect_self_assignments(fn_node, fidx)
    return out

