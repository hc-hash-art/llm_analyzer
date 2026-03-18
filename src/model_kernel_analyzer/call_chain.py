from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, FrozenSet, List, Optional

from .op_chain_filter import is_op_chain_node
from .types import CallEdge, SymbolRef


def build_outgoing_edges(call_edges: List[CallEdge]) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for e in call_edges:
        out[e.caller_id].append(
            {
                "callee_id": e.callee_id,
                "call_expr": e.call_expr,
                "line": e.line,
                "resolved": e.resolved,
                "file_path": e.file_path,
            }
        )
    return dict(out)


def build_call_chain_forest(
    roots: List[str],
    outgoing: Dict[str, List[Dict[str, Any]]],
    symbols: Dict[str, SymbolRef],
    max_depth: int,
    max_branches_per_node: int,
    *,
    op_chain_only: bool,
    project_root: Optional[Any],
    op_chain_include_modules: List[str],
    op_chain_exclude_modules: List[str],
) -> List[Dict[str, Any]]:
    inc: FrozenSet[str] = frozenset(op_chain_include_modules or [])
    exc: FrozenSet[str] = frozenset(op_chain_exclude_modules or [])

    forest: List[Dict[str, Any]] = []
    for root in roots:
        forest.append(
            _expand_node(
                root,
                outgoing,
                symbols,
                depth=0,
                max_depth=max_depth,
                max_branches=max_branches_per_node,
                ancestor_chain=[],
                op_chain_only=op_chain_only,
                project_root=project_root,
                extra_include=inc,
                extra_exclude=exc,
            )
        )
    return forest


def _node_summary(sid: str, symbols: Dict[str, SymbolRef]) -> Dict[str, Any]:
    s = symbols.get(sid)
    if not s:
        return {"symbol_id": sid, "qualname": sid, "module": "", "file": "", "line_start": 0}
    return {
        "symbol_id": sid,
        "qualname": s.qualname,
        "module": s.module,
        "file": s.file_path,
        "line_start": s.line_start,
        "symbol_type": s.symbol_type,
        "class_name": s.class_name,
    }


def _expand_node(
    sid: str,
    outgoing: Dict[str, List[Dict[str, Any]]],
    symbols: Dict[str, SymbolRef],
    depth: int,
    max_depth: int,
    max_branches: int,
    ancestor_chain: List[str],
    op_chain_only: bool,
    project_root: Optional[Any],
    extra_include: FrozenSet[str],
    extra_exclude: FrozenSet[str],
) -> Dict[str, Any]:
    node = _node_summary(sid, symbols)
    node["_is_op_chain_node"] = (not op_chain_only) or is_op_chain_node(
        sid,
        symbols.get(sid),
        project_root=project_root,
        extra_include_modules=extra_include,
        extra_exclude_modules=extra_exclude,
    )
    if sid in ancestor_chain:
        node["cycle_back_to_ancestor"] = True
        node["children"] = []
        return node
    if depth >= max_depth:
        node["truncated"] = True
        node["children"] = []
        return node

    edges = outgoing.get(sid, [])
    if len(edges) > max_branches:
        node["branch_truncated"] = True
        node["branch_total"] = len(edges)
        edges = edges[:max_branches]

    children: List[Dict[str, Any]] = []
    path = ancestor_chain + [sid]
    for e in edges:
        cid = e["callee_id"]
        via = f"{e['call_expr']} @ L{e['line']}"

        if cid in path:
            children.append(
                {"via": via, "resolved": e["resolved"], "callee": _node_summary(cid, symbols) | {"cycle_back_to_ancestor": True, "children": []}}
            )
            continue

        sym = symbols.get(cid)
        want = (not op_chain_only) or is_op_chain_node(
            cid,
            sym,
            project_root=project_root,
            extra_include_modules=extra_include,
            extra_exclude_modules=extra_exclude,
        )

        if want:
            children.append(
                {
                    "via": via,
                    "resolved": e["resolved"],
                    "callee": _expand_node(
                        cid,
                        outgoing,
                        symbols,
                        depth + 1,
                        max_depth,
                        max_branches,
                        path,
                        op_chain_only,
                        project_root,
                        extra_include,
                        extra_exclude,
                    ),
                }
            )
        else:
            # 透明跳过：向下找它的后继，并把路径压缩掉
            children.extend(
                _transparent_lift(
                    cid,
                    via,
                    e["resolved"],
                    outgoing,
                    symbols,
                    depth + 1,
                    max_depth,
                    max_branches,
                    path,
                    op_chain_only,
                    project_root,
                    extra_include,
                    extra_exclude,
                )
            )

    node["children"] = children
    return node


def _transparent_lift(
    skip_id: str,
    prefix_via: str,
    resolved: bool,
    outgoing: Dict[str, List[Dict[str, Any]]],
    symbols: Dict[str, SymbolRef],
    depth: int,
    max_depth: int,
    max_branches: int,
    path_stack: List[str],
    op_chain_only: bool,
    project_root: Optional[Any],
    extra_include: FrozenSet[str],
    extra_exclude: FrozenSet[str],
) -> List[Dict[str, Any]]:
    if skip_id in path_stack:
        stub = _node_summary(skip_id, symbols)
        stub["cycle_back_to_ancestor"] = True
        stub["children"] = []
        return [{"via": prefix_via, "resolved": resolved, "callee": stub}]

    if depth >= max_depth:
        stub = _node_summary(skip_id, symbols)
        stub["truncated"] = True
        stub["children"] = []
        return [{"via": prefix_via + " (经跳过结点)", "resolved": resolved, "callee": stub}]

    sub_edges = outgoing.get(skip_id, [])
    if len(sub_edges) > max_branches:
        sub_edges = sub_edges[:max_branches]

    out: List[Dict[str, Any]] = []
    new_stack = path_stack + [skip_id]
    for se in sub_edges:
        cid = se["callee_id"]
        via = prefix_via + f" » {se['call_expr']} @ L{se['line']}"
        sym = symbols.get(cid)
        want = (not op_chain_only) or is_op_chain_node(
            cid,
            sym,
            project_root=project_root,
            extra_include_modules=extra_include,
            extra_exclude_modules=extra_exclude,
        )
        if want:
            out.append(
                {
                    "via": via,
                    "resolved": se["resolved"],
                    "callee": _expand_node(
                        cid,
                        outgoing,
                        symbols,
                        depth + 1,
                        max_depth,
                        max_branches,
                        new_stack,
                        op_chain_only,
                        project_root,
                        extra_include,
                        extra_exclude,
                    ),
                }
            )
        else:
            out.extend(
                _transparent_lift(
                    cid,
                    via,
                    se["resolved"],
                    outgoing,
                    symbols,
                    depth + 1,
                    max_depth,
                    max_branches,
                    new_stack,
                    op_chain_only,
                    project_root,
                    extra_include,
                    extra_exclude,
                )
            )
    return out


def format_call_forest_text(
    forest: List[Dict[str, Any]],
    indent: str = "  ",
    root_label: str = "入口",
) -> str:
    lines: List[str] = []

    def walk(n: Dict[str, Any], prefix: str) -> None:
        if n.get("module"):
            head = f"{n.get('module','')}:{n.get('qualname','')}"
        else:
            head = f"{n.get('qualname','')}"
        loc = f" [{n.get('file','')}:{n.get('line_start',0)}]"
        if n.get("cycle_back_to_ancestor"):
            lines.append(f"{prefix}↺ {head}{loc} (环回到祖先，已截断)")
            return
        if n.get("truncated"):
            lines.append(f"{prefix}… {head}{loc} (达到 max_depth，已截断)")
            return
        # 顶层根结点如果不是 op/kernel，则只展示其孩子（满足“仅展示 op/kernel”）
        if prefix == "" and not n.get("_is_op_chain_node", False) and n.get("children"):
            for ch in n.get("children", []):
                via = ch.get("via", "")
                lines.append(f"{prefix}{indent}→ {via}")
                walk(ch.get("callee", {}), prefix + indent + indent)
            return

        lines.append(f"{prefix}{head}{loc}")
        for ch in n.get("children", []):
            via = ch.get("via", "")
            lines.append(f"{prefix}{indent}→ {via}")
            walk(ch.get("callee", {}), prefix + indent + indent)

    for i, tree in enumerate(forest):
        lines.append(f"--- {root_label} #{i + 1} ---")
        walk(tree, "")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"

