from __future__ import annotations

import json
from pathlib import Path

from .call_chain import build_call_chain_forest, format_call_forest_text
from .call_graph import build_call_and_class_graph
from .config import AnalysisConfig
from .entry_discovery import discover_entry_files
from .kernel_mapper import map_ops_to_impl_paths
from .op_chain_filter import is_op_chain_node
from .op_extractor import extract_operator_calls
from .project_index import build_project_index, build_project_index_multi
from .types import AnalysisResult
from .call_chain import build_outgoing_edges
import os


def run_pipeline(config: AnalysisConfig) -> AnalysisResult:
    entry_files = discover_entry_files(config)

    # 若由 setup_env/deps_resolver 注入，额外扫描 transformers 源码目录
    extra_scan_env = os.environ.get("LLM_ANALYZER_EXTRA_SCAN_PACKAGES", "").strip()
    scans = [(config.project_root, "")]
    if extra_scan_env:
        # 格式: transformers=/abs/path;torch=/abs/path
        for item in extra_scan_env.split(";"):
            item = item.strip()
            if not item or "=" not in item:
                continue
            k, v = item.split("=", 1)
            k = k.strip()
            v = v.strip()
            if not k or not v:
                continue
            scans.append((Path(v), k))

    if len(scans) == 1:
        project_index = build_project_index(config.project_root, config.exclude_dirs)
    else:
        project_index = build_project_index_multi(scans=scans, exclude_dirs=config.exclude_dirs)

    graph = build_call_and_class_graph(
        project_index=project_index,
        entry_files=entry_files,
        max_depth=config.max_call_depth,
        follow_external_imports=config.follow_external_imports,
        max_external_resolve=config.max_external_resolve,
        entry_mode=config.entry_mode,
        forward_fallback=config.forward_fallback,
    )

    op_calls = extract_operator_calls(
        symbol_refs=graph.symbol_refs,
        visited_nodes=graph.visited_nodes,
        project_index=project_index,
    )
    map_ops_to_impl_paths(op_calls)

    outgoing = build_outgoing_edges(graph.call_edges)
    forest = build_call_chain_forest(
        roots=graph.entry_symbol_ids,
        outgoing=outgoing,
        symbols=graph.symbol_refs,
        max_depth=config.chain_max_depth,
        max_branches_per_node=config.chain_max_branches,
        op_chain_only=config.op_chain_only,
        project_root=config.project_root,
        op_chain_include_modules=config.op_chain_include_modules,
        op_chain_exclude_modules=config.op_chain_exclude_modules,
    )

    return AnalysisResult(
        entry_files=[str(x) for x in entry_files],
        entry_mode=config.entry_mode,
        entry_symbol_ids=graph.entry_symbol_ids,
        entry_fallback_used=graph.entry_fallback_used,
        op_chain_only=config.op_chain_only,
        symbols=graph.symbol_refs,
        call_edges=graph.call_edges,
        op_calls=op_calls,
        call_chain_forest=forest,
    )


def dump_result(result: AnalysisResult, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

