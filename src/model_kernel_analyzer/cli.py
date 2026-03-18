from __future__ import annotations

import argparse

from .call_chain import format_call_forest_text
from .config import AnalysisConfig
from .pipeline import dump_result, run_pipeline
from .deps_resolver import setup_deps_and_run_analyze
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Static analyzer for op/kernel impl paths.")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument(
        "--auto-deps",
        action="store_true",
        help="Auto-resolve torch/transformers versions, install into venv, then run analyzer inside venv.",
    )
    parser.add_argument(
        "--deps-dry-run",
        action="store_true",
        help="Only resolve versions and print planned installs (no download).",
    )
    parser.add_argument(
        "--venv-path",
        default="/home/hc/llm_analyzer/.venv/model_deps_env",
        help="Venv path for auto-deps.",
    )
    parser.add_argument(
        "--torch-index-url",
        default="https://download.pytorch.org/whl/cpu",
        help="Torch wheel index URL (CPU by default).",
    )
    parser.add_argument(
        "--no-print-chains",
        action="store_true",
        help="Only write JSON output, do not print call chains.",
    )
    args = parser.parse_args()

    config = AnalysisConfig.from_file(args.config)
    if args.auto_deps:
        # 解析/安装完成后，由 deps_resolver 内部用 venv python 再跑一次本分析器
        setup_deps_and_run_analyze(
            project_root=config.project_root,
            analyze_config=Path(args.config),
            venv_path=Path(args.venv_path),
            dry_run=bool(args.deps_dry_run),
            explicit_specs=None,
            extra_pip_args=["--progress-bar=off"],
            torch_index_url=args.torch_index_url,
        )
        return

    if args.no_print_chains:
        config.print_call_chains = False  # type: ignore[attr-defined]

    result = run_pipeline(config)
    dump_result(result, config.output_path)

    print(f"Analysis done. Output: {config.output_path}")
    print(f"entry_mode: {result.entry_mode}")
    print(f"entry_symbol_ids: {len(result.entry_symbol_ids)}")
    if result.entry_fallback_used:
        print("提示: forward 入口未命中，已回退到 all/主流程入口。")
    print(f"Op calls: {len(result.op_calls)}")
    print(f"op_chain_only: {result.op_chain_only}")

    if config.print_call_chains:
        print()
        print(format_call_forest_text(result.call_chain_forest, root_label="forward 入口" if result.entry_mode == "forward" else "入口"))


if __name__ == "__main__":
    main()

