from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional

from .deps_resolver import setup_deps_and_run_analyze


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Auto-resolve torch/transformers versions, install into venv, and run analyzer."
    )
    parser.add_argument(
        "--project-root",
        required=True,
        help="Model project root (e.g., /home/hc/Qwen3)",
    )
    parser.add_argument(
        "--analyze-config",
        required=True,
        help="Analyzer YAML config (e.g., configs/qwen3.yaml)",
    )
    parser.add_argument(
        "--venv-path",
        required=False,
        default="/home/hc/llm_analyzer/.venv/model_deps_env",
        help="Where to create the venv",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only parse versions and print plans",
    )
    parser.add_argument(
        "--torch-index-url",
        default="https://download.pytorch.org/whl/cpu",
        help="Torch CPU wheels index",
    )
    args = parser.parse_args()

    project_root = Path(args.project_root)
    analyze_config = Path(args.analyze_config).resolve()
    venv_path = Path(args.venv_path)

    res = setup_deps_and_run_analyze(
        project_root=project_root,
        analyze_config=analyze_config,
        venv_path=venv_path,
        dry_run=args.dry_run,
        explicit_specs=None,
        extra_pip_args=["--progress-bar=off"],
        torch_index_url=args.torch_index_url,
    )

    # Print key outputs
    if res.get("dry_run"):
        print("Dry-run only.")
        print("Resolved specs:")
        for k, v in res["specs"].items():
            print(f"  {k}: {v}")
        print(f"Would create venv: {res['venv_path']}")
        print(f"Would run python: {res['would_run_python']}")
        print(f"Would run analyze config: {res['would_run_analyze_config']}")
    else:
        print("Env setup + analysis done.")
        print("Resolved specs:")
        for k, v in res["specs"].items():
            print(f"  {k}: {v}")
        print("torch/transformers module files:")
        for mod, mf in res["module_files"].items():
            print(f"  {mod}: {mf}")
        print("Source dirs:")
        for mod, md in res["module_dirs"].items():
            print(f"  {mod}: {md}")


if __name__ == "__main__":
    main()

