from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import yaml


@dataclass
class AnalysisConfig:
    project_root: Path
    readme_path: Path
    entry_files: List[Path]
    include_globs: List[str]
    exclude_dirs: List[str]

    # 调用图遍历
    max_call_depth: int
    follow_external_imports: bool
    max_external_resolve: int
    entry_mode: str  # forward | all
    forward_fallback: bool

    # 调用链展示
    chain_max_depth: int
    chain_max_branches: int
    print_call_chains: bool
    op_chain_only: bool
    op_chain_include_modules: List[str]
    op_chain_exclude_modules: List[str]

    # 输出
    output_path: Path

    @classmethod
    def from_file(cls, config_path: str) -> "AnalysisConfig":
        raw = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
        analysis = raw.get("analysis", {})
        output = raw.get("output", {})
        project_root = Path(raw["project_root"]).resolve()
        readme_path = Path(raw["readme_path"]).resolve() if "readme_path" in raw else project_root / "README.md"

        return cls(
            project_root=project_root,
            readme_path=readme_path,
            entry_files=[Path(x).resolve() for x in raw.get("entry_files", [])],
            include_globs=raw.get("include_globs", ["*.py"]),
            exclude_dirs=raw.get("exclude_dirs", []),
            max_call_depth=int(analysis.get("max_call_depth", 20)),
            follow_external_imports=bool(analysis.get("follow_external_imports", True)),
            max_external_resolve=int(analysis.get("max_external_resolve", 200)),
            entry_mode=str(analysis.get("entry_mode", "forward")),
            forward_fallback=bool(analysis.get("forward_fallback", True)),
            chain_max_depth=int(analysis.get("chain_max_depth", analysis.get("max_call_depth", 20))),
            chain_max_branches=int(analysis.get("chain_max_branches", 64)),
            print_call_chains=bool(analysis.get("print_call_chains", True)),
            op_chain_only=bool(analysis.get("op_chain_only", True)),
            op_chain_include_modules=list(analysis.get("op_chain_include_modules", [])),
            op_chain_exclude_modules=list(analysis.get("op_chain_exclude_modules", [])),
            output_path=Path(output.get("path", "outputs/analysis.json")).resolve(),
        )

