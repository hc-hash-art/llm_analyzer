from __future__ import annotations

import re
from pathlib import Path
from typing import List

from .config import AnalysisConfig


PY_FILE_PATTERN = re.compile(r"([A-Za-z0-9_\-/]+\.py)")


def discover_entry_files(config: AnalysisConfig) -> List[Path]:
    # 允许配置里显式指定
    if config.entry_files:
        return [p for p in config.entry_files if p.exists()]

    # README 里粗略抽取形如 xxx.py 的字符串（用于示例）
    if config.readme_path.exists():
        text = config.readme_path.read_text(encoding="utf-8", errors="ignore")
        entries: List[Path] = []
        for m in PY_FILE_PATTERN.findall(text):
            cand = (config.project_root / m).resolve()
            if cand.exists() and cand.suffix == ".py":
                entries.append(cand)
        # 稳定排序 + 去重
        uniq = sorted({str(p): p for p in entries}.values(), key=lambda x: str(x))
        return uniq

    return []

