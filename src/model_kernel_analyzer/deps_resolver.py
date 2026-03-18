from __future__ import annotations

import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class DepSpec:
    name: str  # torch | transformers
    pip_spec: str  # e.g. "torch==2.1.0" or "torch>=2.0"
    version: Optional[str] = None


TORCH_VERSION_NAMES = ("torch",)
TRANSFORMERS_VERSION_NAMES = ("transformers",)


def _read_text_if_exists(p: Path) -> str:
    if not p.exists() or not p.is_file():
        return ""
    return p.read_text(encoding="utf-8", errors="ignore")


REQ_LINE_RE = re.compile(
    r"^(?P<name>torch|transformers)\s*(?P<op>==|~=|>=|<=|!=|===)\s*(?P<ver>[^#\s;]+)",
    re.IGNORECASE,
)


def parse_dep_specs_from_text(text: str) -> Dict[str, DepSpec]:
    """
    解析文本里的 torch/transformers 版本（尽量找到精确==，否则退化到 >=/<= 等）。
    返回 key: torch/transformers
    """
    out: Dict[str, DepSpec] = {}
    for line in text.splitlines():
        line_stripped = line.strip()
        if not line_stripped or line_stripped.startswith("#"):
            continue
        m = REQ_LINE_RE.match(line_stripped)
        if not m:
            continue
        name = m.group("name").lower()
        op = m.group("op")
        ver = m.group("ver")
        pip_spec = f"{name}{op}{ver}"
        existing = out.get(name)
        # 优先选择 "=="；否则用先出现的
        if existing is None:
            out[name] = DepSpec(name=name, pip_spec=pip_spec, version=ver)
        else:
            if "==" in pip_spec and (existing.pip_spec.find("==") == -1):
                out[name] = DepSpec(name=name, pip_spec=pip_spec, version=ver)
    return out


def parse_dep_specs_from_files(project_root: Path) -> Dict[str, DepSpec]:
    """
    优先级：requirements.txt > Dockerfile > README*.md
    """
    project_root = project_root.resolve()
    candidates: List[Tuple[int, Path, str]] = []

    req = project_root / "requirements.txt"
    candidates.append((10, req, "requirements.txt"))

    # Dockerfile 在根目录（或以 Dockerfile 开头的文件）
    docker_files = sorted(project_root.glob("Dockerfile*"))
    for df in docker_files:
        candidates.append((8, df, "Dockerfile*"))

    readmes = sorted(project_root.glob("README*.md"))
    for rf in readmes:
        candidates.append((6, rf, "README*.md"))

    merged: Dict[str, DepSpec] = {}
    for prio, path, _ in sorted(candidates, key=lambda x: -x[0]):
        text = _read_text_if_exists(path)
        if not text:
            continue
        parsed = parse_dep_specs_from_text(text)
        # 按优先级“覆盖”：数值越大优先级越高（上面 candidates prio 越大越先）
        for k, spec in parsed.items():
            if k in merged:
                continue
            merged[k] = spec
    return merged


def resolve_or_fallback_specs(
    project_root: Path,
    explicit: Optional[Dict[str, str]] = None,
    *,
    fallback_torch: str = "torch",
    fallback_transformers: str = "transformers",
) -> Dict[str, DepSpec]:
    """
    explicit: {"torch": "torch==2.1.0", "transformers": "transformers==4.36.0"}
    """
    explicit = explicit or {}
    out: Dict[str, DepSpec] = {}
    for name, pip_spec in explicit.items():
        lname = name.lower()
        m = re.match(rf"^{re.escape(lname)}(==|~=|>=|<=|!=|===)(.+)$", pip_spec)
        ver = m.group(2) if m else None
        out[lname] = DepSpec(name=lname, pip_spec=pip_spec, version=ver)

    parsed = parse_dep_specs_from_files(project_root)
    for lname, spec in parsed.items():
        if lname not in out:
            out[lname] = spec

    if "torch" not in out:
        out["torch"] = DepSpec(name="torch", pip_spec=fallback_torch, version=None)
    if "transformers" not in out:
        out["transformers"] = DepSpec(
            name="transformers", pip_spec=fallback_transformers, version=None
        )
    return out


def _run(cmd: List[str], *, cwd: Optional[Path] = None) -> None:
    subprocess.check_call(cmd, cwd=str(cwd) if cwd else None)


def ensure_venv(venv_path: Path) -> Path:
    venv_path = venv_path.resolve()
    if not venv_path.exists():
        _run([sys.executable, "-m", "venv", str(venv_path)])
    return venv_path


def venv_python(venv_path: Path) -> str:
    venv_path = venv_path.resolve()
    return str(venv_path / "bin" / "python")


def venv_pip(venv_path: Path) -> str:
    venv_path = venv_path.resolve()
    return str(venv_path / "bin" / "pip")


def install_deps(
    venv_path: Path,
    specs: Dict[str, DepSpec],
    *,
    extra_pip_args: Optional[List[str]] = None,
    torch_index_url: Optional[str] = None,
) -> None:
    extra_pip_args = extra_pip_args or []
    ensure_venv(venv_path)

    pip_cmd = [venv_pip(venv_path), "install", "--upgrade", "pip"]
    _run(pip_cmd)

    # 我们需要至少 pyyaml（分析器 config 用）
    _run([venv_pip(venv_path), "install", "pyyaml"] + extra_pip_args)

    install_args = [venv_pip(venv_path), "install", "--no-cache-dir"]
    install_args += extra_pip_args
    if torch_index_url:
        install_args += ["--index-url", torch_index_url]

    # 注意：torch 包可能非常大；这里只做照 spec 安装
    for dep_name in ("torch", "transformers"):
        if dep_name in specs:
            install_args.append(specs[dep_name].pip_spec)
    _run(install_args)


def get_import_file(python_bin: str, module: str) -> Tuple[str, str]:
    """
    返回 (module_file, module_dir)
    """
    # torch: want torch.__file__
    # transformers: transformers.__file__
    code = (
        "import importlib, os; m=importlib.import_module('"
        + module
        + "'); print(m.__file__); print(os.path.dirname(m.__file__))"
    )
    out = subprocess.check_output([python_bin, "-c", code], text=True).strip().splitlines()
    if len(out) < 2:
        return out[0], os.path.dirname(out[0])
    return out[0], out[1]


def setup_deps_and_run_analyze(
    *,
    project_root: Path,
    analyze_config: Path,
    venv_path: Path,
    dry_run: bool = False,
    explicit_specs: Optional[Dict[str, str]] = None,
    extra_pip_args: Optional[List[str]] = None,
    torch_index_url: Optional[str] = "https://download.pytorch.org/whl/cpu",
) -> Dict[str, object]:
    """
    一键流程：
    1) 从 project_root 的 Dockerfile/requirements/README 解析版本
    2) 创建 venv
    3) 安装 torch / transformers
    4) 读取 torch/transformers 的 __file__，定位 source
    5) 用 venv 的 python 重新执行 analyzer
    """
    specs = resolve_or_fallback_specs(project_root, explicit=explicit_specs)
    venv_path = venv_path.resolve()
    analyze_config = analyze_config.resolve()

    python_bin = venv_python(venv_path)
    if dry_run:
        return {
            "dry_run": True,
            "specs": {k: s.pip_spec for k, s in specs.items()},
            "venv_path": str(venv_path),
            "would_run_python": python_bin,
            "would_run_analyze_config": str(analyze_config),
        }

    install_deps(
        venv_path,
        specs,
        extra_pip_args=extra_pip_args,
        torch_index_url=torch_index_url,
    )

    module_files: Dict[str, str] = {}
    module_dirs: Dict[str, str] = {}
    for mod in ("torch", "transformers"):
        mf, md = get_import_file(python_bin, mod)
        module_files[mod] = mf
        module_dirs[mod] = md

    # 让 analyzer 把 transformers 源码也纳入静态索引
    # module_dirs['transformers'] 形如 /path/site-packages/transformers
    # 解析时 module_prefix='transformers'，从而 module 名称能对齐到 transformers.*
    extra_scan_packages: Dict[str, str] = {}
    if "transformers" in module_dirs:
        extra_scan_packages["transformers"] = module_dirs["transformers"]

    # 用 venv python 执行当前分析命令；确保当前目录是 analyzer root
    analyzer_root = Path(__file__).resolve().parents[2]  # /home/hc/llm_analyzer
    cmd = [
        python_bin,
        "-m",
        "src.model_kernel_analyzer.cli",
        "--config",
        str(analyze_config),
    ]
    env = os.environ.copy()
    if extra_scan_packages:
        env["LLM_ANALYZER_EXTRA_SCAN_PACKAGES"] = ";".join(
            [f"{k}={v}" for k, v in extra_scan_packages.items()]
        )
    subprocess.check_call(cmd, cwd=str(analyzer_root), env=env)

    return {
        "dry_run": False,
        "specs": {k: s.pip_spec for k, s in specs.items()},
        "venv_path": str(venv_path),
        "module_files": module_files,
        "module_dirs": module_dirs,
    }

