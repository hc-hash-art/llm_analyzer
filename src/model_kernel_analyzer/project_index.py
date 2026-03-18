from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from .types import SymbolRef


def _iter_py_files(project_root: Path, exclude_dirs: Iterable[str]) -> Iterable[Path]:
    exclude = set(exclude_dirs)
    if not project_root.exists():
        return
    for path in project_root.rglob("*.py"):
        if any(part in exclude for part in path.parts):
            continue
        yield path


def _module_name_for(scan_root: Path, file_path: Path, module_prefix: str) -> str:
    rel = file_path.relative_to(scan_root).with_suffix("")
    parts = list(rel.parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    rel_mod = ".".join(parts)
    if module_prefix:
        return module_prefix if not rel_mod else f"{module_prefix}.{rel_mod}"
    return rel_mod


@dataclass
class FileIndex:
    file_path: Path
    module_name: str
    tree: Optional[ast.AST] = None
    import_aliases: Dict[str, str] = field(default_factory=dict)  # alias -> module/fullname root
    from_imports: Dict[str, str] = field(default_factory=dict)  # symbol -> full dotted path
    symbols: Dict[str, SymbolRef] = field(default_factory=dict)


@dataclass
class ProjectIndex:
    by_file: Dict[str, FileIndex]
    symbol_id_by_module_qualname: Dict[Tuple[str, str], str]
    symbol_by_id: Dict[str, SymbolRef]
    symbol_ids_by_qualname: Dict[str, List[str]]

    def resolve_symbol(self, module: str, qualname: str) -> Optional[str]:
        hit = self.symbol_id_by_module_qualname.get((module, qualname))
        if hit:
            return hit
        # 兜底：module 可能因 re-export 而不一致，用 qualname 找
        lst = self.symbol_ids_by_qualname.get(qualname, [])
        if lst:
            return lst[0]
        return None


def build_project_index(project_root: Path, exclude_dirs: Iterable[str]) -> ProjectIndex:
    return build_project_index_multi(
        scans=[(project_root, "")],
        exclude_dirs=exclude_dirs,
    )


def build_project_index_multi(
    scans: List[Tuple[Path, str]],
    exclude_dirs: Iterable[str],
) -> ProjectIndex:
    by_file: Dict[str, FileIndex] = {}
    symbol_by_id: Dict[str, SymbolRef] = {}
    lookup: Dict[Tuple[str, str], str] = {}
    qual_lookup: Dict[str, List[str]] = {}

    for scan_root, module_prefix in scans:
        scan_root = scan_root.resolve()
        for file_path in _iter_py_files(scan_root, exclude_dirs):
            module_name = _module_name_for(scan_root, file_path, module_prefix)
            text = file_path.read_text(encoding="utf-8", errors="ignore")
            try:
                tree = ast.parse(text)
            except SyntaxError:
                continue

            fidx = FileIndex(file_path=file_path, module_name=module_name, tree=tree)
            _collect_imports(tree, fidx)
            _collect_symbols(tree, fidx, symbol_by_id, lookup, qual_lookup)

            by_file[str(file_path)] = fidx

    return ProjectIndex(
        by_file=by_file,
        symbol_id_by_module_qualname=lookup,
        symbol_by_id=symbol_by_id,
        symbol_ids_by_qualname=qual_lookup,
    )


def _collect_imports(tree: ast.AST, fidx: FileIndex) -> None:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                asname = alias.asname or alias.name.split(".")[0]
                fidx.import_aliases[asname] = alias.name
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            for alias in node.names:
                name = alias.asname or alias.name
                # 例如: from torch.nn import functional as F
                fidx.from_imports[name] = f"{mod}.{alias.name}".strip(".")


def _collect_symbols(
    tree: ast.AST,
    fidx: FileIndex,
    symbol_by_id: Dict[str, SymbolRef],
    lookup: Dict[Tuple[str, str], str],
    qual_lookup: Dict[str, List[str]],
) -> None:
    if not isinstance(tree, ast.Module):
        return

    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            qual = node.name
            sym = SymbolRef(
                module=fidx.module_name,
                qualname=qual,
                file_path=str(fidx.file_path),
                line_start=node.lineno,
                line_end=getattr(node, "end_lineno", node.lineno),
                symbol_type="function",
                class_name=None,
            )
            sid = f"{sym.module}:{sym.qualname}"
            fidx.symbols[qual] = sym
            symbol_by_id[sid] = sym
            lookup[(sym.module, sym.qualname)] = sid
            qual_lookup.setdefault(sym.qualname, []).append(sid)
        elif isinstance(node, ast.ClassDef):
            cls_sym = SymbolRef(
                module=fidx.module_name,
                qualname=node.name,
                file_path=str(fidx.file_path),
                line_start=node.lineno,
                line_end=getattr(node, "end_lineno", node.lineno),
                symbol_type="class",
                class_name=node.name,
            )
            cls_id = f"{cls_sym.module}:{cls_sym.qualname}"
            fidx.symbols[node.name] = cls_sym
            symbol_by_id[cls_id] = cls_sym
            lookup[(cls_sym.module, cls_sym.qualname)] = cls_id
            qual_lookup.setdefault(cls_sym.qualname, []).append(cls_id)

            # methods
            for sub in node.body:
                if isinstance(sub, ast.FunctionDef):
                    qual = f"{node.name}.{sub.name}"
                    m_sym = SymbolRef(
                        module=fidx.module_name,
                        qualname=qual,
                        file_path=str(fidx.file_path),
                        line_start=sub.lineno,
                        line_end=getattr(sub, "end_lineno", sub.lineno),
                        symbol_type="method",
                        class_name=node.name,
                    )
                    sid = f"{m_sym.module}:{m_sym.qualname}"
                    fidx.symbols[qual] = m_sym
                    symbol_by_id[sid] = m_sym
                    lookup[(m_sym.module, m_sym.qualname)] = sid
                    qual_lookup.setdefault(m_sym.qualname, []).append(sid)

