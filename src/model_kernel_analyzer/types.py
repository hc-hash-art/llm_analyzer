from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class SymbolRef:
    module: str
    qualname: str
    file_path: str
    line_start: int
    line_end: int
    symbol_type: str  # function | class | method
    class_name: Optional[str] = None


@dataclass
class CallEdge:
    caller_id: str
    callee_id: str
    call_expr: str
    file_path: str
    line: int
    resolved: bool


@dataclass
class ClassEdge:
    owner_class_id: str
    field_name: str
    target_class_id: str
    file_path: str
    line: int


@dataclass
class OpCall:
    node_id: str
    op_name: str
    call_expr: str
    file_path: str
    line: int
    impl_path: Optional[str] = None
    mapper_note: Optional[str] = None


@dataclass
class AnalysisResult:
    entry_files: List[str]
    entry_mode: str
    entry_symbol_ids: List[str]
    entry_fallback_used: bool
    op_chain_only: bool

    symbols: Dict[str, SymbolRef] = field(default_factory=dict)
    call_edges: List[CallEdge] = field(default_factory=list)
    class_edges: List[ClassEdge] = field(default_factory=list)
    op_calls: List[OpCall] = field(default_factory=list)

    # 显示用的过滤后调用树
    call_chain_forest: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entry_files": self.entry_files,
            "entry_mode": self.entry_mode,
            "entry_symbol_ids": self.entry_symbol_ids,
            "entry_fallback_used": self.entry_fallback_used,
            "op_chain_only": self.op_chain_only,
            "symbols": {k: asdict(v) for k, v in self.symbols.items()},
            "call_edges": [asdict(x) for x in self.call_edges],
            "class_edges": [asdict(x) for x in self.class_edges],
            "op_calls": [asdict(x) for x in self.op_calls],
            "call_chain_forest": self.call_chain_forest,
        }

