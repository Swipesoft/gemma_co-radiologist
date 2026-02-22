"""
nanoathens.core — Tool Infrastructure
══════════════════════════════════════
ToolType, ArgExtractorType, ToolSchema, ToolSchemaValidator, ToolRegistry.
"""

import asyncio
import json
from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


# ── Tool Taxonomy ─────────────────────────────────────────────────────────────

class ToolType(Enum):
    RETRIEVAL   = "retrieval"
    COMPUTATION = "computation"
    KNOWLEDGE   = "knowledge"


class ArgExtractorType(Enum):
    ENUM            = "enum"
    QUOTED          = "quoted"
    LANGUAGE        = "language"
    NUMERIC_ID      = "numeric_id"
    ALPHANUMERIC_ID = "alphanumeric_id"
    NUMBER          = "number"


# ── Tool Schema ───────────────────────────────────────────────────────────────

@dataclass
class ToolSchema:
    name: str
    description: str
    parameters: Dict[str, Any]
    required: List[str]
    example: Dict[str, Any]
    docstring: str
    tool_type: ToolType = ToolType.KNOWLEDGE
    arg_sources: Dict[str, str] = field(default_factory=dict)
    output_keys: Dict[str, str] = field(default_factory=dict)
    explicit_keywords: List[str] = field(default_factory=list)
    arg_extractors: Dict[str, tuple] = field(default_factory=dict)

    def get_bm25_document(self) -> str:
        return f"{self.name} {self.description} {self.docstring} {json.dumps(self.parameters)}"

    def get_medgemma_format(self) -> str:
        return f"- {self.name}: {self.description}\n  args={json.dumps(self.parameters)}"


# ── Collision-Free Validator ──────────────────────────────────────────────────

class ToolSchemaValidationError(Exception):
    pass


class ToolSchemaValidator:
    @staticmethod
    def validate(schema: ToolSchema):
        errors = []
        for req in schema.required:
            if req not in schema.parameters:
                errors.append(f"required '{req}' not in parameters")
        for p in schema.arg_sources:
            if p not in schema.parameters:
                errors.append(f"arg_sources refs unknown param '{p}'")
        for p in schema.arg_extractors:
            if p not in schema.parameters:
                errors.append(f"arg_extractors refs unknown param '{p}'")
        # THE COLLISION-FREE INVARIANT
        arg_source_values = set(schema.arg_sources.values())
        for out_key in schema.output_keys:
            if out_key in arg_source_values:
                errors.append(
                    f"COLLISION: output_key '{out_key}' also in arg_sources -- violates DAG invariant"
                )
        for p, (ext_type, cfg) in schema.arg_extractors.items():
            if ext_type == ArgExtractorType.LANGUAGE:
                if "mapping" not in cfg: errors.append(f"LANGUAGE {p}: missing mapping")
                if "role" not in cfg:    errors.append(f"LANGUAGE {p}: missing role")
            elif ext_type == ArgExtractorType.NUMERIC_ID:
                if "preceding_words" not in cfg:
                    errors.append(f"NUMERIC_ID {p}: missing preceding_words")
            elif ext_type == ArgExtractorType.ALPHANUMERIC_ID:
                if "pattern" not in cfg:
                    errors.append(f"ALPHANUMERIC_ID {p}: missing pattern (regex string)")
            elif ext_type == ArgExtractorType.NUMBER:
                if "units" not in cfg: errors.append(f"NUMBER {p}: missing units")
        if errors:
            raise ToolSchemaValidationError(
                f"Tool '{schema.name}' failed:\n" + "\n".join(f"  - {e}" for e in errors)
            )


# ── Tool Registry ─────────────────────────────────────────────────────────────

class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, ToolSchema] = {}
        self._functions: Dict[str, Callable] = {}

    def register(self, name, description, parameters, required, example, docstring,
                 func, tool_type=ToolType.KNOWLEDGE, arg_sources=None, output_keys=None,
                 explicit_keywords=None, arg_extractors=None):
        schema = ToolSchema(
            name=name, description=description, parameters=parameters,
            required=required, example=example, docstring=docstring,
            tool_type=tool_type,
            arg_sources=arg_sources or {},
            output_keys=output_keys or {},
            explicit_keywords=explicit_keywords or [],
            arg_extractors=arg_extractors or {},
        )
        ToolSchemaValidator.validate(schema)
        self._tools[name] = schema
        self._functions[name] = func

    def get_tool(self, name: str) -> Optional[ToolSchema]:
        return self._tools.get(name)

    def get_function(self, name: str) -> Optional[Callable]:
        return self._functions.get(name)

    async def call(self, name: str, arguments: Dict) -> Any:
        func = self._functions.get(name)
        if not func:
            raise ValueError(f"Unknown tool: {name}")
        if asyncio.iscoroutinefunction(func):
            return await func(**arguments)
        return func(**arguments)

    def get_all_names(self) -> List[str]:
        return list(self._tools.keys())

    def get_all_tools(self) -> List[ToolSchema]:
        return list(self._tools.values())

    def get_all_context_keys(self) -> List[str]:
        keys = set()
        for t in self._tools.values():
            keys.update(t.arg_sources.values())
            keys.update(t.output_keys.keys())
        return sorted(keys)

    def get_all_arg_extractors(self) -> Dict[str, tuple]:
        extractors = {}
        for tool in self._tools.values():
            for param, spec in tool.arg_extractors.items():
                ctx_key = tool.arg_sources.get(param, param)
                if ctx_key not in extractors:
                    extractors[ctx_key] = spec
        return extractors

    def __len__(self):
        return len(self._tools)

    def __repr__(self):
        return f"ToolRegistry({len(self._tools)} tools: {list(self._tools.keys())})"