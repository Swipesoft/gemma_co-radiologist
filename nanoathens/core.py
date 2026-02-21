"""
athena_dda.core — Tool Infrastructure
══════════════════════════════════════
ToolType, ArgExtractorType, ToolSchema, ToolSchemaValidator, ToolRegistry.
"""

import asyncio
import inspect
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
    LLM             = "llm"
    CODE            = "code"
    HUMAN           = "human"
    QUOTED          = "quoted"
    ALPHANUMERIC_ID = "alphanumeric_id"
    NUMERIC_ID      = "numeric_id"
    NUMBER          = "number"


# ── Tool Schema ───────────────────────────────────────────────────────────────

@dataclass
class ToolSchema:
    name: str
    description: str
    tool_type: ToolType
    arg_sources: Dict[str, str]
    arg_extractor: Dict[str, Any]
    func: Callable
    output_keys: Dict[str, str]
    context_keys: List[str] = field(default_factory=list)

    def __post_init__(self):
        if self.context_keys is None:
            self.context_keys = []

    @property
    def parameters(self) -> Dict[str, str]:
        """Derive parameter descriptions from arg_extractor."""
        if not self.arg_extractor:
            return {k: "string" for k in self.arg_sources}
        args = self.arg_extractor.get("arguments", {})
        return {
            k: v.get("description", "string") if isinstance(v, dict) else str(v)
            for k, v in args.items()
        }

    @property
    def required(self) -> List[str]:
        """All arg_sources keys are required by default."""
        return list(self.arg_sources.keys())


# ── Schema Validator ──────────────────────────────────────────────────────────

class ToolSchemaValidationError(Exception):
    pass


class ToolSchemaValidator:
    REQUIRED_FIELDS = [
        "name", "description", "tool_type", "arg_sources",
        "arg_extractor", "func", "output_keys",
    ]

    @classmethod
    def validate(cls, schema: ToolSchema) -> List[str]:
        errors = []
        for fld in cls.REQUIRED_FIELDS:
            if getattr(schema, fld, None) is None:
                errors.append(f"Missing required field: {fld}")

        if schema.arg_sources and schema.output_keys:
            for out_key in schema.output_keys:
                if out_key in schema.arg_sources.values():
                    errors.append(
                        f"Collision: output_key '{out_key}' appears in arg_sources"
                    )

        if schema.func is not None:
            sig = inspect.signature(schema.func)
            func_params = set(sig.parameters.keys())
            schema_params = set(schema.arg_sources.keys())
            if not schema_params.issubset(func_params):
                missing = schema_params - func_params
                errors.append(
                    f"arg_sources keys {missing} not in function signature"
                )

        if errors:
            raise ToolSchemaValidationError(
                f"Validation failed for '{schema.name}': {errors}"
            )
        return errors


# ── Tool Registry ─────────────────────────────────────────────────────────────

class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, ToolSchema] = {}

    def register(self, **kwargs):
        schema = ToolSchema(**kwargs)
        ToolSchemaValidator.validate(schema)
        self._tools[schema.name] = schema
        return schema

    def get_tool(self, name: str) -> Optional[ToolSchema]:
        return self._tools.get(name)

    def get_all_names(self) -> List[str]:
        return list(self._tools.keys())

    def get_all_tools(self) -> List[ToolSchema]:
        return list(self._tools.values())

    def get_all_context_keys(self) -> List[str]:
        keys = set()
        for tool in self._tools.values():
            keys.update(tool.arg_sources.values())
            keys.update(tool.output_keys.keys())
        return sorted(keys)

    def get_all_arg_extractors(self) -> Dict[str, tuple]:
        """Return {context_key: (ArgExtractorType, config)} for schema-based extraction."""
        extractors = {}
        for tool in self._tools.values():
            args = tool.arg_extractor.get("arguments", {}) if tool.arg_extractor else {}
            for param_name, param_cfg in args.items():
                if not isinstance(param_cfg, dict):
                    continue
                ctx_key = tool.arg_sources.get(param_name, param_name)
                ext_type = param_cfg.get("type", ArgExtractorType.LLM)
                extractors[ctx_key] = (ext_type, param_cfg)
        return extractors

    async def call(self, name: str, args: dict) -> Any:
        tool = self._tools.get(name)
        if tool is None:
            raise ValueError(f"Tool '{name}' not found in registry")
        if asyncio.iscoroutinefunction(tool.func):
            return await tool.func(**args)
        return tool.func(**args)

    def __len__(self):
        return len(self._tools)

    def __repr__(self):
        return f"ToolRegistry({len(self._tools)} tools: {list(self._tools.keys())})"
