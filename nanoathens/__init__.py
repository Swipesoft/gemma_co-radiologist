"""
nanoathens — Declarative DataFlow Agent SDK
═════════════════════════════════════════════
A schema-driven, collision-free agent framework where LLMs extract values
(not plans), and a static DAG resolves deterministic execution paths.

Usage:
    from nanoathens import (
        ToolRegistry, ToolSchema, ToolType, ArgExtractorType,
        ContextBank, LLMValueExtractor,
        GroundedArgumentFiller,
        DataFlowEngine,
        GoalKeyResolver,
        DeclarativeDataFlowAgent, ToolRAGAgent,
        SessionStore, SESSION_STORE,
        BM25ToolRetriever,
        run_medgemma, load_medgemma, set_pipeline,
    )
"""

__version__ = "0.2.2"

# Core tool infrastructure
from .core import (
    ToolType,
    ArgExtractorType,
    ToolSchema,
    ToolSchemaValidator,
    ToolSchemaValidationError,
    ToolRegistry,
)

# Context and value extraction
from .context import (
    BaseValueExtractor,
    LLMValueExtractor,
    ContextBank,
)

# Argument filling
from .filler import GroundedArgumentFiller

# BM25 tool retriever (optional)
from .retriever import BM25ToolRetriever

# DataFlow engine (DAG)
from .engine import DataFlowEngine

# Goal resolution
from .resolver import GoalKeyResolver

# Main agent
from .agent import DeclarativeDataFlowAgent, ToolRAGAgent

# Session state
from .session import SessionStore, SESSION_STORE

# Model inference
from .inference import run_medgemma, load_medgemma, set_pipeline, _stub_llm

__all__ = [
    # Core
    "ToolType", "ArgExtractorType", "ToolSchema",
    "ToolSchemaValidator", "ToolSchemaValidationError", "ToolRegistry",
    # Context
    "BaseValueExtractor", "LLMValueExtractor", "ContextBank",
    # Filler
    "GroundedArgumentFiller",
    # Retriever
    "BM25ToolRetriever",
    # Engine
    "DataFlowEngine",
    # Resolver
    "GoalKeyResolver",
    # Agent
    "DeclarativeDataFlowAgent", "ToolRAGAgent",
    # Session
    "SessionStore", "SESSION_STORE",
    # Inference
    "run_medgemma", "load_medgemma", "set_pipeline",
]
