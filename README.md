# nanoathens — Declarative DataFlow Agent SDK

A schema-driven, collision-free agent framework where **LLMs extract values (not plans)** and a **static DAG resolves deterministic execution paths**.

## Install

```bash
pip install nanoathens                     # Core SDK only (no GPU deps)
pip install nanoathens[medgemma]           # + MedGemma support
pip install nanoathens[all]                # Everything
```

Or install from source:
```bash
pip install -e .
```

## Quick Start

```python
from nanoathens import (
    ToolRegistry, ToolSchema, ToolType, ArgExtractorType,
    DeclarativeDataFlowAgent, run_medgemma, load_medgemma,
)

# 1. Build your tool registry
registry = ToolRegistry()
registry.register(
    name="my_tool",
    description="Does something useful",
    tool_type=ToolType.COMPUTATION,
    arg_sources={"input_val": "user_input"},
    arg_extractor={"name": "my_tool", "arguments": {"input_val": {"type": ArgExtractorType.LLM}}},
    func=lambda input_val: f"Result for {input_val}",
    output_keys={"output_val": "string"},
)

# 2. Create the agent
agent = DeclarativeDataFlowAgent(
    registry=registry,
    reasoning_caller=run_medgemma,  # Uses stub if MedGemma not loaded
)

# 3. Run
import asyncio
result = asyncio.run(agent.run("Process my input", target_key="output_val"))
print(result["response"])
```

## Architecture

```
User Query
    │
    ▼
┌─────────────────────┐
│  LLMValueExtractor   │  ← Extract values from query
│  ContextBank         │  ← Accumulate context
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│  GoalKeyResolver     │  ← Map query → target key
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│  DataFlowEngine      │  ← Resolve DAG path
│  (collision-free)    │  ← Backward DFS
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│  GroundedArgFiller   │  ← Fill args from context
│  Tool Execution      │  ← Run each tool in order
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│  Synthesis           │  ← LLM summarizes results
└─────────────────────┘
```

## Key Properties

- **Collision-free DAG**: No output key appears in any tool's input sources
- **Deterministic**: Same query always produces same execution plan
- **Minimum LLM calls**: 2 (extraction + goal resolution) + 1 per tool for arg filling
- **Explicit null_plan**: Returns registry gap information if no path exists
- **Domain-agnostic**: Works for any domain (oncology, radiology, etc.)

## Modules

| Module | Classes |
|--------|---------|
| `core` | `ToolType`, `ArgExtractorType`, `ToolSchema`, `ToolRegistry` |
| `context` | `ContextBank`, `LLMValueExtractor`, `BaseValueExtractor` |
| `filler` | `GroundedArgumentFiller` |
| `engine` | `DataFlowEngine` |
| `resolver` | `GoalKeyResolver` |
| `agent` | `DeclarativeDataFlowAgent`, `ConfigurableOrchestrator` |
| `session` | `SessionStore`, `SESSION_STORE` |
| `inference` | `run_medgemma`, `load_medgemma`, `set_pipeline` |
| `retriever` | `BM25ToolRetriever` |
