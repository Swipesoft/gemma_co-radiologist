# nanoathens — Declarative DataFlow Agent SDK

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![Version](https://img.shields.io/badge/version-0.2.0-orange.svg)](#changelog)

A schema-driven, collision-free agent framework where **LLMs extract values (not plans)** and a **static DAG resolves deterministic execution paths**.

> Developed by **Emmanuel Uramah** ([@boochi](https://github.com/boochi))

---

## Install

```bash
pip install nanoathens                     # Core SDK only (no GPU deps)
pip install nanoathens[medgemma]           # + MedGemma support
pip install nanoathens[all]                # Everything
```

Or install from source:
```bash
git clone https://github.com/boochi/nanoathens.git
cd nanoathens
pip install -e .
```

## Quick Start

```python
from nanoathens import (
    ToolRegistry, ToolType, ArgExtractorType,
    DeclarativeDataFlowAgent, run_medgemma, load_medgemma,
)

# 1. Build your tool registry
registry = ToolRegistry()
registry.register(
    name="my_tool",
    description="Does something useful",
    parameters={"input_val": "The user input string"},
    required=["input_val"],
    example={"input_val": "sample input"},
    docstring="Processes user input and returns a result.",
    tool_type=ToolType.COMPUTATION,
    func=lambda input_val: f"Result for {input_val}",
    arg_sources={"input_val": "user_input"},
    output_keys={"output_val": "string"},
    arg_extractors={"input_val": (ArgExtractorType.QUOTED, {})},
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
print(result["raw_results"])  # Direct tool outputs
```

## Architecture

```
User Query
    │
    ▼
┌─────────────────────┐
│  LLMValueExtractor   │  ← Extract values from query (not plans)
│  ContextBank         │  ← Accumulate context across tools
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│  GoalKeyResolver     │  ← Map query → target key (LLM + fuzzy fallback)
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│  DataFlowEngine      │  ← Resolve DAG path (backward DFS)
│  (collision-free)    │  ← output_keys ∩ arg_sources = ∅
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│  GroundedArgFiller   │  ← Fill args from context + LLM fallback
│  Tool Execution      │  ← Run each tool in deterministic order
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│  Synthesis           │  ← LLM summarizes results
│  raw_results         │  ← Direct tool outputs preserved
└─────────────────────┘
```

## Key Properties

- **Collision-free DAG**: No output key appears in any tool's input sources — guarantees no self-loops
- **Deterministic**: Same query always produces same execution plan
- **Minimum LLM calls**: 2 (extraction + goal resolution) + 1 per tool for arg filling
- **Explicit null_plan**: Returns registry gap information if no path exists
- **Schema-driven extraction**: `ENUM`, `QUOTED`, `LANGUAGE`, `ALPHANUMERIC_ID`, `NUMERIC_ID`, `NUMBER` extractors — no blind LLM parsing
- **One-shot examples**: Every tool carries an `example` dict that guides LLM argument extraction
- **Domain-agnostic**: Works for any domain (oncology, radiology, NLP, etc.)

## Agents

| Agent | Strategy | Best for |
|-------|----------|----------|
| `DeclarativeDataFlowAgent` | Static DAG resolution via backward DFS | Production — deterministic, auditable |
| `ToolRAGAgent` | BM25 retrieval + LLM planning | Exploration — flexible, handles ambiguity |

## Modules

| Module | Key Classes | Purpose |
|--------|-------------|---------|
| `core` | `ToolType`, `ArgExtractorType`, `ToolSchema`, `ToolRegistry` | Tool infrastructure with collision-free validation |
| `context` | `ContextBank`, `LLMValueExtractor` | Value extraction + context accumulation |
| `filler` | `GroundedArgumentFiller` | Fills tool args from context with LLM fallback |
| `engine` | `DataFlowEngine` | Collision-free DAG builder + backward DFS |
| `resolver` | `GoalKeyResolver` | Maps query → target key (LLM + fuzzy matching) |
| `agent` | `DeclarativeDataFlowAgent`, `ToolRAGAgent` | Main orchestrators |
| `session` | `SessionStore`, `SESSION_STORE` | Multi-turn session state manager |
| `inference` | `run_medgemma`, `load_medgemma`, `set_pipeline` | Model adapter (any HF pipeline) |
| `retriever` | `BM25ToolRetriever` | Optional BM25-based tool recommender |

## Tool Registration

Tools are registered with full schemas including parameters, examples, and typed extractors:

```python
registry.register(
    name="retrieve_similar_images",
    description="Retrieve similar medical images using MedSigLIP + FAISS",
    parameters={
        "patient_image": "Path to the patient image file",
        "image_type": "Modality: xray|ct|mri",
    },
    required=["patient_image", "image_type"],
    example={"patient_image": "/data/patient_001.jpg", "image_type": "xray"},
    docstring="Embeds query image with MedSigLIP-448 and retrieves top-K similar cases from FAISS index.",
    tool_type=ToolType.RETRIEVAL,
    func=_retrieve_similar_images,
    arg_sources={"patient_image": "patient_image_input", "image_type": "image_type_input"},
    output_keys={"knn_images": "JSON array of similar images with scores"},
    explicit_keywords=["similar", "retrieve", "knn", "search"],
    arg_extractors={
        "patient_image": (ArgExtractorType.QUOTED, {}),
        "image_type": (ArgExtractorType.ENUM, {"values": ["xray", "ct", "mri"]}),
    },
)
```

## Use with MedGemma

```python
from nanoathens import set_pipeline, run_medgemma
from transformers import pipeline as hf_pipeline
import torch

# Load your model externally
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = hf_pipeline(
    "image-text-to-text",
    model="google/medgemma-1.5-4b-it",
    device=device,
    torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
)

# Register with nanoathens
set_pipeline(pipe)

# run_medgemma() now uses your pipeline
result = run_medgemma(
    messages=[{"role": "user", "content": [{"type": "text", "text": "Analyze this image"}]}],
    max_new_tokens=512,
)
```

## Changelog

### v0.2.0 (2026-02-22)
- **BREAKING**: Restored full `ToolSchema` with `parameters`, `required`, `example`, `docstring`, `explicit_keywords`, `arg_extractors`
- **BREAKING**: `ArgExtractorType` now includes `ENUM`, `LANGUAGE` (restored from SOTA benchmark)
- Added `ToolRAGAgent` (BM25-based agent) alongside `DeclarativeDataFlowAgent`
- Added `raw_results` to agent return dict — direct access to tool outputs
- Added `difflib` fuzzy matching in `GoalKeyResolver`
- Added `set_pipeline()` / `set_stub()` for clean model injection
- Fixed token budget: all internal LLM calls default to 512 tokens (was 4096)
- Fixed HF `pad_token_id` warning in inference
- Fixed `source_lang_code` / `target_lang_code` in message formatting

### v0.1.0 (2026-02-21)
- Initial release — extracted from monolithic notebook
- 10 modules, zero hard dependencies

## Citation

If you use nanoathens in your research, please cite:

```bibtex
@software{uramah2026nanoathens,
  author       = {Uramah, Emmanuel},
  title        = {nanoathens: Declarative DataFlow Agent SDK},
  year         = {2026},
  url          = {https://github.com/boochi/nanoathens},
  version      = {0.2.0},
  description  = {A schema-driven, collision-free agent framework for deterministic tool orchestration}
}
```

## License

Apache 2.0 — see [LICENSE](LICENSE) for details.

---

**Author**: Emmanuel Uramah  
**Contact**: [GitHub](https://github.com/boochi)  
**Built with**: Python, HuggingFace Transformers, FAISS, MedGemma