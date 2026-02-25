# nanoathens — Declarative DataFlow Agent SDK

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![Version](https://img.shields.io/badge/version-0.2.0-orange.svg)](#changelog)

A schema-driven, collision-free agent framework where **LLMs extract values (not plans)** and a **static DAG resolves deterministic execution paths**. Built for orchestrating small language models like [MedGemma 4B](https://huggingface.co/google/medgemma-4b-it) in production clinical workflows.

> Developed by **Emmanuel Uramah** ([@Swipesoft](https://github.com/Swipesoft))

---

## Flagship Application: Gemma Co-Radiologist

nanoathens powers the **Gemma Co-Radiologist** — an agentic radiology workstation that orchestrates 16 tools through a collision-free dataflow graph to deliver autonomous, end-to-end diagnostic imaging analysis.

```
Patient Image + Query
        │
        ▼
┌──────────────────────────────────────────────────────────────────┐
│  DeclarativeDataFlowAgent (nanoathens SDK)                       │
│                                                                  │
│  retrieve_similar_images ──► few_shot_image_analysis             │
│          │                           │                           │
│          ▼                           ▼                           │
│  [MedSigLIP + FAISS]    verify_few_shot_image_analysis           │
│                                 │          │                     │
│                                 ▼          ▼                     │
│              synthesize_clinical_narrative  localize_abnormalities│
│                          │                        │              │
│    retrieve_ehr ─────────┤                        │              │
│         │                ▼                        │              │
│         └──► generate_soap_report                 │              │
│                     │                             │              │
│                     ▼                             │              │
│              build_pdf_soap_report ◄──────────────┘              │
│                     │                                            │
│                     ▼                                            │
│              deep_analysis (meta-tool aggregator)                 │
└──────────────────────────────────────────────────────────────────┘
        │
        ▼
  Clinical Narrative · Annotated Image · SOAP Report · PDF Export
```

**Key capabilities:**
- **Modality-adaptive analysis** — CXR, CT Head, CT Abdomen with modality-specific Pydantic schemas and prompts
- **Two workflow modes** — Structured (button-triggered progressive pipeline) and Autopilot (natural language → autonomous DAG resolution)
- **MedSigLIP-448 + FAISS** retrieval-augmented analysis with few-shot learning from CheXpert
- **Bounding box localization** using MedGemma's spatial prediction (Google reference implementation)
- **Longitudinal review** with prior study comparison from mock EHR database
- **Interactive revision** — clinicians can critique findings; the agent re-generates narrative, localization, SOAP, and PDF
- **BM25-powered follow-up chat** for targeted Q&A over analysis context

---

## Install

```bash
pip install nanoathens                     # Core SDK only (no GPU deps)
pip install nanoathens[medgemma]           # + MedGemma support
pip install nanoathens[all]                # Everything
```

Or install from source:
```bash
git clone https://github.com/Swipesoft/gemma_co-radiologist.git
cd gemma_co-radiologist
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

### How the DDA Differs from ReAct / Function-Calling Agents

| Property | ReAct / FC Agents | DeclarativeDataFlowAgent |
|----------|-------------------|--------------------------|
| **Who plans?** | LLM decides tool order at runtime | Static DAG resolves execution path at build time |
| **LLM role** | Plan + extract + execute | Extract values only — planning is deterministic |
| **Hallucination risk** | LLM may skip tools or invent wrong sequences | Impossible — path is resolved by backward DFS |
| **Auditability** | Opaque chain-of-thought | Full execution plan available before first tool runs |
| **Reproducibility** | Non-deterministic (temperature, sampling) | Deterministic — same inputs always yield same plan |

## Key Properties

- **Collision-free DAG**: No output key appears in any tool's input sources — guarantees no self-loops and termination
- **Deterministic**: Same query always produces same execution plan
- **Minimum LLM calls**: 2 (extraction + goal resolution) + 1 per tool for arg filling
- **Explicit null_plan**: Returns registry gap information if no path exists
- **Schema-driven extraction**: `ENUM`, `QUOTED`, `LANGUAGE`, `ALPHANUMERIC_ID`, `NUMERIC_ID`, `NUMBER` extractors — no blind LLM parsing
- **Hallucination guard**: Tool-produced keys are stripped from initial context to prevent the LLM from fabricating outputs (ChexPertBench fix)
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
| `core` | `ToolType`, `ArgExtractorType`, `ToolSchema`, `ToolSchemaValidator`, `ToolRegistry` | Tool infrastructure with collision-free validation |
| `context` | `ContextBank`, `LLMValueExtractor` | Value extraction + context accumulation |
| `filler` | `GroundedArgumentFiller` | Fills tool args from context with LLM fallback |
| `engine` | `DataFlowEngine` | Collision-free DAG builder + backward DFS resolver |
| `resolver` | `GoalKeyResolver` | Maps query → target key (LLM + `difflib` fuzzy matching) |
| `agent` | `DeclarativeDataFlowAgent`, `ToolRAGAgent` | Main orchestrators |
| `session` | `SessionStore`, `SESSION_STORE` | Multi-turn session state manager |
| `inference` | `run_medgemma`, `load_medgemma`, `set_pipeline` | Model adapter (any HF pipeline) |
| `retriever` | `BM25ToolRetriever` | Optional BM25-based tool recommender |

## Tool Registration

Tools are registered with full schemas including typed parameters, one-shot examples, dataflow wiring (`arg_sources` / `output_keys`), and extraction hints:

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

### Collision-Free Invariant

The `ToolSchemaValidator` enforces that no tool's `output_keys` overlap with its `arg_sources`. This guarantees the dataflow graph is a proper DAG with no self-loops, so backward DFS always terminates:

```python
# This would FAIL validation:
registry.register(
    ...
    arg_sources={"input": "some_key"},
    output_keys={"some_key": "..."},  # COLLISION — same key as input!
)
# ToolSchemaValidationError: COLLISION: output_key 'some_key' also in arg_sources
```

## Co-Radiologist Tool Registry (16 Tools)

The Gemma Co-Radiologist registers 16 tools across three categories:

| Tool | Type | Inputs (arg_sources) | Outputs (output_keys) |
|------|------|---------------------|----------------------|
| `retrieve_similar_images` | RETRIEVAL | `patient_image_input`, `image_type_input` | `knn_images` |
| `few_shot_image_analysis` | KNOWLEDGE | `knn_images`, `patient_image_input`, `body_part_input` | `few_shot_analysis` |
| `verify_few_shot_image_analysis` | KNOWLEDGE | `few_shot_analysis`, `patient_image_input`, `body_part_input` | `verified_analysis` |
| `synthesize_clinical_narrative` | KNOWLEDGE | `verified_analysis`, `patient_image_input`, `body_part_input` | `clinical_narrative` |
| `localize_abnormalities` | COMPUTATION | `verified_analysis`, `patient_image_input` | `localized_image` |
| `retrieve_patient_previous_images` | RETRIEVAL | `patient_id_input`, `image_type_input` | `previous_images` |
| `run_longitudinal_review` | KNOWLEDGE | `patient_image_input`, `previous_images`, `verified_analysis` | `longitudinal_review` |
| `revise_report` | KNOWLEDGE | `clinician_critique_input`, `verified_analysis`, `longitudinal_review` | `updated_report` |
| `retrieve_ehr` | RETRIEVAL | `patient_id_input` | `medical_records` |
| `generate_soap_report` | KNOWLEDGE | `medical_records`, `verified_analysis` | `soap_report` |
| `build_pdf_soap_report` | COMPUTATION | `soap_report`, `localized_image`, `patient_image_input` | `generated_pdf_path` |
| `retrieve_session_memory` | RETRIEVAL | `session_id_input` | `session_context` |
| `qa_followup` | KNOWLEDGE | `followup_question_input`, `session_context` | `qa_response` |
| `store_session_memory` | COMPUTATION | `session_id_input`, `data_to_store_input` | `session_save_confirmation` |
| `classify_image_modality` | KNOWLEDGE | `patient_image_input` | `image_classification` |
| `deep_analysis` | COMPUTATION | `clinical_narrative`, `localized_image`, `soap_report`, `generated_pdf_path` | `deep_analysis_result` |

### DAG Resolution Example

When the DDA targets `deep_analysis_result`, backward DFS resolves a 10-tool execution plan:

```
Available: {patient_image_input, image_type_input, body_part_input, patient_id_input}
Target:    deep_analysis_result

Resolved plan (10 tools):
  1. retrieve_similar_images        → knn_images
  2. few_shot_image_analysis        → few_shot_analysis
  3. verify_few_shot_image_analysis → verified_analysis
  4. synthesize_clinical_narrative  → clinical_narrative
  5. localize_abnormalities         → localized_image
  6. retrieve_ehr                   → medical_records
  7. generate_soap_report           → soap_report
  8. build_pdf_soap_report          → generated_pdf_path
  9. deep_analysis                  → deep_analysis_result
```

Targeting a different key (e.g., `verified_analysis`) automatically resolves a shorter 3-tool sub-plan — the DDA only executes what's needed.

## Use with MedGemma

```python
from nanoathens import set_pipeline, run_medgemma
from transformers import pipeline as hf_pipeline
import torch

# Load your model externally
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = hf_pipeline(
    "image-text-to-text",
    model="google/medgemma-4b-it",
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

## Session Management

nanoathens includes `SessionStore` for multi-turn agent conversations. The `RadiologySessionAgent` wrapper (in the notebook) demonstrates session-aware execution with automatic follow-up detection:

```python
from nanoathens import SessionStore

store = SessionStore()
session_id = store.create_session()

# Save context from tool execution
store.save_context(session_id, "patient_image_input", "/path/to/image.png")
store.save_run_result(session_id, "Analyze this X-ray", result_dict)

# Retrieve for follow-up queries
summary = store.get_session_summary(session_id)
```

## Repository Structure

```
gemma_co-radiologist/
├── nanoathens/              # SDK source
│   ├── __init__.py          # Public API exports
│   ├── core.py              # ToolType, ToolSchema, ToolRegistry, validators
│   ├── context.py           # ContextBank, LLMValueExtractor
│   ├── filler.py            # GroundedArgumentFiller
│   ├── engine.py            # DataFlowEngine (collision-free DAG + backward DFS)
│   ├── resolver.py          # GoalKeyResolver (LLM + fuzzy matching)
│   ├── agent.py             # DeclarativeDataFlowAgent, ToolRAGAgent
│   ├── session.py           # SessionStore
│   ├── inference.py         # run_medgemma, set_pipeline, model adapter
│   └── retriever.py         # BM25ToolRetriever
├── app/                     # Application layer
├── notebooks/               # Kaggle competition notebooks
├── pyproject.toml
├── setup.py
├── requirements.txt
└── README.md
```

## Changelog

### v0.2.0 (2026-02-22)
- **BREAKING**: Restored full `ToolSchema` with `parameters`, `required`, `example`, `docstring`, `explicit_keywords`, `arg_extractors`
- **BREAKING**: `ArgExtractorType` now includes `ENUM`, `LANGUAGE` (restored from SOTA benchmark)
- Added `ToolRAGAgent` (BM25-based agent) alongside `DeclarativeDataFlowAgent`
- Added `raw_results` to agent return dict — direct access to tool outputs
- Added `difflib` fuzzy matching in `GoalKeyResolver`
- Added `set_pipeline()` / `set_stub()` for clean model injection
- Added hallucination guard: drops tool-produced keys from initial variable extraction
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
  url          = {https://github.com/Swipesoft/gemma_co-radiologist},
  version      = {0.2.0},
  description  = {A schema-driven, collision-free agent framework for deterministic tool orchestration}
}
```

## License

Apache 2.0 — see [LICENSE](LICENSE) for details.

---

**Author**: Emmanuel Uramah
**Repository**: [github.com/Swipesoft/gemma_co-radiologist](https://github.com/Swipesoft/gemma_co-radiologist)
**Built with**: Python · HuggingFace Transformers · MedGemma 4B · MedSigLIP-448 · FAISS · Gradio