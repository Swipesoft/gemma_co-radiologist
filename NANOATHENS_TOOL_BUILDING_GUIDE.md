# nanoathens Tool Building Guide
## A Skill File for Developers and LLM Agents

> **Version**: 0.2.0  
> **Author**: Emmanuel Uramah  
> **Purpose**: Complete reference for building tools that work correctly with the nanoathens Declarative DataFlow Agent SDK. Put this file in your LLM context window when asking an AI to build nanoathens tools.

---

## 1. How the DDA Agent Works (Mental Model)

The DDA agent is NOT a planner. It follows a fixed 5-step pipeline:

```
Step 1: Extract values from user query (LLM + schema-based extractors)
Step 2: Resolve which output key the user wants (goal resolution)
Step 3: Build execution plan via backward DFS on the DAG (deterministic)
Step 4: Execute tools in order, passing outputs forward through ContextBank
Step 5: Synthesize a prose response from collected tool results
```

**Critical insight**: The LLM never decides which tools to call. The DAG does. The LLM only extracts values from text. This means your tool schemas must be precise — the DAG resolver depends entirely on `arg_sources` and `output_keys` to build the execution plan.

---

## 2. ToolSchema Anatomy

Every tool is registered with a `ToolSchema` containing these fields:

```python
registry.register(
    # ── IDENTITY ──
    name="tool_name",                    # Unique identifier (snake_case)
    description="What this tool does",   # 1-line summary for LLM/BM25
    docstring="Detailed explanation...",  # Rich text for BM25 indexing + LLM context
    tool_type=ToolType.RETRIEVAL,        # RETRIEVAL | COMPUTATION | KNOWLEDGE

    # ── FUNCTION ──
    func=my_function,                    # The actual Python callable

    # ── PARAMETERS ──
    parameters={                         # Dict of {param_name: description}
        "patient_image": "Path to the patient image file",
        "image_type": "Imaging modality: xray|ct|mri",
    },
    required=["patient_image", "image_type"],  # Which params are mandatory

    # ── ONE-SHOT EXAMPLE ──
    example={                            # Concrete example for LLM arg filling
        "patient_image": "/data/patient_001.jpg",
        "image_type": "xray",
    },

    # ── DATA FLOW (the DAG wiring) ──
    arg_sources={                        # param_name → context_key mapping
        "patient_image": "patient_image_input",
        "image_type": "image_type_input",
    },
    output_keys={                        # context_key → type description
        "knn_images": "JSON array of similar images",
    },

    # ── EXTRACTION HINTS ──
    explicit_keywords=["similar", "retrieve", "knn"],  # BM25 matching boost
    arg_extractors={                     # How to extract each param from text
        "patient_image": (ArgExtractorType.QUOTED, {}),
        "image_type": (ArgExtractorType.ENUM, {"values": ["xray", "ct", "mri"]}),
    },
)
```

### Field Reference

| Field | Type | Required | Purpose |
|-------|------|:--------:|---------|
| `name` | str | ✓ | Unique tool identifier, used as DAG node name |
| `description` | str | ✓ | Short summary — shown to LLM during planning |
| `parameters` | Dict[str, str] | ✓ | Parameter names → human-readable descriptions |
| `required` | List[str] | ✓ | Parameters that MUST be filled for tool to run |
| `example` | Dict[str, Any] | ✓ | One-shot example — guides LLM argument extraction |
| `docstring` | str | ✓ | Detailed description — used by BM25 indexing |
| `tool_type` | ToolType | ✓ | Category: RETRIEVAL, COMPUTATION, or KNOWLEDGE |
| `func` | Callable | ✓ | The Python function to execute |
| `arg_sources` | Dict[str, str] | ✓ | Maps function params to ContextBank keys |
| `output_keys` | Dict[str, str] | ✓ | Keys this tool produces in the ContextBank |
| `explicit_keywords` | List[str] | ○ | Keywords for BM25 tool retrieval |
| `arg_extractors` | Dict[str, tuple] | ○ | Per-param extraction strategy |

---

## 3. The Collision-Free Invariant (MOST IMPORTANT RULE)

**RULE**: No output key may appear in any tool's `arg_sources` values.

```
output_keys ∩ arg_source_values = ∅   (ALWAYS)
```

This guarantees the DAG has no self-loops and backward DFS always terminates.

### Why This Matters

The `DataFlowEngine` builds a directed graph:
- **Edges**: Tool A's output key → Tool B's arg_source → Tool B
- **Resolution**: Backward DFS from target key to find which tools to run

If an output key appears in arg_sources of the SAME tool, you get a cycle. The validator catches this at registration time.

### Example: VALID Chain

```
Tool A:  arg_sources={} → output_keys={"knn_images": "..."}
Tool B:  arg_sources={"knn_images": "knn_images"} → output_keys={"analysis": "..."}
Tool C:  arg_sources={"analysis": "analysis"} → output_keys={"verified": "..."}
```

DAG: `A → B → C` (no collisions)

### Example: INVALID (will throw ToolSchemaValidationError)

```
Tool X:  arg_sources={"data": "result_key"} → output_keys={"result_key": "..."}
```

`result_key` appears in both arg_sources AND output_keys → COLLISION.

---

## 4. arg_sources: Wiring the DAG

`arg_sources` is the most critical field. It maps function parameter names to **ContextBank keys**.

```python
arg_sources = {
    "patient_image": "patient_image_input",   # function param → context key
    "image_type": "image_type_input",
}
```

### Two Types of Context Keys

**Source keys (leaf inputs)**: Keys that come from the user query. No tool produces them.
- Named with `_input` suffix by convention: `patient_image_input`, `image_type_input`
- Extracted by `ContextBank.set_goal()` from the user's natural language query
- The DDA agent treats these as "already available" when building the plan

**Intermediate keys (tool outputs)**: Keys produced by tools and consumed by downstream tools.
- Named descriptively: `knn_images`, `few_shot_analysis`, `verified_analysis`
- Created when a tool runs and its output is stored under `output_keys`

### How the DAG Resolver Uses This

When you ask for `target_key="verified_analysis"`:

```
1. Who produces "verified_analysis"? → verify_few_shot_image_analysis
2. What does it need? → arg_sources: {"analysis": "few_shot_analysis", "patient_image": "patient_image_input"}
3. "patient_image_input" is a source key → available ✓
4. "few_shot_analysis" is NOT available → who produces it?
5. few_shot_image_analysis produces it → needs "knn_images" + "patient_image_input"
6. "knn_images" → produced by retrieve_similar_images → needs "patient_image_input" + "image_type_input"
7. Both are source keys → available ✓

Final plan: [retrieve_similar_images, few_shot_image_analysis, verify_few_shot_image_analysis]
```

### Naming Conventions

```
Source keys (from user):     *_input      → patient_image_input, image_type_input
Intermediate keys (tools):   descriptive  → knn_images, few_shot_analysis
Final output keys:           descriptive  → verified_analysis, soap_report
Raw tool results:            _raw_*       → _raw_retrieve_similar_images (auto-created)
```

---

## 5. output_keys: What Your Tool Produces

```python
output_keys = {"knn_images": "JSON array of similar images with scores"}
```

### Rules

1. **Keys must be unique across the entire registry.** Two tools cannot produce the same key.
2. **Keys must NOT appear in any tool's arg_sources values** (collision-free invariant).
3. **Single output key is the common pattern.** Most tools produce exactly one thing.
4. **The value description is for documentation only** — it doesn't affect execution.

### How Output Storage Works (ContextBank)

When a tool executes, the result flows through `add_tool_result()`:

```python
# Single output key (most common case):
#   The raw tool return value IS the output
#   Stored directly: parsed_values["knn_images"] = result

# Multiple output keys + dict result:
#   Each key mapped from the result dict
#   parsed_values["diagnosis"] = result["diagnosis"]
#   parsed_values["confidence"] = result["confidence"]

# Multiple output keys + string result:
#   Attempts JSON parse, then maps keys
#   Falls back to storing full result under each key
```

### Important: Single-Output Tools

If your tool has 1 output key, its return value is stored directly under that key. This means:

```python
# Tool returns a JSON string:
def my_tool():
    return json.dumps({"data": [1, 2, 3]})

# The ENTIRE string is stored as the value of the output key
# The next tool receives this string, not a parsed dict
```

The next tool in the chain must handle the raw return type. If you return a JSON string, the consumer should `json.loads()` it. If you return a dict, the consumer receives a dict.

**Best practice**: Always return strings from tools. Strings are safe to store, pass through context, and display in logs. Parse inside the consuming tool.

---

## 6. ArgExtractorType: How Values Are Extracted from Queries

The `arg_extractors` field tells the ContextBank HOW to pull values from the user's natural language query — without needing an LLM call.

```python
arg_extractors = {
    "param_name": (ArgExtractorType.ENUM, {"values": ["xray", "ct", "mri"]}),
}
```

### Available Extractor Types

#### ENUM — Fixed set of values
```python
"image_type": (ArgExtractorType.ENUM, {"values": ["xray", "ct", "mri"]})

# Query: "Analyze this xray" → extracts "xray"
# Also handles aliases: "chest" → matches via _ENUM_ALIASES
```
**Use when**: Parameter has a small, known set of valid values.
**Config**: `{"values": ["val1", "val2", ...]}` — list of allowed values.

#### QUOTED — Value wrapped in quotes
```python
"patient_image": (ArgExtractorType.QUOTED, {})

# Query: "Patient image: '/data/img.jpg'" → extracts "/data/img.jpg"
# Works with single or double quotes
```
**Use when**: Value is a file path, name, or any string the user wraps in quotes.
**Config**: Empty dict `{}`.
**Critical**: Users must quote the value in their query for this to work. Format queries like:
```
"Analyze this chest X-ray. Patient image: '/path/to/image.jpg' Image type: xray"
```

#### LANGUAGE — Mapped natural language terms
```python
"scan_type": (ArgExtractorType.LANGUAGE, {
    "role": "radiologist",
    "mapping": {"chest x-ray": "cxr", "ct scan": "ct", "mri scan": "mri"}
})

# Query: "I need a chest x-ray analysis" → extracts "cxr"
```
**Use when**: Users describe things in natural language that map to specific codes.
**Config**: `{"role": "domain_expert", "mapping": {"phrase": "code", ...}}`.

#### ALPHANUMERIC_ID — Pattern-matched IDs
```python
"patient_id": (ArgExtractorType.ALPHANUMERIC_ID, {
    "pattern": r"RAD\d{3}",
    "preceding_words": ["patient", "id"]
})

# Query: "Look up patient RAD042" → extracts "RAD042"
```
**Use when**: IDs follow a regex pattern (e.g., P001, MRI-003, RAD042).
**Config**: `{"pattern": "regex_string", "preceding_words": ["optional", "context"]}`.

#### NUMERIC_ID — Pure numeric identifiers
```python
"record_number": (ArgExtractorType.NUMERIC_ID, {
    "preceding_words": ["record", "case", "number"]
})

# Query: "Pull up record 42871" → extracts "42871"
```
**Use when**: ID is purely numeric, at least 2 digits.
**Config**: `{"preceding_words": ["word1", "word2"]}` — words that precede the number.

#### NUMBER — Numeric values with units
```python
"dose_mg": (ArgExtractorType.NUMBER, {
    "units": ["mg", "milligrams"]
})

# Query: "Administer 500mg of medication" → extracts 500.0
```
**Use when**: Extracting measurements, dosages, quantities.
**Config**: `{"units": ["unit1", "unit2"]}` — unit strings to match after the number.

### Extraction Priority

When `context.set_goal(query)` runs:

```
1. Schema-based extraction (ENUM matching from parameter descriptions)
2. arg_extractors (QUOTED, ALPHANUMERIC_ID, NUMERIC_ID, NUMBER)
3. LLM extraction (fallback — sends query to LLM with field names)
```

Schema-based and arg_extractors are **fast and deterministic**. LLM extraction is slow and unreliable (especially with small models). Always prefer typed extractors.

---

## 7. Building a Tool Chain (Step by Step)

### Step 1: Define your functions

```python
def _step_one(patient_image: str, image_type: str) -> str:
    """First processing step. Returns JSON string."""
    # ... your logic ...
    return json.dumps({"results": [...], "metadata": {...}})

def _step_two(step_one_output: str, patient_image: str) -> str:
    """Uses output from step one. Returns analysis string."""
    data = json.loads(step_one_output) if isinstance(step_one_output, str) else step_one_output
    # ... your logic ...
    return json.dumps({"analysis": "..."})

def _step_three(analysis: str, patient_image: str) -> str:
    """Final verification. Returns validated JSON."""
    # ... your logic ...
    return json.dumps({"verified": True, "findings": [...]})
```

### Step 2: Map the data flow on paper

```
Source keys:          patient_image_input, image_type_input
                              ↓                    ↓
Tool 1 (step_one):    patient_image_input + image_type_input → step_one_results
                              ↓                                       ↓
Tool 2 (step_two):    patient_image_input + step_one_results  → analysis_output
                              ↓                                       ↓
Tool 3 (step_three):  patient_image_input + analysis_output   → verified_output
```

### Step 3: Register tools

```python
from nanoathens import ToolRegistry, ToolType, ArgExtractorType

registry = ToolRegistry()

# Tool 1: Entry point
registry.register(
    name="step_one",
    description="First processing step — retrieves relevant data",
    parameters={
        "patient_image": "Path to patient image file",
        "image_type": "Modality: xray|ct|mri",
    },
    required=["patient_image", "image_type"],
    example={"patient_image": "/data/patient_001.jpg", "image_type": "xray"},
    docstring="Processes the input image and retrieves relevant reference data.",
    tool_type=ToolType.RETRIEVAL,
    func=_step_one,
    arg_sources={
        "patient_image": "patient_image_input",    # ← source key (from user)
        "image_type": "image_type_input",          # ← source key (from user)
    },
    output_keys={
        "step_one_results": "JSON with retrieved data",  # ← NEW key, produced here
    },
    explicit_keywords=["retrieve", "search", "find"],
    arg_extractors={
        "patient_image": (ArgExtractorType.QUOTED, {}),
        "image_type": (ArgExtractorType.ENUM, {"values": ["xray", "ct", "mri"]}),
    },
)

# Tool 2: Middle of chain
registry.register(
    name="step_two",
    description="Analyzes data using context from step one",
    parameters={
        "step_one_output": "JSON results from step_one tool",
        "patient_image": "Path to patient image file",
    },
    required=["step_one_output", "patient_image"],
    example={
        "step_one_output": '{"results": [...]}',
        "patient_image": "/data/patient_001.jpg",
    },
    docstring="Takes retrieval results and performs detailed analysis.",
    tool_type=ToolType.COMPUTATION,
    func=_step_two,
    arg_sources={
        "step_one_output": "step_one_results",     # ← intermediate key (from Tool 1)
        "patient_image": "patient_image_input",     # ← source key (reused)
    },
    output_keys={
        "analysis_output": "Analysis JSON string",  # ← NEW key, produced here
    },
    explicit_keywords=["analyze", "process", "diagnose"],
    arg_extractors={
        "step_one_output": (ArgExtractorType.QUOTED, {}),
        "patient_image": (ArgExtractorType.QUOTED, {}),
    },
)

# Tool 3: End of chain
registry.register(
    name="step_three",
    description="Verifies and validates the analysis output",
    parameters={
        "analysis": "Analysis string from step_two",
        "patient_image": "Path to patient image file",
    },
    required=["analysis", "patient_image"],
    example={
        "analysis": '{"analysis": "..."}',
        "patient_image": "/data/patient_001.jpg",
    },
    docstring="Senior verification pass. Validates findings and enforces schema.",
    tool_type=ToolType.COMPUTATION,
    func=_step_three,
    arg_sources={
        "analysis": "analysis_output",              # ← intermediate key (from Tool 2)
        "patient_image": "patient_image_input",     # ← source key (reused)
    },
    output_keys={
        "verified_output": "Validated analysis JSON",  # ← FINAL key (target)
    },
    explicit_keywords=["verify", "validate", "check"],
    arg_extractors={
        "analysis": (ArgExtractorType.QUOTED, {}),
        "patient_image": (ArgExtractorType.QUOTED, {}),
    },
)
```

### Step 4: Validate the DAG

```python
from nanoathens import DataFlowEngine

engine = DataFlowEngine(registry, verbose=True)
plan = engine.resolve_execution_plan(
    available_keys={"patient_image_input", "image_type_input"},
    target_key="verified_output"
)
print(f"Plan: {plan}")
# Expected: ['step_one', 'step_two', 'step_three']

print(engine.visualize_graph(highlight=plan))
```

### Step 5: Run the agent

```python
from nanoathens import DeclarativeDataFlowAgent, run_medgemma

agent = DeclarativeDataFlowAgent(
    registry=registry,
    reasoning_caller=run_medgemma,
    verbose=True,
    system_prompt="You are a medical AI assistant.",
    goal_few_shot_examples=(
        "EXAMPLES:\n"
        "Query: Analyze this image -> verified_output\n"
        "Query: Find similar cases -> step_one_results\n"
    ),
)

result = await agent.run(
    "Analyze this image. Patient image: '/data/img.jpg' Image type: xray",
    target_key="verified_output"
)
```

---

## 8. Query Formatting (Critical for Extraction)

The user query format directly affects whether the ContextBank can extract source values. Bad formatting = tools can't fill args = pipeline fails.

### Good Query Patterns

```python
# QUOTED extractor — wrap paths/IDs in quotes
"Patient image: '/path/to/image.jpg' Image type: xray"

# ENUM extractor — include the enum value directly
"Analyze this xray of the chest"

# ALPHANUMERIC_ID — include the ID naturally
"Look up patient RAD042 records"

# NUMBER — include value with unit
"Administer 500mg dose"
```

### Bad Query Patterns

```python
# No quotes around path — QUOTED extractor fails
"Patient image: /path/to/image.jpg"  # BAD

# Ambiguous modality — ENUM can't match
"Analyze this scan"  # BAD — "scan" isn't in ["xray", "ct", "mri"]

# ID without preceding keyword — ALPHANUMERIC_ID may miss it
"RAD042 needs analysis"  # RISKY — no preceding "patient" or "id"
```

### Benchmark Query Template

For benchmarking, use a consistent template that guarantees extraction:

```python
query = f"Analyze this chest X-ray. Patient image: '{img_path}' Image type: xray"
```

---

## 9. Common Pitfalls and How to Avoid Them

### Pitfall 1: Output Key Collision
```python
# BAD — "analysis" appears in both arg_sources and output_keys
registry.register(
    arg_sources={"text": "analysis"},
    output_keys={"analysis": "..."},
)
# → ToolSchemaValidationError at registration
```
**Fix**: Use distinct names. Input: `"preliminary_analysis"`, Output: `"verified_analysis"`.

### Pitfall 2: Hallucinated Context Keys
The LLM extractor may hallucinate values for intermediate keys (tool outputs) during Step 1. If it invents a value for `knn_images`, the DAG thinks it's already available and skips the retrieval tool.

**Fix**: The DDA agent drops any extracted key that matches a tool-produced key:
```python
all_tool_produced = set(self.engine._producers.keys())
hallucinated = [k for k in context.parsed_values if k in all_tool_produced]
for k in hallucinated:
    del context.parsed_values[k]
```
This is built into agent.py v0.2.0. Do not remove it.

### Pitfall 3: Function Signature Mismatch
```python
# Function expects "img" but arg_sources says "patient_image"
def my_func(img, modality):  ...

registry.register(
    arg_sources={"patient_image": "...", "image_type": "..."},
    func=my_func,
)
# → Filler passes {"patient_image": val} but function expects "img"
# → TypeError at runtime
```
**Fix**: `arg_sources` keys MUST match function parameter names exactly.

### Pitfall 4: Missing Required Parameters
```python
# "k" is in parameters but not in arg_sources — filler can't find it
parameters={"patient_image": "...", "image_type": "...", "k": "Top-K count"},
required=["patient_image", "image_type", "k"],
arg_sources={"patient_image": "...", "image_type": "..."},
# → Filler fails: missing "k"
```
**Fix**: Either add `"k"` to `arg_sources`, remove it from `required`, or give it a default value in the function signature: `def my_func(patient_image, image_type, k=5)`.

### Pitfall 5: LLM Extraction for Complex Values
Small models (4B params) cannot reliably extract file paths, JSON strings, or complex values from natural language. Don't rely on `ArgExtractorType.LLM` for critical parameters.

**Fix**: Use typed extractors (QUOTED, ENUM, ALPHANUMERIC_ID) for all source keys. Reserve LLM extraction only as a last-resort fallback.

### Pitfall 6: Returning Non-String Results
```python
# Tool returns a dict
def my_tool():
    return {"key": "value"}

# Next tool tries json.loads() on it → crashes
def next_tool(data):
    parsed = json.loads(data)  # TypeError: dict is not a string
```
**Fix**: Always return `json.dumps(result)` from tools. Or use defensive parsing:
```python
data = json.loads(input_data) if isinstance(input_data, str) else input_data
```

### Pitfall 7: Synthesis max_new_tokens Too High
The DDA agent's Step 5 synthesis calls the LLM. If `max_new_tokens` is set too high (e.g., 4096), HuggingFace pipelines emit noisy warnings and waste GPU time.

**Fix**: Keep all internal LLM calls at `max_new_tokens=512`. The extraction prompts need even less — 40-256 tokens.

---

## 10. Checklist Before Registering a Tool

Run through this checklist for every tool:

- [ ] **Function params match arg_sources keys exactly**
- [ ] **All required params have an arg_source OR a default value**
- [ ] **output_keys are unique across the entire registry**
- [ ] **output_keys do NOT appear in any tool's arg_source values** (collision-free)
- [ ] **example dict contains realistic values for every required param**
- [ ] **docstring is descriptive enough for BM25 to index meaningfully**
- [ ] **arg_extractors use typed extractors (ENUM, QUOTED, etc.) for source keys**
- [ ] **Function returns a string** (use json.dumps for structured data)
- [ ] **Consumer tools handle both string and dict inputs defensively**
- [ ] **DAG resolves correctly**: `engine.resolve_execution_plan(source_keys, target)` returns expected plan

---

## 11. Real-World Example: Chest X-Ray Analysis Pipeline

### The 3-Tool Chain

```
User Query: "Analyze this chest X-ray. Patient image: '/data/cxr.jpg' Image type: xray"
                                    ↓
                    ContextBank extracts:
                      patient_image_input = "/data/cxr.jpg"  (QUOTED)
                      image_type_input = "xray"              (ENUM)
                                    ↓
                    DAG resolves plan for "verified_analysis":
                      [retrieve_similar_images, few_shot_image_analysis, verify_few_shot_image_analysis]
                                    ↓
Tool 1: retrieve_similar_images
  IN:  patient_image="/data/cxr.jpg", image_type="xray"
  DO:  MedSigLIP embed → FAISS search → top-5 similar cases
  OUT: knn_images = '{"similar_images": [{"path": "...", "score": 0.89, "description": "Present: Cardiomegaly..."}]}'
                                    ↓
Tool 2: few_shot_image_analysis
  IN:  knn_images=<Tool 1 output>, patient_image="/data/cxr.jpg"
  DO:  Build few-shot prompt from KNN descriptions → MedGemma analyzes image
  OUT: few_shot_analysis = '{"no_finding": "no", "cardiomegaly": "yes", ...}'
                                    ↓
Tool 3: verify_few_shot_image_analysis
  IN:  analysis=<Tool 2 output>, patient_image="/data/cxr.jpg"
  DO:  Senior radiologist verification prompt → Pydantic CXRReview validation
  OUT: verified_analysis = '{"no_finding": "no", "cardiomegaly": "yes", ..., "verification_confidence": "high"}'
```

### The Registrations

```python
radiology_registry = ToolRegistry()

radiology_registry.register(
    name="retrieve_similar_images",
    description="Retrieve similar CXR images from FAISS index using MedSigLIP embeddings",
    parameters={
        "patient_image": "Path to the patient image file",
        "image_type": "Imaging modality: xray|ct|mri",
    },
    required=["patient_image", "image_type"],
    example={
        "patient_image": "/kaggle/input/chexpert/valid/patient001/view1_frontal.jpg",
        "image_type": "xray",
    },
    docstring="Embeds query image with MedSigLIP-448, searches FAISS index, returns top-K similar cases with scores and label descriptions.",
    tool_type=ToolType.RETRIEVAL,
    func=_retrieve_similar_images,
    arg_sources={"patient_image": "patient_image_input", "image_type": "image_type_input"},
    output_keys={"knn_images": "JSON array of similar images with scores and descriptions"},
    explicit_keywords=["similar", "retrieve", "knn", "search", "faiss", "neighbors"],
    arg_extractors={
        "patient_image": (ArgExtractorType.QUOTED, {}),
        "image_type": (ArgExtractorType.ENUM, {"values": ["xray", "ct", "mri"]}),
    },
)

radiology_registry.register(
    name="few_shot_image_analysis",
    description="Analyze CXR using retrieved similar images as few-shot context",
    parameters={
        "knn_images": "JSON string of KNN retrieval results from retrieve_similar_images",
        "patient_image": "Path to the patient image file",
    },
    required=["knn_images", "patient_image"],
    example={
        "knn_images": '{"similar_images": [...]}',
        "patient_image": "/kaggle/input/chexpert/valid/patient001/view1_frontal.jpg",
    },
    docstring="Uses top-K similar cases as few-shot context for MedGemma CXR analysis. Returns structured JSON findings.",
    tool_type=ToolType.COMPUTATION,
    func=_few_shot_image_analysis,
    arg_sources={"knn_images": "knn_images", "patient_image": "patient_image_input"},
    output_keys={"few_shot_analysis": "Preliminary CXR analysis JSON string"},
    explicit_keywords=["analyze", "analysis", "few-shot", "diagnose", "findings"],
    arg_extractors={
        "knn_images": (ArgExtractorType.QUOTED, {}),
        "patient_image": (ArgExtractorType.QUOTED, {}),
    },
)

radiology_registry.register(
    name="verify_few_shot_image_analysis",
    description="Verify and validate the CXR analysis with a second LLM pass and Pydantic schema enforcement",
    parameters={
        "analysis": "Preliminary CXR analysis string from few_shot_image_analysis",
        "patient_image": "Path to the patient image file",
    },
    required=["analysis", "patient_image"],
    example={
        "analysis": '{"no_finding": "no", "cardiomegaly": "yes", ...}',
        "patient_image": "/kaggle/input/chexpert/valid/patient001/view1_frontal.jpg",
    },
    docstring="Senior radiologist verification pass. Re-examines image, corrects errors, enforces CXRReview Pydantic schema for 14 CheXpert labels.",
    tool_type=ToolType.COMPUTATION,
    func=_verify_few_shot_image_analysis,
    arg_sources={"analysis": "few_shot_analysis", "patient_image": "patient_image_input"},
    output_keys={"verified_analysis": "Validated CXR analysis JSON (Pydantic-enforced)"},
    explicit_keywords=["verify", "validate", "check", "review", "confirm"],
    arg_extractors={
        "analysis": (ArgExtractorType.QUOTED, {}),
        "patient_image": (ArgExtractorType.QUOTED, {}),
    },
)
```

### DAG Validation

```python
engine = DataFlowEngine(radiology_registry, verbose=True)
plan = engine.resolve_execution_plan(
    {"patient_image_input", "image_type_input"}, "verified_analysis"
)
assert plan == ["retrieve_similar_images", "few_shot_image_analysis", "verify_few_shot_image_analysis"]
```

---

## 12. ToolType Guide

| Type | Use When | Example |
|------|----------|---------|
| `RETRIEVAL` | Tool fetches external data (DB, index, API) | FAISS search, EHR lookup |
| `COMPUTATION` | Tool processes/transforms data | Image analysis, report generation |
| `KNOWLEDGE` | Tool provides static knowledge | Protocol lookup, reference data |

ToolType is metadata — it doesn't affect execution. It's used by BM25 retrieval and logging.

---

## 13. Testing Without GPU

Use the stub LLM for local testing:

```python
from nanoathens import run_medgemma, DeclarativeDataFlowAgent

# No GPU? run_medgemma falls back to _stub_llm automatically
agent = DeclarativeDataFlowAgent(
    registry=registry,
    reasoning_caller=run_medgemma,
)

# For domain-specific testing, register a custom stub:
from nanoathens import set_stub

def my_stub(messages, max_new_tokens=512, temperature=0.1):
    text = " ".join(str(m.get("content", "")) for m in messages).lower()
    if "json only" in text:
        return '{"patient_image_input": "/test/img.jpg", "image_type_input": "xray"}'
    return "Analysis complete."

set_stub(my_stub)
```

---

## 14. Summary: The Golden Rules

1. **output_keys ∩ arg_source_values = ∅** — collision-free invariant, enforced at registration
2. **arg_sources keys must match function parameter names** — exact string match
3. **Use typed extractors for source keys** — ENUM, QUOTED, ALPHANUMERIC_ID over LLM
4. **Return strings from tools** — `json.dumps()` for structured data
5. **Quote paths in queries** — `Patient image: '/path/to/file.jpg'`
6. **Provide realistic examples** — the `example` field guides LLM argument filling
7. **Validate the DAG before running** — `engine.resolve_execution_plan()` should return the expected plan
8. **Keep max_new_tokens ≤ 512** — for all internal LLM calls
9. **Never trust LLM extraction for tool-produced keys** — the hallucination fix in agent.py handles this
10. **Test the chain end-to-end** — run with `verbose=True` and check all 5 steps complete