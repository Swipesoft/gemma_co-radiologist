# MedGemma Deep Research Agent

> **A multi-agent deep research system built on MedGemma-1.5-4B** — capable of autonomously producing 35–80 page, citation-rich research papers on any medical topic, running entirely on a small language model.

---

## Overview

This repository contains the source code and generated artifacts for the **MedGemma Deep Research Agent**, an experimental system built to stress-test the generalisability of **ContextBank** — the context management module powering the Declarative DataFlow Architecture (DDFA).

The core research question: *Can a 4B-parameter small language model, given a well-designed orchestration architecture and persistent memory, produce coherent long-horizon research documents that rival those of much larger models?*

The results suggest: **yes, with the right architecture it can.**

---

## Architecture

The system follows a structured **orchestrator–subagent workflow**. A single orchestrator (MedGemma-1.5-4B) delegates to five specialised subagents, each also running on MedGemma-1.5-4B.

```
Orchestrator (DeepResearchOrchestratorV4)
├── PlannerAgent      → Creates section outlines with diverse structural frames
├── SearchAgent       → Performs Tavily web searches per section
├── WriterAgent       → Drafts content section by section at PhD level
├── VerifierAgent     → Fact-checks claims and flags uncertain content
└── MemoryAgent       → Persists all artifacts to disk for fault tolerance
```

### Key Architectural Innovations

| Component | Challenge Addressed | Solution |
|-----------|---------------------|----------|
| **ContextBank** | Small models lose context across long tasks | Dynamic context bank with collision-free tool descriptions |
| **Adaptive Temperature** | Greedy decoding produces repetitive output | Task-specific presets: writing (0.8), planning (0.3), factual (0.1) |
| **Persistent Memory** | Long research tasks exceed context windows | File-based state that decouples execution from model context limits |
| **FactFilter** | Strict verification kills recall on medical queries | Confidence thresholding with `allow_unverified=True` and caveats |
| **OutputCleaner** | Model leaks prompt artifacts into output | 30+ regex patterns stripping meta-text and template leakage |
| **OverlapChecker** | Risk of uncredited source borrowing | Jaccard + semantic similarity flagging across subsections |
| **ContextualImageGen** | Images disconnected from surrounding text | Full section content analysis before Gemini image prompting |

---

## Persistent Memory & Fault Tolerance

Each research run produces a fully checkpointed project saved to local filesystem under `/content/research_projects/{project_id}/`:

```
{project_id}/
├── state.json              # Full project state (resumable)
├── sections/               # Per-section markdown files
│   ├── sec_01_intro.md
│   ├── sec_02_methods.md
│   └── ...
├── images/                 # AI-generated section illustrations
├── citations.json          # All references in Vancouver format
├── FINAL_REPORT.md         # Assembled full paper
└── RESEARCH_PAPER.pdf      # Final formatted PDF
```

If a run crashes mid-way (API error, rate limit, timeout), the system resumes **exactly where it left off** — no lost progress.

---

## Results

Running entirely on **MedGemma-1.5-4B**, the system produces:

- Research documents between **35–80 pages**
- **Inline Vancouver-style citations** with reasonable accuracy
- **AI-generated section illustrations** (flowcharts, infographics, anatomical diagrams) via Gemini image API
- **Structured abstracts, table of contents, and reference lists**
- Graceful **failure recovery** and support for **human-in-the-loop corrections**

---

## Repository Structure

```
.
├── GemmaDeepResearchAgent.ipynb     # Full source: all agents, tools, orchestrator
├── artifacts/                       # Generated research paper outputs
│   ├── mechanical_ventilation/
│   ├── llm_in_healthcare/
│   ├── digital_health/
│   └── regenerative_medicine/
└── README.md
```

---

## Benchmark Topics

The following topics were used for benchmarking:

| # | Topic |
|---|-------|
| 0 | Recent Advances in Artificial Intelligence for Medical Diagnosis |
| 1 | Mechanical Ventilation in the ICU: Frontier Innovations 2010–2026 |
| 2 | CRISPR Gene Editing in Clinical Medicine |
| 3 | The Role of Large Language Models in Healthcare |
| 4 | Immunotherapy in Oncology: Breakthrough Treatments and Resistance Mechanisms |
| 5 | Digital Health and Remote Patient Monitoring |
| 6 | Antibiotic Resistance: Global Trends and Novel Therapeutic Strategies |
| 7 | Precision Medicine in Cardiovascular Disease |
| 8 | Mental Health in the Post-Pandemic Era |
| 9 | Regenerative Medicine and Stem Cell Therapy |

---

## Quickstart

### Prerequisites

```bash
pip install transformers torch tavily-python weasyprint markdown json-repair sentence-transformers rank-bm25
```

You will need:
- **Hugging Face** access to [`google/medgemma-4b-it`](https://huggingface.co/google/medgemma-4b-it)
- A **Tavily API key** (web search) — [tavily.com](https://tavily.com)
- A **Gemini API key** (image generation, optional) — [Google AI Studio](https://aistudio.google.com)

### Running on Google Colab (A100 recommended)

Set your secrets in Colab:
```
TAVILY_API_KEY
GEMINI_API_KEY
HF_TOKEN
```

Open `GemmaDeepResearchAgent.ipynb` and run all cells in order. Then to run the full benchmark:

```python
# All 10 topics
results = await run_benchmark()

# Specific topics only
results = await run_benchmark_subset([1, 3, 5, 9])

# Single topic
result = await run_deep_research(topic_index=0, target_pages=20)
```

Results are automatically exported as PDFs to Google Drive under `DeepResearch_Benchmark/`.

---

## Human Review

After a research run completes, the `HumanReviewSystem` allows post-hoc editing:

```python
review.list_projects()                          # See all completed runs
review.load_project("project_id")               # Load one for review
review.show_pending_reviews()                   # See auto-flagged items
review.add_correction("sec_03", "Fix stat X")  # Annotate a section
review.request_regeneration("sec_07", "Outdated data — rewrite with 2025 sources")
await orchestrator.resume("project_id")         # Re-run flagged sections only
```

---

## Limitations

- **Model size**: MedGemma-1.5-4B produces strong output for a 4B model but will occasionally hallucinate or produce shallow analysis on highly specialised subtopics. The VerifierAgent and human review system exist to catch and correct this.
- **Context window**: Each agent turn is capped at 8,192 tokens. ContextBank manages what information is in scope, but very large section bodies may be truncated.
- **Image generation**: Requires a Gemini API key and can be disabled with `allow_image_generation=False`.
- **Sequential GPU inference**: LLM inference is serialised even in concurrent benchmark mode. Tavily web searches genuinely overlap; LLM phases do not.

---

## Citation

If you use this work, please cite:

```bibtex
@Swipesoft{medgemma_deepresearch_2025,
  title   = {MedGemma Deep Research Agent: Long-Horizon Research Generation with Small Language Models},
  year    = {2025},
  note    = {Experimental system built on MedGemma-1.5-4B with Declarative DataFlow Dynamic Context Architecture},
  url     = {https://github.com/Swipesoft/gemma_co-radiologist}
}
```

---

## Acknowledgements

- [MedGemma](https://huggingface.co/google/medgemma-4b-it) by Google DeepMind
- [Tavily](https://tavily.com) for research-grade web search
- [WeasyPrint](https://weasyprint.org) for PDF generation
- [Sentence Transformers](https://www.sbert.net) for semantic similarity in overlap detection