"""
╔══════════════════════════════════════════════════════════════╗
║              Gemma Co-Radiologist — Streamlit GUI             ║
║                                                              ║
║  A medical-grade radiology workstation interface built on    ║
║  the Declarative DataFlow Agent architecture.                ║
║                                                              ║
║  Usage:                                                      ║
║    streamlit run gemma_co_radiologist_app.py                 ║
╚══════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import asyncio
import json
import time
import os
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

# ── Import SDK from athena_dda package (pip install athena_dda) ────────────────
try:
    from athena_dda import (
        ToolRegistry, ToolSchema, ToolType, ContextBank,
        LLMValueExtractor, GroundedArgumentFiller, DataFlowEngine,
        GoalKeyResolver, DeclarativeDataFlowAgent,
        SessionStore, SESSION_STORE,
        run_medgemma,
    )
    print("✓ SDK loaded from athena_dda package")
except ImportError:
    # Fallback: import from monolithic file
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from co_radiologist_agent import (
        ToolRegistry, ToolSchema, ToolType, ContextBank,
        LLMValueExtractor, GroundedArgumentFiller, DataFlowEngine,
        GoalKeyResolver, DeclarativeDataFlowAgent,
        SessionStore, SESSION_STORE,
        run_medgemma,
    )
    print("✓ SDK loaded from co_radiologist_agent (fallback)")

# ── Import agent-specific modules (local) ─────────────────────────────────────
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from co_radiologist_agent import (
    radiology_agent_registry,
    RadiologySessionAgent,
    RADIOLOGY_SYSTEM_PROMPT, RADIOLOGY_GOAL_EXAMPLES,
    extract_and_validate_json,
    RADIOLOGY_EHR_DB,
    CXRReview, CTHeadReview, CTAbdominalClassification,
    get_prompt_skill,
)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG & CUSTOM CSS
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Gemma Co-Radiologist",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Dark Medical Workstation Theme ────────────────────────────────────────────
st.markdown("""
<style>
/* ── Import fonts ── */
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,400&family=JetBrains+Mono:wght@300;400;500&family=Instrument+Serif:ital@0;1&display=swap');

/* ── Root variables ── */
:root {
    --bg-primary: #0a0e17;
    --bg-secondary: #111827;
    --bg-card: #151d2e;
    --bg-card-hover: #1a2540;
    --bg-elevated: #1e293b;
    --border-subtle: #1e293b;
    --border-active: #3b82f6;
    --text-primary: #e2e8f0;
    --text-secondary: #94a3b8;
    --text-muted: #64748b;
    --accent-blue: #3b82f6;
    --accent-cyan: #06b6d4;
    --accent-emerald: #10b981;
    --accent-amber: #f59e0b;
    --accent-rose: #f43f5e;
    --accent-violet: #8b5cf6;
    --gradient-hero: linear-gradient(135deg, #0f172a 0%, #0c1425 40%, #111827 100%);
    --shadow-card: 0 1px 3px rgba(0,0,0,.4), 0 1px 2px rgba(0,0,0,.3);
    --shadow-elevated: 0 10px 40px rgba(0,0,0,.5);
    --radius-sm: 6px;
    --radius-md: 10px;
    --radius-lg: 16px;
    --font-sans: 'DM Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    --font-mono: 'JetBrains Mono', 'SF Mono', monospace;
    --font-display: 'Instrument Serif', Georgia, serif;
}

/* ── Global overrides ── */
.stApp {
    background: var(--gradient-hero) !important;
    color: var(--text-primary) !important;
    font-family: var(--font-sans) !important;
}

/* Make sidebar dark */
section[data-testid="stSidebar"] {
    background: var(--bg-secondary) !important;
    border-right: 1px solid var(--border-subtle) !important;
}
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] .stMarkdown li,
section[data-testid="stSidebar"] label {
    color: var(--text-secondary) !important;
    font-family: var(--font-sans) !important;
}

/* ── Typography ── */
h1, h2, h3, h4 {
    font-family: var(--font-sans) !important;
    color: var(--text-primary) !important;
    letter-spacing: -0.02em;
}
p, li, span, div {
    color: var(--text-secondary);
}

/* ── Header bar ── */
.app-header {
    background: linear-gradient(90deg, var(--bg-secondary), var(--bg-card));
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-lg);
    padding: 1.1rem 1.8rem;
    margin-bottom: 1.2rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    box-shadow: var(--shadow-card);
}
.app-header .title-group {
    display: flex;
    align-items: center;
    gap: 14px;
}
.app-header .logo-mark {
    width: 42px; height: 42px;
    background: linear-gradient(135deg, var(--accent-cyan), var(--accent-blue));
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 22px;
    box-shadow: 0 0 20px rgba(6,182,212,.25);
}
.app-header .app-title {
    font-family: var(--font-sans) !important;
    font-size: 1.35rem;
    font-weight: 700;
    color: var(--text-primary) !important;
    letter-spacing: -0.03em;
    margin: 0 !important; padding: 0 !important;
    line-height: 1.2;
}
.app-header .app-subtitle {
    font-size: 0.78rem;
    color: var(--text-muted);
    font-weight: 400;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    margin-top: 2px;
}
.app-header .session-badge {
    font-family: var(--font-mono);
    font-size: 0.72rem;
    color: var(--accent-cyan);
    background: rgba(6,182,212,.1);
    border: 1px solid rgba(6,182,212,.2);
    padding: 4px 12px;
    border-radius: 20px;
    letter-spacing: 0.02em;
}

/* ── Panel cards ── */
.panel-card {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-md);
    padding: 1.1rem 1.3rem;
    margin-bottom: 0.9rem;
    box-shadow: var(--shadow-card);
    transition: border-color 0.2s ease;
}
.panel-card:hover {
    border-color: rgba(59,130,246,.25);
}
.panel-card .panel-title {
    font-family: var(--font-sans);
    font-size: 0.7rem;
    font-weight: 600;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.7rem;
    display: flex; align-items: center; gap: 6px;
}
.panel-card .panel-title .dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    display: inline-block;
}

/* ── Pipeline tracker ── */
.pipeline-step {
    display: flex;
    align-items: flex-start;
    gap: 12px;
    padding: 8px 0;
    border-bottom: 1px solid rgba(30,41,59,.6);
    animation: fadeSlideIn 0.35s ease-out both;
}
.pipeline-step:last-child { border-bottom: none; }
.pipeline-step .step-indicator {
    width: 28px; height: 28px; min-width: 28px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 12px;
    font-weight: 600;
    margin-top: 1px;
    transition: all 0.3s ease;
}
.step-pending .step-indicator {
    background: rgba(100,116,139,.15);
    border: 1.5px dashed var(--text-muted);
    color: var(--text-muted);
}
.step-running .step-indicator {
    background: rgba(59,130,246,.15);
    border: 2px solid var(--accent-blue);
    color: var(--accent-blue);
    box-shadow: 0 0 12px rgba(59,130,246,.3);
    animation: pulseGlow 1.5s infinite;
}
.step-done .step-indicator {
    background: rgba(16,185,129,.15);
    border: 2px solid var(--accent-emerald);
    color: var(--accent-emerald);
}
.step-error .step-indicator {
    background: rgba(244,63,94,.15);
    border: 2px solid var(--accent-rose);
    color: var(--accent-rose);
}
.step-skipped .step-indicator {
    background: rgba(245,158,11,.1);
    border: 1.5px solid var(--accent-amber);
    color: var(--accent-amber);
}
.pipeline-step .step-info { flex: 1; }
.pipeline-step .step-name {
    font-family: var(--font-mono);
    font-size: 0.82rem;
    font-weight: 500;
    color: var(--text-primary);
    line-height: 1.3;
}
.pipeline-step .step-detail {
    font-size: 0.72rem;
    color: var(--text-muted);
    margin-top: 2px;
}
.pipeline-step .step-time {
    font-family: var(--font-mono);
    font-size: 0.68rem;
    color: var(--text-muted);
    min-width: 48px;
    text-align: right;
    margin-top: 3px;
}

/* ── Finding badges ── */
.finding-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 7px 0;
    border-bottom: 1px solid rgba(30,41,59,.4);
}
.finding-row:last-child { border-bottom: none; }
.finding-label {
    font-size: 0.85rem;
    color: var(--text-secondary);
    font-weight: 400;
}
.finding-badge {
    font-family: var(--font-mono);
    font-size: 0.72rem;
    font-weight: 500;
    padding: 3px 10px;
    border-radius: 12px;
    letter-spacing: 0.02em;
}
.badge-yes {
    background: rgba(244,63,94,.12);
    color: #fb7185;
    border: 1px solid rgba(244,63,94,.25);
}
.badge-no {
    background: rgba(16,185,129,.08);
    color: #6ee7b7;
    border: 1px solid rgba(16,185,129,.2);
}
.badge-unclear {
    background: rgba(245,158,11,.1);
    color: #fbbf24;
    border: 1px solid rgba(245,158,11,.2);
}

/* ── Metric tiles ── */
.metric-tile {
    background: var(--bg-elevated);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-sm);
    padding: 12px 14px;
    text-align: center;
}
.metric-tile .metric-value {
    font-family: var(--font-mono);
    font-size: 1.3rem;
    font-weight: 600;
    color: var(--text-primary);
    line-height: 1.2;
}
.metric-tile .metric-label {
    font-size: 0.65rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-top: 4px;
}

/* ── Status pill ── */
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-family: var(--font-mono);
    font-size: 0.72rem;
    font-weight: 500;
    padding: 4px 12px;
    border-radius: 20px;
}
.status-success {
    background: rgba(16,185,129,.1);
    color: var(--accent-emerald);
    border: 1px solid rgba(16,185,129,.25);
}
.status-running {
    background: rgba(59,130,246,.1);
    color: var(--accent-blue);
    border: 1px solid rgba(59,130,246,.25);
}
.status-error {
    background: rgba(244,63,94,.1);
    color: var(--accent-rose);
    border: 1px solid rgba(244,63,94,.25);
}
.status-idle {
    background: rgba(100,116,139,.1);
    color: var(--text-muted);
    border: 1px solid rgba(100,116,139,.2);
}

/* ── Animations ── */
@keyframes pulseGlow {
    0%, 100% { box-shadow: 0 0 8px rgba(59,130,246,.2); }
    50% { box-shadow: 0 0 18px rgba(59,130,246,.45); }
}
@keyframes fadeSlideIn {
    from { opacity: 0; transform: translateY(6px); }
    to { opacity: 1; transform: translateY(0); }
}
@keyframes shimmer {
    0% { background-position: -200% 0; }
    100% { background-position: 200% 0; }
}
.shimmer-text {
    background: linear-gradient(90deg, var(--text-muted) 25%, var(--accent-cyan) 50%, var(--text-muted) 75%);
    background-size: 200% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: shimmer 2s linear infinite;
}

/* ── Report block ── */
.report-block {
    background: var(--bg-secondary);
    border: 1px solid var(--border-subtle);
    border-left: 3px solid var(--accent-blue);
    border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
    padding: 1rem 1.2rem;
    font-family: var(--font-sans);
    font-size: 0.85rem;
    color: var(--text-secondary);
    line-height: 1.65;
    max-height: 420px;
    overflow-y: auto;
    white-space: pre-wrap;
}

/* ── Image viewer ── */
.image-viewer {
    background: #000;
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-md);
    padding: 4px;
    display: flex;
    align-items: center;
    justify-content: center;
    min-height: 280px;
    position: relative;
    overflow: hidden;
}
.image-viewer img {
    max-width: 100%;
    max-height: 380px;
    object-fit: contain;
    border-radius: 6px;
}
.image-placeholder {
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    gap: 10px;
    min-height: 240px;
    color: var(--text-muted);
    font-size: 0.85rem;
}
.image-placeholder .icon {
    font-size: 2.8rem;
    opacity: 0.3;
}

/* ── Streamlit overrides ── */
.stButton > button {
    background: linear-gradient(135deg, var(--accent-blue), var(--accent-cyan)) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: var(--font-sans) !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    padding: 0.55rem 1.2rem !important;
    letter-spacing: 0.01em;
    transition: all 0.2s ease !important;
    box-shadow: 0 2px 8px rgba(59,130,246,.3) !important;
    width: 100%;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 16px rgba(59,130,246,.4) !important;
}

/* Secondary buttons */
.secondary-btn > button {
    background: var(--bg-elevated) !important;
    border: 1px solid var(--border-subtle) !important;
    box-shadow: none !important;
    color: var(--text-secondary) !important;
}
.secondary-btn > button:hover {
    background: var(--bg-card-hover) !important;
    border-color: rgba(59,130,246,.3) !important;
}

.stSelectbox label, .stTextInput label, .stTextArea label, .stFileUploader label {
    color: var(--text-secondary) !important;
    font-family: var(--font-sans) !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
}
.stSelectbox > div > div,
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    background: var(--bg-elevated) !important;
    border: 1px solid var(--border-subtle) !important;
    color: var(--text-primary) !important;
    font-family: var(--font-sans) !important;
    border-radius: var(--radius-sm) !important;
}

div[data-testid="stFileUploader"] {
    background: var(--bg-elevated) !important;
    border: 1px dashed var(--border-subtle) !important;
    border-radius: var(--radius-md) !important;
    padding: 0.8rem !important;
}

/* Tab overrides */
.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    background: var(--bg-card);
    border-radius: var(--radius-sm);
    padding: 3px;
    border: 1px solid var(--border-subtle);
}
.stTabs [data-baseweb="tab"] {
    font-family: var(--font-sans) !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    color: var(--text-muted) !important;
    border-radius: 5px !important;
    padding: 6px 16px !important;
}
.stTabs [aria-selected="true"] {
    background: var(--accent-blue) !important;
    color: white !important;
}
.stTabs [data-baseweb="tab-highlight"] { display: none; }
.stTabs [data-baseweb="tab-border"] { display: none; }

/* Expander */
.streamlit-expanderHeader {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius-sm) !important;
    font-family: var(--font-sans) !important;
    color: var(--text-secondary) !important;
}

/* Hide default streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Divider */
.subtle-divider {
    border: none;
    border-top: 1px solid var(--border-subtle);
    margin: 0.8rem 0;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  SESSION STATE INITIALIZATION
# ══════════════════════════════════════════════════════════════════════════════

def init_session_state():
    """Initialize all session state variables."""
    defaults = {
        "session_id": None,
        "agent": None,
        "run_history": [],
        "pipeline_steps": [],
        "current_status": "idle",
        "uploaded_image": None,
        "uploaded_image_name": None,
        "analysis_result": None,
        "findings": None,
        "soap_report": None,
        "pdf_path": None,
        "localization_data": None,
        "qa_history": [],
        "active_tab": "analyze",
        "execution_log": [],
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

    # Create agent once per session
    if st.session_state.agent is None:
        st.session_state.agent = RadiologySessionAgent(
            reasoning_caller=run_medgemma,
            verbose=False,
        )


init_session_state()


# ══════════════════════════════════════════════════════════════════════════════
#  PIPELINE EXECUTION WITH REAL-TIME STATUS
# ══════════════════════════════════════════════════════════════════════════════

async def execute_with_tracking(
    query: str,
    image_path: str = None,
    image_type: str = None,
    patient_id: str = None,
    target_key: str = None,
    pipeline_container=None,
    status_container=None,
):
    """Execute the DDA agent with real-time pipeline step tracking.

    Instead of using the simple agent.run(), we replicate the DDA flow
    with explicit step-by-step UI updates.
    """
    agent = st.session_state.agent
    registry = radiology_agent_registry
    reasoning_llm = run_medgemma

    t0 = time.time()
    steps = []
    st.session_state.current_status = "running"

    def update_pipeline_ui():
        """Rerender the pipeline tracker."""
        if pipeline_container is None:
            return
        html_parts = []
        for s in steps:
            css_class = f"step-{s['status']}"
            icon_map = {
                "pending": "○",
                "running": "◉",
                "done": "✓",
                "error": "✗",
                "skipped": "−",
            }
            icon = icon_map.get(s["status"], "○")
            time_str = f"{s.get('elapsed', 0):.1f}s" if s.get("elapsed") else ""
            detail = s.get("detail", "")

            html_parts.append(f"""
            <div class="pipeline-step {css_class}">
                <div class="step-indicator">{icon}</div>
                <div class="step-info">
                    <div class="step-name">{s['name']}</div>
                    <div class="step-detail">{detail}</div>
                </div>
                <div class="step-time">{time_str}</div>
            </div>""")

        pipeline_container.markdown(
            '<div class="panel-card">'
            '<div class="panel-title"><span class="dot" style="background:var(--accent-blue)"></span>EXECUTION PIPELINE</div>'
            + "".join(html_parts)
            + "</div>",
            unsafe_allow_html=True,
        )

    def set_status(text):
        if status_container:
            status_container.markdown(
                f'<span class="status-pill status-running">◉ {text}</span>',
                unsafe_allow_html=True,
            )

    # ─── Step 1: Variable Extraction ──────────────────────────────────────
    steps.append({"name": "Variable Extraction", "status": "running",
                  "detail": "Parsing query for context keys..."})
    update_pipeline_ui()
    set_status("Extracting variables from query...")

    extractor = LLMValueExtractor(reasoning_llm, verbose=False)
    known_keys = registry.get_all_context_keys()
    context = ContextBank(extractor, known_keys, registry)

    # Build augmented query with metadata
    parts = [query]
    if image_path:
        parts.append(f"Patient image: {image_path}")
        context.parsed_values["patient_image_input"] = image_path
    if image_type:
        parts.append(f"Image type: {image_type}")
        context.parsed_values["image_type_input"] = image_type
    if patient_id:
        parts.append(f"Patient ID: {patient_id}")
        context.parsed_values["patient_id_input"] = patient_id.upper()
    augmented = " | ".join(parts)

    t_step = time.time()
    context.set_goal(augmented)

    populated = {k for k, v in context.parsed_values.items()
                 if v is not None and v != "" and v != "None"}
    steps[-1]["status"] = "done"
    steps[-1]["elapsed"] = time.time() - t_step
    steps[-1]["detail"] = f"{len(populated)} values extracted"
    update_pipeline_ui()

    # ─── Step 2: Goal Resolution ──────────────────────────────────────────
    steps.append({"name": "Goal Resolution", "status": "running",
                  "detail": "Identifying target output key..."})
    update_pipeline_ui()
    set_status("Resolving execution target...")

    t_step = time.time()
    engine = DataFlowEngine(registry, verbose=False)
    goal_resolver = GoalKeyResolver(
        reasoning_llm, engine, verbose=False,
        few_shot_examples=RADIOLOGY_GOAL_EXAMPLES,
    )

    resolved_key = target_key or goal_resolver.resolve(query)
    if resolved_key is None:
        steps[-1]["status"] = "error"
        steps[-1]["detail"] = "Could not identify target"
        steps[-1]["elapsed"] = time.time() - t_step
        update_pipeline_ui()
        st.session_state.current_status = "error"
        return {"status": "goal_resolution_failed", "response": "Could not resolve goal."}

    steps[-1]["status"] = "done"
    steps[-1]["elapsed"] = time.time() - t_step
    steps[-1]["detail"] = f"Target → {resolved_key}"
    update_pipeline_ui()

    # ─── Step 3: DAG Resolution ───────────────────────────────────────────
    steps.append({"name": "Graph Resolution", "status": "running",
                  "detail": "Computing execution path in DAG..."})
    update_pipeline_ui()
    set_status("Resolving dataflow graph...")

    t_step = time.time()
    all_produced = set(engine._producers.keys())
    source_keys = {k for k in known_keys if k not in all_produced}
    available = populated | source_keys
    available.discard(resolved_key)

    plan = engine.resolve_execution_plan(available, resolved_key)

    if plan is None:
        steps[-1]["status"] = "error"
        steps[-1]["detail"] = f"No path to '{resolved_key}'"
        steps[-1]["elapsed"] = time.time() - t_step
        update_pipeline_ui()
        st.session_state.current_status = "error"
        return {"status": "null_plan",
                "response": f"No execution path found for '{resolved_key}'."}

    steps[-1]["status"] = "done"
    steps[-1]["elapsed"] = time.time() - t_step
    steps[-1]["detail"] = f"{len(plan)} tools in chain"
    update_pipeline_ui()

    # ─── Step 4: Add pending tools ────────────────────────────────────────
    for tool_name in plan:
        tool = registry.get_tool(tool_name)
        steps.append({
            "name": tool_name.replace("_", " ").title(),
            "tool_name": tool_name,
            "status": "pending",
            "detail": tool.description[:60] + "..." if len(tool.description) > 60 else tool.description,
        })
    update_pipeline_ui()

    # ─── Step 5: Execute each tool ────────────────────────────────────────
    filler = GroundedArgumentFiller(reasoning_llm, verbose=False)
    tools_executed = []
    failed = []

    for i, tool_name in enumerate(plan):
        step_idx = 3 + i  # offset for extraction + resolution + graph steps
        steps[step_idx]["status"] = "running"
        steps[step_idx]["detail"] = "Filling arguments..."
        update_pipeline_ui()
        set_status(f"Running {tool_name}...")

        tool = registry.get_tool(tool_name)
        t_step = time.time()

        args = filler.fill_arguments(tool, context)
        if args is None:
            steps[step_idx]["status"] = "skipped"
            steps[step_idx]["detail"] = "Could not fill required args"
            steps[step_idx]["elapsed"] = time.time() - t_step
            failed.append(tool_name)
            update_pipeline_ui()
            continue

        steps[step_idx]["detail"] = "Executing..."
        update_pipeline_ui()

        try:
            result = await registry.call(tool_name, args)
            context.add_tool_result(tool_name, str(result), tool.output_keys)
            tools_executed.append(tool_name)

            elapsed = time.time() - t_step
            steps[step_idx]["status"] = "done"
            steps[step_idx]["elapsed"] = elapsed

            # Show brief result
            result_str = str(result)
            if len(result_str) > 80:
                result_str = result_str[:77] + "..."
            steps[step_idx]["detail"] = f"✓ {result_str}"
            update_pipeline_ui()

        except Exception as e:
            steps[step_idx]["status"] = "error"
            steps[step_idx]["detail"] = str(e)[:80]
            steps[step_idx]["elapsed"] = time.time() - t_step
            failed.append(tool_name)
            update_pipeline_ui()

    # ─── Step 6: Synthesis ────────────────────────────────────────────────
    steps.append({"name": "Synthesis", "status": "running",
                  "detail": "Generating clinical summary..."})
    update_pipeline_ui()
    set_status("Synthesizing response...")

    t_step = time.time()
    results_text = "\n".join(
        f"- {t}: {r[:250]}" for t, r in context.retrieved_data.items()
    )
    synth_prompt = (
        f"{RADIOLOGY_SYSTEM_PROMPT}\n\n"
        f"QUESTION: {query}\n\n"
        f"CLINICAL DATA:\n{results_text or 'No data collected.'}\n\nRESPONSE:"
    )
    try:
        response = reasoning_llm(
            messages=[{"role": "user", "content": [{"type": "text", "text": synth_prompt}]}],
            max_new_tokens=512, temperature=0.2,
        )
    except Exception:
        response = f"Collected data:\n{results_text}"

    steps[-1]["status"] = "done"
    steps[-1]["elapsed"] = time.time() - t_step
    steps[-1]["detail"] = "Complete"
    update_pipeline_ui()

    # ─── Finalize ─────────────────────────────────────────────────────────
    total_elapsed = round(time.time() - t0, 2)
    st.session_state.current_status = "success" if not failed else "partial"
    st.session_state.pipeline_steps = steps

    if status_container:
        if failed:
            status_container.markdown(
                f'<span class="status-pill status-error">● Partial — {total_elapsed}s</span>',
                unsafe_allow_html=True,
            )
        else:
            status_container.markdown(
                f'<span class="status-pill status-success">● Complete — {total_elapsed}s</span>',
                unsafe_allow_html=True,
            )

    # Save to session
    result_dict = {
        "response": response.strip() if isinstance(response, str) else str(response),
        "tools_executed": tools_executed,
        "failed_tools": failed,
        "execution_plan": plan,
        "target_key": resolved_key,
        "latency_s": total_elapsed,
        "status": "success" if not failed else "partial",
        "context_data": dict(context.parsed_values),
        "raw_results": dict(context.retrieved_data),
    }

    SESSION_STORE.save_run_result(
        st.session_state.agent.session_id or "default",
        query, result_dict,
    )

    return result_dict


# ══════════════════════════════════════════════════════════════════════════════
#  HELPER: PARSE AND DISPLAY FINDINGS
# ══════════════════════════════════════════════════════════════════════════════

def render_findings_card(findings_json: str) -> str:
    """Convert verified analysis JSON into a styled findings HTML card."""
    try:
        data = json.loads(findings_json) if isinstance(findings_json, str) else findings_json
    except (json.JSONDecodeError, TypeError):
        data = {}

    skip = {"_meta", "verification_confidence", "verification_notes",
            "other_abnormal_features", "other_findings"}
    rows = []
    for key, value in data.items():
        if key.startswith("_") or key in skip:
            continue
        label = key.replace("_", " ").title()
        val_str = str(value)
        val_lower = val_str.lower()

        if val_lower == "yes":
            badge_class = "badge-yes"
        elif val_lower == "no":
            badge_class = "badge-no"
        elif val_lower in ("unclear", "uncertain"):
            badge_class = "badge-unclear"
        else:
            badge_class = "badge-no"
            val_str = val_str[:30]

        rows.append(f"""
        <div class="finding-row">
            <span class="finding-label">{label}</span>
            <span class="finding-badge {badge_class}">{val_str}</span>
        </div>""")

    # Confidence and notes
    confidence = data.get("verification_confidence", "—")
    notes = data.get("verification_notes", "")
    other = data.get("other_abnormal_features") or data.get("other_findings") or ""

    footer = ""
    if confidence != "—":
        badge_c = "badge-no" if confidence.lower() == "high" else ("badge-unclear" if confidence.lower() == "medium" else "badge-yes")
        footer += f'<div class="finding-row"><span class="finding-label">Verification Confidence</span><span class="finding-badge {badge_c}">{confidence}</span></div>'
    if other:
        footer += f'<div style="margin-top:8px;font-size:0.8rem;color:var(--text-muted)"><strong>Other:</strong> {other[:200]}</div>'

    return "".join(rows) + footer


# ══════════════════════════════════════════════════════════════════════════════
#  RENDER: HEADER
# ══════════════════════════════════════════════════════════════════════════════

def render_header():
    session_id = st.session_state.agent.session_id or "—"
    status_map = {
        "idle": ("● Idle", "status-idle"),
        "running": ("◉ Running", "status-running"),
        "success": ("● Complete", "status-success"),
        "partial": ("● Partial", "status-error"),
        "error": ("● Error", "status-error"),
    }
    status_text, status_class = status_map.get(
        st.session_state.current_status, ("● Unknown", "status-idle")
    )

    st.markdown(f"""
    <div class="app-header">
        <div class="title-group">
            <div class="logo-mark">🫁</div>
            <div>
                <div class="app-title">Gemma Co-Radiologist</div>
                <div class="app-subtitle">Declarative DataFlow Agent · MedGemma</div>
            </div>
        </div>
        <div style="display:flex;align-items:center;gap:12px;">
            <span class="status-pill {status_class}">{status_text}</span>
            <span class="session-badge">{session_id}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  RENDER: SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="padding:0.5rem 0 1rem 0;">
            <div style="font-size:0.7rem;font-weight:600;color:var(--text-muted);
                 text-transform:uppercase;letter-spacing:0.08em;margin-bottom:8px;">
                Patient Configuration
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Image upload
        uploaded = st.file_uploader(
            "Upload Medical Image",
            type=["png", "jpg", "jpeg", "dcm", "dicom"],
            help="Supported: PNG, JPG, DICOM",
        )
        if uploaded:
            st.session_state.uploaded_image = uploaded
            st.session_state.uploaded_image_name = uploaded.name

        st.markdown('<hr class="subtle-divider">', unsafe_allow_html=True)

        # Modality
        image_type = st.selectbox(
            "Imaging Modality",
            ["xray", "ct", "mri"],
            format_func=lambda x: {"xray": "X-Ray", "ct": "CT Scan", "mri": "MRI"}[x],
        )

        # Body part
        body_part = st.selectbox(
            "Body Region",
            ["chest", "head", "abdomen", "brain", "other"],
            format_func=lambda x: x.title(),
        )

        st.markdown('<hr class="subtle-divider">', unsafe_allow_html=True)

        # Patient ID
        patient_id = st.selectbox(
            "Patient ID",
            ["—", "RAD001", "RAD002", "RAD003"],
            help="Select a patient for EHR integration",
        )
        if patient_id != "—":
            pt = RADIOLOGY_EHR_DB.get(patient_id, {})
            if pt:
                st.markdown(f"""
                <div style="background:var(--bg-elevated);border:1px solid var(--border-subtle);
                     border-radius:6px;padding:10px 12px;margin-top:4px;">
                    <div style="font-size:0.78rem;font-weight:600;color:var(--text-primary);">{pt.get('name','')}</div>
                    <div style="font-size:0.72rem;color:var(--text-muted);margin-top:2px;">
                        {pt.get('age','')}y · {pt.get('sex','')} · {len(pt.get('prior_imaging',[]))} prior studies
                    </div>
                    <div style="font-size:0.7rem;color:var(--text-muted);margin-top:4px;">
                        {pt.get('complaints','')[:80]}
                    </div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown('<hr class="subtle-divider">', unsafe_allow_html=True)

        # Target key override (advanced)
        with st.expander("⚙ Advanced — Target Key"):
            all_keys = DataFlowEngine(radiology_agent_registry, verbose=False).get_all_output_keys()
            target_override = st.selectbox(
                "Override target",
                ["auto"] + all_keys,
            )

        st.markdown('<hr class="subtle-divider">', unsafe_allow_html=True)

        # Session controls
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 New Session", use_container_width=True):
                st.session_state.agent.reset_session()
                st.session_state.analysis_result = None
                st.session_state.findings = None
                st.session_state.soap_report = None
                st.session_state.pdf_path = None
                st.session_state.localization_data = None
                st.session_state.pipeline_steps = []
                st.session_state.current_status = "idle"
                st.session_state.qa_history = []
                st.session_state.execution_log = []
                st.rerun()
        with col2:
            st.markdown('<div class="secondary-btn">', unsafe_allow_html=True)
            if st.button("📋 Copy Log", use_container_width=True):
                pass  # Could copy execution log to clipboard
            st.markdown('</div>', unsafe_allow_html=True)

    return {
        "image_type": image_type,
        "body_part": body_part,
        "patient_id": patient_id if patient_id != "—" else None,
        "target_override": target_override if target_override != "auto" else None,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN APP LAYOUT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    render_header()
    config = render_sidebar()

    # ── Main content with tabs ────────────────────────────────────────────
    tab_analyze, tab_report, tab_followup, tab_graph = st.tabs([
        "🔬 Analyze", "📄 Report", "💬 Follow-up", "🔗 Dataflow Graph",
    ])

    # ══════════════════════════════════════════════════════════════════════
    #  TAB 1: ANALYZE
    # ══════════════════════════════════════════════════════════════════════
    with tab_analyze:
        col_left, col_right = st.columns([1.15, 0.85], gap="medium")

        with col_left:
            # ── Image viewer ──
            st.markdown("""
            <div class="panel-card">
                <div class="panel-title">
                    <span class="dot" style="background:var(--accent-cyan)"></span>
                    IMAGE VIEWER
                </div>
            """, unsafe_allow_html=True)

            if st.session_state.uploaded_image:
                st.image(
                    st.session_state.uploaded_image,
                    use_container_width=True,
                    caption=st.session_state.uploaded_image_name,
                )
            else:
                st.markdown("""
                <div class="image-placeholder">
                    <div class="icon">🫁</div>
                    <div>Upload a medical image to begin analysis</div>
                    <div style="font-size:0.72rem;">Supports PNG, JPG, DICOM</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)

            # ── Action buttons ──
            acol1, acol2, acol3, acol4 = st.columns(4)
            with acol1:
                analyze_btn = st.button("🔬 Analyze Image", use_container_width=True, type="primary")
            with acol2:
                localize_btn = st.button("📍 Localize Findings", use_container_width=True)
            with acol3:
                soap_btn = st.button("📋 Generate SOAP", use_container_width=True)
            with acol4:
                pdf_btn = st.button("📥 Export PDF", use_container_width=True)

            # ── Findings display ──
            if st.session_state.findings:
                st.markdown(f"""
                <div class="panel-card">
                    <div class="panel-title">
                        <span class="dot" style="background:var(--accent-emerald)"></span>
                        STRUCTURED FINDINGS
                    </div>
                    {render_findings_card(st.session_state.findings)}
                </div>
                """, unsafe_allow_html=True)

        with col_right:
            # ── Pipeline tracker (real-time) ──
            pipeline_slot = st.empty()

            # Render initial state or last execution
            if st.session_state.pipeline_steps:
                # Show cached steps from last run
                html_parts = []
                for s in st.session_state.pipeline_steps:
                    css_class = f"step-{s['status']}"
                    icon_map = {"pending": "○", "running": "◉", "done": "✓",
                                "error": "✗", "skipped": "−"}
                    icon = icon_map.get(s["status"], "○")
                    time_str = f"{s.get('elapsed', 0):.1f}s" if s.get("elapsed") else ""
                    detail = s.get("detail", "")
                    html_parts.append(f"""
                    <div class="pipeline-step {css_class}">
                        <div class="step-indicator">{icon}</div>
                        <div class="step-info">
                            <div class="step-name">{s['name']}</div>
                            <div class="step-detail">{detail}</div>
                        </div>
                        <div class="step-time">{time_str}</div>
                    </div>""")
                pipeline_slot.markdown(
                    '<div class="panel-card">'
                    '<div class="panel-title"><span class="dot" style="background:var(--accent-blue)"></span>EXECUTION PIPELINE</div>'
                    + "".join(html_parts) + "</div>",
                    unsafe_allow_html=True,
                )
            else:
                pipeline_slot.markdown("""
                <div class="panel-card">
                    <div class="panel-title">
                        <span class="dot" style="background:var(--accent-blue)"></span>
                        EXECUTION PIPELINE
                    </div>
                    <div style="text-align:center;padding:2rem 0;color:var(--text-muted);font-size:0.82rem;">
                        Pipeline steps will appear here during analysis
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # ── Status indicator ──
            status_slot = st.empty()

            # ── Response panel ──
            if st.session_state.analysis_result:
                result = st.session_state.analysis_result
                st.markdown(f"""
                <div class="panel-card">
                    <div class="panel-title">
                        <span class="dot" style="background:var(--accent-violet)"></span>
                        AGENT RESPONSE
                    </div>
                    <div class="report-block">{result.get('response', 'No response.')[:1500]}</div>
                </div>
                """, unsafe_allow_html=True)

                # Metrics
                mc1, mc2, mc3, mc4 = st.columns(4)
                with mc1:
                    st.markdown(f"""<div class="metric-tile">
                        <div class="metric-value">{len(result.get('tools_executed',[]))}</div>
                        <div class="metric-label">Tools Run</div>
                    </div>""", unsafe_allow_html=True)
                with mc2:
                    st.markdown(f"""<div class="metric-tile">
                        <div class="metric-value">{result.get('latency_s', '—')}s</div>
                        <div class="metric-label">Latency</div>
                    </div>""", unsafe_allow_html=True)
                with mc3:
                    st.markdown(f"""<div class="metric-tile">
                        <div class="metric-value">{len(result.get('execution_plan',[]))}</div>
                        <div class="metric-label">Plan Size</div>
                    </div>""", unsafe_allow_html=True)
                with mc4:
                    st.markdown(f"""<div class="metric-tile">
                        <div class="metric-value">{result.get('target_key','—')[:12]}</div>
                        <div class="metric-label">Target Key</div>
                    </div>""", unsafe_allow_html=True)

        # ── Handle button actions ─────────────────────────────────────────
        if analyze_btn:
            image_path = st.session_state.uploaded_image_name or "demo_image.png"
            if st.session_state.agent.session_id is None:
                st.session_state.agent.session_id = SESSION_STORE.create_session()

            result = asyncio.run(execute_with_tracking(
                query=f"Analyze this {config['image_type']} of the {config['body_part']}",
                image_path=image_path,
                image_type=config["image_type"],
                patient_id=config["patient_id"],
                target_key=config.get("target_override") or "verified_analysis",
                pipeline_container=pipeline_slot,
                status_container=status_slot,
            ))
            st.session_state.analysis_result = result

            # Try to extract structured findings
            raw = result.get("raw_results", {})
            for key in ["verify_few_shot_image_analysis", "few_shot_image_analysis"]:
                if key in raw:
                    st.session_state.findings = raw[key]
                    break

            st.session_state.run_history.append({
                "query": "Analyze image",
                "target": "verified_analysis",
                "tools": result.get("tools_executed", []),
                "time": datetime.now().strftime("%H:%M:%S"),
            })
            st.rerun()

        if localize_btn:
            image_path = st.session_state.uploaded_image_name or "demo_image.png"
            if st.session_state.agent.session_id is None:
                st.session_state.agent.session_id = SESSION_STORE.create_session()

            result = asyncio.run(execute_with_tracking(
                query=f"Localize abnormalities on this {config['image_type']}",
                image_path=image_path,
                image_type=config["image_type"],
                patient_id=config["patient_id"],
                target_key="localized_image_path",
                pipeline_container=pipeline_slot,
                status_container=status_slot,
            ))
            st.session_state.analysis_result = result
            raw = result.get("raw_results", {})
            if "localize_abnormalities" in raw:
                st.session_state.localization_data = raw["localize_abnormalities"]
            st.rerun()

        if soap_btn:
            image_path = st.session_state.uploaded_image_name or "demo_image.png"
            if st.session_state.agent.session_id is None:
                st.session_state.agent.session_id = SESSION_STORE.create_session()

            result = asyncio.run(execute_with_tracking(
                query=f"Generate SOAP report for patient {config.get('patient_id') or 'RAD001'}",
                image_path=image_path,
                image_type=config["image_type"],
                patient_id=config["patient_id"] or "RAD001",
                target_key="soap_report",
                pipeline_container=pipeline_slot,
                status_container=status_slot,
            ))
            st.session_state.analysis_result = result
            raw = result.get("raw_results", {})
            if "generate_soap_report" in raw:
                st.session_state.soap_report = raw["generate_soap_report"]
            st.rerun()

        if pdf_btn:
            image_path = st.session_state.uploaded_image_name or "demo_image.png"
            if st.session_state.agent.session_id is None:
                st.session_state.agent.session_id = SESSION_STORE.create_session()

            result = asyncio.run(execute_with_tracking(
                query=f"Generate a full PDF radiology report for patient {config.get('patient_id') or 'RAD001'}",
                image_path=image_path,
                image_type=config["image_type"],
                patient_id=config["patient_id"] or "RAD001",
                target_key="generated_pdf_path",
                pipeline_container=pipeline_slot,
                status_container=status_slot,
            ))
            st.session_state.analysis_result = result
            raw = result.get("raw_results", {})

            # Capture SOAP report along the way (it's in the chain)
            if "generate_soap_report" in raw:
                st.session_state.soap_report = raw["generate_soap_report"]

            # Extract PDF file path from the build_pdf tool result
            if "build_pdf_soap_report" in raw:
                try:
                    pdf_data = json.loads(raw["build_pdf_soap_report"]) \
                        if isinstance(raw["build_pdf_soap_report"], str) \
                        else raw["build_pdf_soap_report"]
                    st.session_state.pdf_path = pdf_data.get("generated_pdf_path")
                except (json.JSONDecodeError, TypeError, AttributeError):
                    st.session_state.pdf_path = None

            # Also try the context data (GroundedArgumentFiller may have stored it)
            if not st.session_state.pdf_path:
                ctx = result.get("context_data", {})
                if ctx.get("generated_pdf_path"):
                    st.session_state.pdf_path = ctx["generated_pdf_path"]

            st.rerun()

    # ══════════════════════════════════════════════════════════════════════
    #  TAB 2: REPORT
    # ══════════════════════════════════════════════════════════════════════
    with tab_report:
        st.markdown("""
        <div class="panel-card">
            <div class="panel-title">
                <span class="dot" style="background:var(--accent-emerald)"></span>
                CLINICAL REPORT
            </div>
        """, unsafe_allow_html=True)

        if st.session_state.soap_report:
            st.markdown(f"""
            <div class="report-block">{st.session_state.soap_report[:3000]}</div>
            """, unsafe_allow_html=True)
        elif st.session_state.analysis_result:
            st.markdown(f"""
            <div class="report-block">{st.session_state.analysis_result.get('response', 'No report generated.')[:3000]}</div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="text-align:center;padding:3rem;color:var(--text-muted);">
                Run an analysis first, then generate a SOAP report.
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        # ── PDF Download ──
        pdf_path = st.session_state.pdf_path
        if pdf_path and os.path.exists(pdf_path):
            file_size = os.path.getsize(pdf_path)
            size_str = f"{file_size / 1024:.1f} KB" if file_size < 1_048_576 else f"{file_size / 1_048_576:.1f} MB"
            st.markdown(f"""
            <div class="panel-card" style="border-left:3px solid var(--accent-emerald);">
                <div class="panel-title">
                    <span class="dot" style="background:var(--accent-emerald)"></span>
                    PDF REPORT READY
                </div>
                <div style="display:flex;align-items:center;justify-content:space-between;">
                    <div>
                        <div style="font-family:var(--font-mono);font-size:0.82rem;color:var(--text-primary);">
                            {os.path.basename(pdf_path)}
                        </div>
                        <div style="font-size:0.72rem;color:var(--text-muted);margin-top:2px;">
                            {size_str} · Generated {datetime.now().strftime('%H:%M:%S')}
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            with open(pdf_path, "rb") as pdf_file:
                st.download_button(
                    label="📥 Download PDF Report",
                    data=pdf_file,
                    file_name=os.path.basename(pdf_path),
                    mime="application/pdf",
                    use_container_width=True,
                )
        elif pdf_path and not os.path.exists(pdf_path):
            st.markdown(f"""
            <div class="panel-card" style="border-left:3px solid var(--accent-amber);">
                <div class="panel-title">
                    <span class="dot" style="background:var(--accent-amber)"></span>
                    PDF GENERATION
                </div>
                <div style="font-size:0.82rem;color:var(--accent-amber);">
                    ⚠ PDF was generated at <code>{pdf_path}</code> but the file was not found on disk.
                    This may happen with a stub LLM — run with MedGemma for full PDF output.
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="panel-card" style="border-left:3px solid var(--border-subtle);">
                <div class="panel-title">
                    <span class="dot" style="background:var(--text-muted)"></span>
                    PDF EXPORT
                </div>
                <div style="font-size:0.82rem;color:var(--text-muted);padding:4px 0;">
                    Click <strong>📥 Export PDF</strong> on the Analyze tab to generate the full
                    radiology report as a downloadable PDF. This runs the longest pipeline chain
                    (10 tools) including image localization and SOAP generation.
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Revision panel
        st.markdown("""
        <div class="panel-card" style="margin-top:0.5rem;">
            <div class="panel-title">
                <span class="dot" style="background:var(--accent-amber)"></span>
                CLINICIAN REVISION
            </div>
        </div>
        """, unsafe_allow_html=True)

        critique = st.text_area(
            "Provide feedback to revise the report:",
            placeholder="e.g., Add bilateral pleural effusion to findings. The lung opacity in the right lower lobe was missed.",
            height=100,
        )
        if st.button("📝 Revise Report", use_container_width=False) and critique:
            st.info("Revision pipeline would run here with the `revise_report` tool...")

    # ══════════════════════════════════════════════════════════════════════
    #  TAB 3: FOLLOW-UP
    # ══════════════════════════════════════════════════════════════════════
    with tab_followup:
        st.markdown("""
        <div class="panel-card">
            <div class="panel-title">
                <span class="dot" style="background:var(--accent-violet)"></span>
                FOLLOW-UP Q&A
            </div>
            <div style="font-size:0.82rem;color:var(--text-muted);margin-bottom:12px;">
                Ask questions about the current analysis without re-running the full pipeline.
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Display QA history
        for qa in st.session_state.qa_history:
            st.markdown(f"""
            <div style="margin-bottom:12px;">
                <div style="font-size:0.78rem;color:var(--accent-cyan);font-weight:600;margin-bottom:4px;">
                    🩺 {qa['question']}
                </div>
                <div class="report-block" style="max-height:200px;">
                    {qa['answer'][:800]}
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Input
        question = st.text_input(
            "Ask a follow-up question:",
            placeholder="e.g., Is the cardiomegaly new compared to prior? Is there a pneumothorax?",
        )
        if st.button("Ask", use_container_width=False) and question:
            # Get session context
            session_summary = SESSION_STORE.get_session_summary(
                st.session_state.agent.session_id or "default"
            )
            if st.session_state.analysis_result:
                session_summary += f"\n\nLast analysis:\n{st.session_state.analysis_result.get('response', '')[:500]}"

            # Call QA tool directly
            from co_radiologist_agent import _qa_followup
            answer = _qa_followup(question, session_summary)
            st.session_state.qa_history.append({
                "question": question,
                "answer": answer,
            })
            st.rerun()

    # ══════════════════════════════════════════════════════════════════════
    #  TAB 4: DATAFLOW GRAPH
    # ══════════════════════════════════════════════════════════════════════
    with tab_graph:
        st.markdown("""
        <div class="panel-card">
            <div class="panel-title">
                <span class="dot" style="background:var(--accent-amber)"></span>
                TOOL DEPENDENCY GRAPH
            </div>
        """, unsafe_allow_html=True)

        engine = DataFlowEngine(radiology_agent_registry, verbose=False)
        highlight = st.session_state.analysis_result.get("tools_executed", []) if st.session_state.analysis_result else []

        # Build a visual graph representation
        for name in radiology_agent_registry.get_all_names():
            tool = radiology_agent_registry.get_tool(name)
            is_active = name in highlight
            border_color = "var(--accent-emerald)" if is_active else "var(--border-subtle)"
            bg = "rgba(16,185,129,.06)" if is_active else "transparent"
            badge = '<span style="font-size:0.6rem;background:var(--accent-emerald);color:white;padding:1px 6px;border-radius:8px;margin-left:6px;">EXECUTED</span>' if is_active else ""

            inputs = ", ".join(tool.arg_sources.values())
            outputs = ", ".join(tool.output_keys.keys())
            type_badge_color = {
                ToolType.RETRIEVAL: "var(--accent-cyan)",
                ToolType.COMPUTATION: "var(--accent-violet)",
                ToolType.KNOWLEDGE: "var(--accent-amber)",
            }.get(tool.tool_type, "var(--text-muted)")

            st.markdown(f"""
            <div style="background:{bg};border:1px solid {border_color};border-radius:8px;
                 padding:10px 14px;margin-bottom:6px;font-family:var(--font-sans);">
                <div style="display:flex;justify-content:space-between;align-items:center;">
                    <div>
                        <span style="font-family:var(--font-mono);font-size:0.82rem;font-weight:600;
                              color:var(--text-primary);">{name}</span>
                        {badge}
                    </div>
                    <span style="font-size:0.6rem;color:{type_badge_color};font-weight:600;
                          text-transform:uppercase;letter-spacing:0.05em;">{tool.tool_type.value}</span>
                </div>
                <div style="font-size:0.72rem;color:var(--text-muted);margin-top:4px;">
                    <span style="color:var(--accent-cyan);">IN:</span> {inputs or '(leaf)'}
                    <span style="margin:0 8px;">→</span>
                    <span style="color:var(--accent-emerald);">OUT:</span> {outputs}
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        # Graph stats
        stats = engine.get_graph_stats()
        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            st.markdown(f"""<div class="metric-tile">
                <div class="metric-value">{stats['total_tools']}</div>
                <div class="metric-label">Total Tools</div>
            </div>""", unsafe_allow_html=True)
        with sc2:
            st.markdown(f"""<div class="metric-tile">
                <div class="metric-value">{stats['unique_producer_keys']}</div>
                <div class="metric-label">Context Keys</div>
            </div>""", unsafe_allow_html=True)
        with sc3:
            st.markdown(f"""<div class="metric-tile">
                <div class="metric-value">{len(stats['source_tools'])}</div>
                <div class="metric-label">Source Tools</div>
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  RUN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    main()
