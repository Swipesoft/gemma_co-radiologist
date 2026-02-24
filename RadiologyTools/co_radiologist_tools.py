
RADIOLOGY_SYSTEM_PROMPT = (
    "You are an expert Co-Radiologist AI assistant. "
    "Provide structured CXR findings based on the clinical data."
)

RADIOLOGY_GOAL_EXAMPLES = (
    "EXAMPLES:\n"
    "Query: Analyze this chest X-ray -> verified_analysis\n"
    "Query: Find similar images -> knn_images\n"
    "Query: Verify the analysis -> verified_analysis\n"
    "Query: What are the findings? -> verified_analysis\n"
)

RADIOLOGY_GOAL_EXAMPLES = (
    "EXAMPLES:\n"
    "Query: Find similar images -> knn_images\n"
    "Query: Analyze this chest X-ray -> few_shot_analysis\n"
    "Query: Verify the analysis -> verified_analysis\n"
    "Query: Localize the abnormalities -> localized_image\n"
    "Query: Get prior imaging studies -> previous_images\n"
    "Query: Compare with previous X-ray -> longitudinal_review\n"
    "Query: Revise the report -> updated_report\n"
    "Query: Pull up patient records -> medical_records\n"
    "Query: Generate SOAP note -> soap_report\n"
    "Query: Export PDF report -> generated_pdf_path\n"
    "Query: Load previous session -> session_context\n"
    "Query: Is the effusion bilateral? -> qa_response\n"
    "Query: Save this analysis -> session_save_confirmation\n"
    "Query: What type of scan is this? -> image_classification\n"
)

RADIOLOGY_GOAL_EXAMPLES = (
    "EXAMPLES:\n"
    # Core analysis chain (Tools 1-3)
    "Query: Find similar images to this X-ray -> knn_images\n"
    "Query: Analyze this chest X-ray -> few_shot_analysis\n"
    "Query: Verify the analysis -> verified_analysis\n"
    "Query: What are the findings? -> verified_analysis\n"
    "Query: Analyze and verify this CXR -> verified_analysis\n"
    # Localization (Tool 4)
    "Query: Where are the abnormalities on this image? -> localized_image\n"
    "Query: Localize the findings -> localized_image\n"
    "Query: Show me bounding boxes for the findings -> localized_image\n"
    # Prior imaging (Tool 5)
    "Query: Get this patient's previous X-rays -> previous_images\n"
    "Query: Retrieve prior imaging studies -> previous_images\n"
    # Longitudinal review (Tool 6)
    "Query: Compare with prior studies -> longitudinal_review\n"
    "Query: Has the cardiomegaly progressed since last time? -> longitudinal_review\n"
    "Query: What changed since the last X-ray? -> longitudinal_review\n"
    # Report revision (Tool 7)
    "Query: Revise the report based on my feedback -> updated_report\n"
    "Query: The right lower lobe opacity was missed, please correct -> updated_report\n"
    # EHR retrieval (Tool 8)
    "Query: Pull up the patient's medical records -> medical_records\n"
    "Query: What are the patient's allergies and medications? -> medical_records\n"
    # SOAP report (Tool 9)
    "Query: Generate a SOAP note -> soap_report\n"
    "Query: Create a clinical report for this patient -> soap_report\n"
    # PDF export (Tool 10)
    "Query: Export the full report as PDF -> generated_pdf_path\n"
    "Query: Build a downloadable radiology report -> generated_pdf_path\n"
    # Session memory (Tools 11, 13)
    "Query: Load my previous session -> session_context\n"
    "Query: Save this analysis for later -> session_save_confirmation\n"
    # Follow-up QA (Tool 12)
    "Query: Is the pleural effusion bilateral? -> qa_response\n"
    "Query: Should we order a follow-up CT? -> qa_response\n"
    # Image classification (Tool 14)
    "Query: What type of scan is this? -> image_classification\n"
    "Query: Identify the imaging modality -> image_classification\n"
)
# ══════════════════════════════════════════════════════════════════════════════
#  TOOL REGISTRATION — nanoathens v0.2.0 Schema
# ══════════════════════════════════════════════════════════════════════════════

radiology_agent_registry = ToolRegistry()

# ── Shared Extractors ─────────────────────────────────────────────────────────
_PATIENT_ID_EXTRACTOR = (
    ArgExtractorType.ALPHANUMERIC_ID,
    {"pattern": r"RAD\d{3}", "preceding_words": ["patient", "id", "rad"]},
)

_SESSION_ID_EXTRACTOR = (
    ArgExtractorType.ALPHANUMERIC_ID,
    {"pattern": r"RAD-[A-Z0-9]{8}", "preceding_words": ["session"]},
)

_IMAGE_TYPE_EXTRACTOR = (
    ArgExtractorType.ENUM, {"values": ["xray", "ct", "mri"]},
)

_QUOTED_EXTRACTOR = (ArgExtractorType.QUOTED, {})


# ═══════════════════════════════════════════════════════════════════════════════
#  TOOL 1: retrieve_similar_images
# ═══════════════════════════════════════════════════════════════════════════════

radiology_agent_registry.register(
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
    docstring="Embeds query image with MedSigLIP-448, searches FAISS index, returns top-K similar cases with cosine similarity scores and CheXpert label descriptions.",
    tool_type=ToolType.RETRIEVAL,
    func=_retrieve_similar_images,
    arg_sources={"patient_image": "patient_image_input", "image_type": "image_type_input"},
    output_keys={"knn_images": "JSON array of similar images with scores and descriptions"},
    explicit_keywords=["similar", "retrieve", "knn", "search", "faiss", "neighbors"],
    arg_extractors={
        "patient_image": _QUOTED_EXTRACTOR,
        "image_type": _IMAGE_TYPE_EXTRACTOR,
    },
)


# ═══════════════════════════════════════════════════════════════════════════════
#  TOOL 2: few_shot_image_analysis
# ═══════════════════════════════════════════════════════════════════════════════

radiology_agent_registry.register(
    name="few_shot_image_analysis",
    description="Analyze CXR using retrieved similar images as few-shot context",
    parameters={
        "knn_images": "JSON string of KNN retrieval results from retrieve_similar_images",
        "patient_image": "Path to the patient image file",
    },
    required=["knn_images", "patient_image"],
    example={
        "knn_images": '{"similar_images": [{"path": "...", "score": 0.89}]}',
        "patient_image": "/kaggle/input/chexpert/valid/patient001/view1_frontal.jpg",
    },
    docstring="Uses top-K similar cases as few-shot context for MedGemma CXR analysis. Builds a prompt with neighbor descriptions and returns structured JSON findings for 14 CheXpert labels.",
    tool_type=ToolType.COMPUTATION,
    func=_few_shot_image_analysis,
    arg_sources={"knn_images": "knn_images", "patient_image": "patient_image_input"},
    output_keys={"few_shot_analysis": "Preliminary CXR analysis JSON string"},
    explicit_keywords=["analyze", "analysis", "few-shot", "diagnose", "findings", "chest"],
    arg_extractors={
        "knn_images": _QUOTED_EXTRACTOR,
        "patient_image": _QUOTED_EXTRACTOR,
    },
)


# ═══════════════════════════════════════════════════════════════════════════════
#  TOOL 3: verify_few_shot_image_analysis
# ═══════════════════════════════════════════════════════════════════════════════

radiology_agent_registry.register(
    name="verify_few_shot_image_analysis",
    description="Verify and validate the CXR analysis with a second LLM pass and Pydantic schema enforcement",
    parameters={
        "analysis": "Preliminary CXR analysis string from few_shot_image_analysis",
        "patient_image": "Path to the patient image file",
    },
    required=["analysis", "patient_image"],
    example={
        "analysis": '{"no_finding": "no", "cardiomegaly": "yes", "lung_opacity": "no"}',
        "patient_image": "/kaggle/input/chexpert/valid/patient001/view1_frontal.jpg",
    },
    docstring="Senior radiologist verification pass. Re-examines the image, corrects errors in the preliminary analysis, and enforces CXRReview Pydantic schema for all 14 CheXpert labels.",
    tool_type=ToolType.COMPUTATION,
    func=_verify_few_shot_image_analysis,
    arg_sources={"analysis": "few_shot_analysis", "patient_image": "patient_image_input"},
    output_keys={"verified_analysis": "Validated CXR analysis JSON (Pydantic-enforced)"},
    explicit_keywords=["verify", "validate", "check", "review", "confirm", "pydantic"],
    arg_extractors={
        "analysis": _QUOTED_EXTRACTOR,
        "patient_image": _QUOTED_EXTRACTOR,
    },
)


# ═══════════════════════════════════════════════════════════════════════════════
#  TOOL 4: localize_abnormalities
# ═══════════════════════════════════════════════════════════════════════════════

radiology_agent_registry.register(
    name="localize_abnormalities",
    description="Localize abnormalities on the CXR using MedGemma bounding box prediction",
    parameters={
        "verified_analysis": "Validated CXR analysis JSON from verify_few_shot_image_analysis",
        "patient_image": "Path to the patient image file",
    },
    required=["verified_analysis", "patient_image"],
    example={
        "verified_analysis": '{"cardiomegaly": "yes", "pleural_effusion": "yes"}',
        "patient_image": "/kaggle/input/chexpert/valid/patient001/view1_frontal.jpg",
    },
    docstring="Extracts positive findings from verified analysis, prompts MedGemma for bounding box coordinates [y0, x0, y1, x1] normalized to [0, 1000], and returns annotated image path with localization data.",
    tool_type=ToolType.COMPUTATION,
    func=_localize_abnormalities,
    arg_sources={"verified_analysis": "verified_analysis", "patient_image": "patient_image_input"},
    output_keys={"localized_image": "JSON with localized_image_path and bounding_boxes array"},
    explicit_keywords=["localize", "locate", "bounding", "box", "annotate", "highlight", "region"],
    arg_extractors={
        "verified_analysis": _QUOTED_EXTRACTOR,
        "patient_image": _QUOTED_EXTRACTOR,
    },
)


# ═══════════════════════════════════════════════════════════════════════════════
#  TOOL 5: retrieve_patient_previous_images
# ═══════════════════════════════════════════════════════════════════════════════

radiology_agent_registry.register(
    name="retrieve_patient_previous_images",
    description="Retrieve a patient's prior imaging studies from EHR/PACS",
    parameters={
        "patient_id": "Patient identifier (e.g. RAD001)",
        "image_type": "Filter by modality: xray|ct|mri",
    },
    required=["patient_id", "image_type"],
    example={"patient_id": "RAD001", "image_type": "xray"},
    docstring="Looks up the patient in RADIOLOGY_EHR_DB, filters prior imaging by modality, returns JSON with study dates, reports, and image paths for longitudinal comparison.",
    tool_type=ToolType.RETRIEVAL,
    func=_retrieve_patient_previous_images,
    arg_sources={"patient_id": "patient_id_input", "image_type": "image_type_input"},
    output_keys={"previous_images": "JSON array of prior imaging studies"},
    explicit_keywords=["previous", "prior", "history", "past", "imaging", "pacs"],
    arg_extractors={
        "patient_id": _PATIENT_ID_EXTRACTOR,
        "image_type": _IMAGE_TYPE_EXTRACTOR,
    },
)


# ═══════════════════════════════════════════════════════════════════════════════
#  TOOL 6: run_longitudinal_review
# ═══════════════════════════════════════════════════════════════════════════════

radiology_agent_registry.register(
    name="run_longitudinal_review",
    description="Compare current study against prior imaging to identify interval changes",
    parameters={
        "patient_image": "Path to the current patient image",
        "previous_images": "JSON from retrieve_patient_previous_images",
        "verified_analysis": "JSON from verify_few_shot_image_analysis",
    },
    required=["patient_image", "previous_images", "verified_analysis"],
    example={
        "patient_image": "/kaggle/input/chexpert/valid/patient001/view1_frontal.jpg",
        "previous_images": '{"previous_images": [{"date": "2025-01-15", "type": "xray", "report": "..."}]}',
        "verified_analysis": '{"cardiomegaly": "yes", "pleural_effusion": "no"}',
    },
    docstring="Senior radiologist longitudinal comparison. Identifies new findings, resolved findings, progressing disease, and stable findings. Returns structured interval change assessment.",
    tool_type=ToolType.COMPUTATION,
    func=_run_longitudinal_review,
    arg_sources={
        "patient_image": "patient_image_input",
        "previous_images": "previous_images",
        "verified_analysis": "verified_analysis",
    },
    output_keys={"longitudinal_review": "Structured longitudinal comparison narrative"},
    explicit_keywords=["longitudinal", "compare", "interval", "change", "progression", "prior"],
    arg_extractors={
        "patient_image": _QUOTED_EXTRACTOR,
        "previous_images": _QUOTED_EXTRACTOR,
        "verified_analysis": _QUOTED_EXTRACTOR,
    },
)


# ═══════════════════════════════════════════════════════════════════════════════
#  TOOL 7: revise_report
# ═══════════════════════════════════════════════════════════════════════════════

radiology_agent_registry.register(
    name="revise_report",
    description="Revise the radiology report based on clinician feedback and critique",
    parameters={
        "clinician_critique": "The clinician's feedback text",
        "verified_analysis": "JSON from verify_few_shot_image_analysis",
        "longitudinal_review": "Optional longitudinal comparison text",
    },
    required=["clinician_critique", "verified_analysis"],
    example={
        "clinician_critique": "Please re-evaluate the right lower lobe opacity",
        "verified_analysis": '{"lung_opacity": "yes", "consolidation": "unclear"}',
        "longitudinal_review": "New finding: right lower lobe opacity not seen on prior.",
    },
    docstring="Radiology report editor. Incorporates clinician corrections while maintaining clinical accuracy. Outputs revised report with INDICATION, TECHNIQUE, FINDINGS, IMPRESSION sections.",
    tool_type=ToolType.COMPUTATION,
    func=_revise_report,
    arg_sources={
        "clinician_critique": "clinician_critique_input",
        "verified_analysis": "verified_analysis",
        "longitudinal_review": "longitudinal_review",
    },
    output_keys={"updated_report": "Revised radiology report text"},
    explicit_keywords=["revise", "edit", "correct", "feedback", "critique", "update", "report"],
    arg_extractors={
        "clinician_critique": _QUOTED_EXTRACTOR,
        "verified_analysis": _QUOTED_EXTRACTOR,
        "longitudinal_review": _QUOTED_EXTRACTOR,
    },
)


# ═══════════════════════════════════════════════════════════════════════════════
#  TOOL 8: retrieve_ehr
# ═══════════════════════════════════════════════════════════════════════════════

radiology_agent_registry.register(
    name="retrieve_ehr",
    description="Retrieve the patient's Electronic Health Record including complaints, notes, prescriptions, and allergies",
    parameters={
        "patient_id": "Patient identifier (e.g. RAD001)",
    },
    required=["patient_id"],
    example={"patient_id": "RAD001"},
    docstring="Retrieves structured EHR data from RADIOLOGY_EHR_DB including demographics, chief complaint, clinical notes, current medications, allergies, and prior imaging count.",
    tool_type=ToolType.RETRIEVAL,
    func=_retrieve_ehr,
    arg_sources={"patient_id": "patient_id_input"},
    output_keys={"medical_records": "Structured text of patient medical records"},
    explicit_keywords=["ehr", "medical", "records", "patient", "history", "chart", "clinical"],
    arg_extractors={
        "patient_id": _PATIENT_ID_EXTRACTOR,
    },
)


# ═══════════════════════════════════════════════════════════════════════════════
#  TOOL 9: generate_soap_report
# ═══════════════════════════════════════════════════════════════════════════════

radiology_agent_registry.register(
    name="generate_soap_report",
    description="Generate a SOAP note integrating EHR data and radiology findings",
    parameters={
        "medical_records": "Text from retrieve_ehr",
        "verified_analysis": "JSON from verify_few_shot_image_analysis",
        "updated_report": "Optional revised report from revise_report",
        "longitudinal_review": "Optional longitudinal comparison text",
    },
    required=["medical_records", "verified_analysis"],
    example={
        "medical_records": "PATIENT: John Doe | Age: 65 | CHIEF COMPLAINT: Shortness of breath",
        "verified_analysis": '{"cardiomegaly": "yes", "pleural_effusion": "yes"}',
    },
    docstring="Generates comprehensive SOAP (Subjective, Objective, Assessment, Plan) note for radiology consultation. Integrates EHR history with imaging findings. Flags critical findings requiring immediate action.",
    tool_type=ToolType.COMPUTATION,
    func=_generate_soap_report,
    arg_sources={
        "medical_records": "medical_records",
        "verified_analysis": "verified_analysis",
        "updated_report": "updated_report",
        "longitudinal_review": "longitudinal_review",
    },
    output_keys={"soap_report": "Complete SOAP note text"},
    explicit_keywords=["soap", "report", "note", "subjective", "objective", "assessment", "plan"],
    arg_extractors={
        "medical_records": _QUOTED_EXTRACTOR,
        "verified_analysis": _QUOTED_EXTRACTOR,
        "updated_report": _QUOTED_EXTRACTOR,
        "longitudinal_review": _QUOTED_EXTRACTOR,
    },
)


# ═══════════════════════════════════════════════════════════════════════════════
#  TOOL 10: build_pdf_soap_report
# ═══════════════════════════════════════════════════════════════════════════════

radiology_agent_registry.register(
    name="build_pdf_soap_report",
    description="Build a PDF document containing the SOAP report and annotated image",
    parameters={
        "soap_report": "SOAP note text from generate_soap_report",
        "localized_image": "JSON from localize_abnormalities with image path and bounding boxes",
        "patient_image": "Original patient image path",
    },
    required=["soap_report", "localized_image", "patient_image"],
    example={
        "soap_report": "SUBJECTIVE: 65yo male with SOB...",
        "localized_image": '{"localized_image_path": "localized_001.png", "bounding_boxes": []}',
        "patient_image": "/kaggle/input/chexpert/valid/patient001/view1_frontal.jpg",
    },
    docstring="Builds a professional PDF radiology report using ReportLab. Includes SOAP sections, localization bounding box data, and annotated CXR image. Falls back to text-based PDF if ReportLab unavailable.",
    tool_type=ToolType.COMPUTATION,
    func=_build_pdf_soap_report,
    arg_sources={
        "soap_report": "soap_report",
        "localized_image": "localized_image",
        "patient_image": "patient_image_input",
    },
    output_keys={"generated_pdf_path": "File path to the generated PDF report"},
    explicit_keywords=["pdf", "document", "report", "generate", "build", "export", "download"],
    arg_extractors={
        "soap_report": _QUOTED_EXTRACTOR,
        "localized_image": _QUOTED_EXTRACTOR,
        "patient_image": _QUOTED_EXTRACTOR,
    },
)


# ═══════════════════════════════════════════════════════════════════════════════
#  TOOL 11: retrieve_session_memory
# ═══════════════════════════════════════════════════════════════════════════════

radiology_agent_registry.register(
    name="retrieve_session_memory",
    description="Retrieve accumulated context from a prior session for follow-up queries",
    parameters={
        "session_id": "Session identifier (e.g. RAD-A1B2C3D4)",
    },
    required=["session_id"],
    example={"session_id": "RAD-A1B2C3D4"},
    docstring="Retrieves full session state from SessionStore including conversation history, accumulated context keys, and prior tool outputs. Enables multi-turn interaction without re-running the pipeline.",
    tool_type=ToolType.RETRIEVAL,
    func=_retrieve_session_memory,
    arg_sources={"session_id": "session_id_input"},
    output_keys={"session_context": "Text summary of all prior context in this session"},
    explicit_keywords=["session", "memory", "context", "history", "previous", "prior", "recall"],
    arg_extractors={
        "session_id": _SESSION_ID_EXTRACTOR,
    },
)


# ═══════════════════════════════════════════════════════════════════════════════
#  TOOL 12: qa_followup
# ═══════════════════════════════════════════════════════════════════════════════

radiology_agent_registry.register(
    name="qa_followup",
    description="Answer a follow-up question about a previous radiology analysis using session context",
    parameters={
        "followup_question": "The clinician's follow-up question",
        "session_context": "Text summary from retrieve_session_memory",
    },
    required=["followup_question", "session_context"],
    example={
        "followup_question": "Is the cardiomegaly getting worse compared to the prior study?",
        "session_context": "Session RAD-A1B2C3D4: verified_analysis shows cardiomegaly=yes...",
    },
    docstring="Conversational follow-up tool. Answers clinician questions using accumulated session context without re-running the full imaging pipeline. Responds only from available context and flags information gaps.",
    tool_type=ToolType.KNOWLEDGE,
    func=_qa_followup,
    arg_sources={
        "followup_question": "followup_question_input",
        "session_context": "session_context",
    },
    output_keys={"qa_response": "Clinically grounded answer to the follow-up question"},
    explicit_keywords=["question", "followup", "ask", "clarify", "explain", "why", "what"],
    arg_extractors={
        "followup_question": _QUOTED_EXTRACTOR,
        "session_context": _QUOTED_EXTRACTOR,
    },
)


# ═══════════════════════════════════════════════════════════════════════════════
#  TOOL 13: store_session_memory
# ═══════════════════════════════════════════════════════════════════════════════

radiology_agent_registry.register(
    name="store_session_memory",
    description="Persist analysis results into the session store for future follow-up queries",
    parameters={
        "session_id": "Session identifier (e.g. RAD-A1B2C3D4)",
        "data_to_store": "Stringified data to persist",
    },
    required=["session_id", "data_to_store"],
    example={
        "session_id": "RAD-A1B2C3D4",
        "data_to_store": '{"verified_analysis": {...}, "soap_report": "..."}',
    },
    docstring="Writes analysis data to the in-memory SessionStore keyed by session_id. Enables multi-turn workflows where follow-up queries can access prior results without re-computation.",
    tool_type=ToolType.COMPUTATION,
    func=_store_session_memory,
    arg_sources={
        "session_id": "session_id_input",
        "data_to_store": "data_to_store_input",
    },
    output_keys={"session_save_confirmation": "Confirmation message that data was stored"},
    explicit_keywords=["store", "save", "persist", "remember", "session", "memory"],
    arg_extractors={
        "session_id": _SESSION_ID_EXTRACTOR,
        "data_to_store": _QUOTED_EXTRACTOR,
    },
)


# ═══════════════════════════════════════════════════════════════════════════════
#  TOOL 14: classify_image_modality
# ═══════════════════════════════════════════════════════════════════════════════

radiology_agent_registry.register(
    name="classify_image_modality",
    description="Auto-detect the imaging modality and body region from the image",
    parameters={
        "patient_image": "Path to the medical image file",
    },
    required=["patient_image"],
    example={"patient_image": "/kaggle/input/chexpert/valid/patient001/view1_frontal.jpg"},
    docstring="Uses MedGemma to classify imaging modality (xray, ct, mri) and body region (chest, head, abdomen) when not explicitly provided by the clinician. Returns JSON with detected image_type and body_part.",
    tool_type=ToolType.COMPUTATION,
    func=_classify_image_modality,
    arg_sources={"patient_image": "patient_image_input"},
    output_keys={"image_classification": "JSON with detected image_type and body_part"},
    explicit_keywords=["classify", "detect", "modality", "type", "identify", "what kind"],
    arg_extractors={
        "patient_image": _QUOTED_EXTRACTOR,
    },
)


# ═══════════════════════════════════════════════════════════════════════════════
#  DAG VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

print(f"✓ {len(radiology_agent_registry)} tools registered")

engine = DataFlowEngine(radiology_agent_registry, verbose=True)

# Validate key paths
test_paths = {
    "verified_analysis": {"patient_image_input", "image_type_input"},
    "localized_image": {"patient_image_input", "image_type_input"},
    "soap_report": {"patient_image_input", "image_type_input", "patient_id_input"},
    "generated_pdf_path": {"patient_image_input", "image_type_input", "patient_id_input"},
    "qa_response": {"session_id_input", "followup_question_input"},
    "longitudinal_review": {"patient_image_input", "image_type_input", "patient_id_input"},
}

print("\nDAG Path Validation:")
for target, sources in test_paths.items():
    plan = engine.resolve_execution_plan(sources, target)
    status = "✓" if plan else "✗"
    print(f"  {status} {target}: {plan}")

print(f"\n{engine.visualize_graph()}")