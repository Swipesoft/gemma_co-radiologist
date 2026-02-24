# ══════════════════════════════════════════════════════════════════════════════
#  TOOL FUNCTIONS — Gemma Co-Radiologist
# ══════════════════════════════════════════════════════════════════════════════

def _retrieve_similar_images(patient_image: str, image_type: str, k: int = 5) -> str:
    """Retrieve similar images using MedSigLIP + FAISS."""
    if os.path.exists(patient_image):
        query_vec  = embed_query_image(patient_image)
        query_type = "image"
    else:
        inputs = siglip_processor(text=[f"{image_type} chest radiograph"], return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            outputs = siglip_model.get_text_features(**inputs)
            emb = outputs if torch.is_tensor(outputs) else getattr(outputs, "pooler_output", outputs[0])
        emb = emb / emb.norm(dim=-1, keepdim=True)
        query_vec = emb.cpu().float().numpy().astype("float32")
        query_type = "text"

    rel_path = patient_image.replace(KAGGLE_ROOT + "/", "")
    query_desc = description_lookup.get(rel_path, "No description.")
    results = faiss_retrieve(query_vec, k=k)

    return json.dumps({
        "query_image": patient_image, "query_description": query_desc,
        "image_type": image_type, "query_type": query_type,
        "k": len(results), "similar_images": results,
    }, indent=2)


def _few_shot_image_analysis(knn_images: str, patient_image: str) -> str:
    """Analyze image using KNN examples as few-shot context."""
    try:
        knn_data = json.loads(knn_images) if isinstance(knn_images, str) else knn_images
        similar  = knn_data.get("similar_images", [])
    except (json.JSONDecodeError, TypeError):
        similar = []

    context_parts = []
    for r in similar[:5]:
        context_parts.append(f"- Similar case (score={r.get('score', 0)}): {r.get('description', '')}")
    context_str = "\n".join(context_parts) if context_parts else "No similar cases found."

    prompt = (
        f"You are an expert radiologist performing chest X-ray analysis.\n\n"
        f"SIMILAR CASES FROM DATABASE:\n{context_str}\n\n"
        f"{CXR_ANALYSIS_PROMPT}\n\n"
        f"Analyze the patient's chest X-ray and return ONLY the JSON object."
    )

    content_blocks = [{"type": "text", "text": prompt}]
    if os.path.exists(patient_image):
        content_blocks.insert(0, {"type": "image", "image": Image.open(patient_image).convert("RGB")})

    response = run_medgemma(
        messages=[{"role": "user", "content": content_blocks}],
        max_new_tokens=512, temperature=0.1,
    )
    return response


def _verify_few_shot_image_analysis(analysis: str, patient_image: str) -> str:
    """Verify and correct the analysis with a second LLM pass + Pydantic validation."""
    prompt = (
        f"You are a senior radiologist verifying a preliminary chest X-ray report.\n\n"
        f"PRELIMINARY ANALYSIS:\n{analysis}\n\n"
        f"Review the image again and correct any errors. "
        f"{CXR_ANALYSIS_PROMPT}"
    )

    content_blocks = [{"type": "text", "text": prompt}]
    if os.path.exists(patient_image):
        content_blocks.insert(0, {"type": "image", "image": Image.open(patient_image).convert("RGB")})

    response = run_medgemma(
        messages=[{"role": "user", "content": content_blocks}],
        max_new_tokens=512, temperature=0.1,
    )

    # Try to parse and validate with Pydantic
    try:
        s = response.find("{")
        e = response.rfind("}")
        if s != -1 and e > s:
            data = json.loads(response[s:e+1])
            review = CXRReview(**data)
            return review.model_dump_json()
    except Exception:
        pass

    return response


# ─── Tool 4: localize_abnormalities ───────────────────────────────────────────
# Based on cxr_anatomy_localization_with_hugging_face.ipynb

def _localize_abnormalities(verified_analysis: str, patient_image: str) -> str:
    """Localize abnormalities on the patient image using MedGemma's
    bounding box prediction capability.

    Based on the CXR anatomy localization notebook:
    1. Extract positive findings from verified analysis
    2. For each finding, prompt MedGemma for bounding box coordinates
    3. Return image path with drawn bounding boxes

    Bounding box format: [y0, x0, y1, x1] normalized to [0, 1000]

    Args:
        verified_analysis: JSON string from verify_few_shot_image_analysis
        patient_image: Path/URL/b64 of the patient's image

    Returns:
        JSON string with localized_image_path and bounding_boxes
    """
    try:
        analysis = json.loads(verified_analysis) if isinstance(verified_analysis, str) else verified_analysis
    except (json.JSONDecodeError, TypeError):
        analysis = {}

    # Extract positive / uncertain findings to localize
    findings_to_localize = []
    skip_keys = {"_meta", "verification_confidence", "verification_notes",
                 "view", "section", "other_abnormal_features"}
    for key, value in analysis.items():
        if key in skip_keys or key.startswith("_"):
            continue
        val_str = str(value).lower()
        if val_str in ("yes", "unclear", "true"):
            # Convert key to anatomical label
            label = key.replace("_", " ").title()
            findings_to_localize.append(label)

    if not findings_to_localize:
        return json.dumps({
            "localized_image_path": patient_image,
            "bounding_boxes": [],
            "message": "No positive findings to localize.",
        })

    # For each finding, generate bounding box prompt
    # (In production, this queries MedGemma with the image)
    all_boxes = []
    for finding in findings_to_localize:
        localization_prompt = (
            f'Instructions:\nThe following user query will require outputting '
            f'bounding boxes. The format of bounding boxes coordinates is '
            f'[y0, x0, y1, x1] where (y0, x0) must be top-left corner and '
            f'(y1, x1) the bottom-right corner. Always normalize the x and y '
            f'coordinates the range [0, 1000].\n'
            f'You MUST output a single parseable json list of objects enclosed '
            f'into ```json...``` brackets.\n\n'
            f'Remember "left" refers to the patient\'s left side where the '
            f'heart is.\n\n'
            f'Query:\nWhere is the {finding}? Output the final answer in the '
            f'format "Final Answer: X" where X is a JSON list of objects with '
            f'"box_2d" and "label" keys. Answer:'
        )

        messages = [{"role": "user", "content": [
            # In production: {"type": "image", "image": load_image(patient_image)},
            {"type": "text", "text": localization_prompt},
        ]}]

        response = run_medgemma(messages=messages, max_new_tokens=500, temperature=0.0)

        # Parse bounding boxes from response
        json_match = re.search(r"```json\s*(.*?)```", response, re.DOTALL)
        if json_match:
            try:
                boxes = json.loads(json_match.group(1).strip())
                if isinstance(boxes, list):
                    all_boxes.extend(boxes)
                    continue
            except json.JSONDecodeError:
                pass

        # Fallback: try extracting from "Final Answer:" pattern
        fa_match = re.search(r"Final Answer:\s*(\[.*?\])", response, re.DOTALL)
        if fa_match:
            try:
                boxes = json.loads(fa_match.group(1))
                if isinstance(boxes, list):
                    all_boxes.extend(boxes)
                    continue
            except json.JSONDecodeError:
                pass

        # If MedGemma didn't produce boxes, add a placeholder
        all_boxes.append({
            "box_2d": None,
            "label": finding,
            "note": "Localization unavailable — manual review recommended",
        })

    # In production: draw bounding boxes on image and save
    # output_path = _draw_bounding_boxes(patient_image, all_boxes)
    output_path = f"localized_{Path(patient_image).stem}_{uuid.uuid4().hex[:6]}.png"

    return json.dumps({
        "localized_image_path": output_path,
        "bounding_boxes": all_boxes,
        "num_findings_localized": len(findings_to_localize),
        "findings": findings_to_localize,
    })


# ─── Tool 5: retrieve_patient_previous_images ────────────────────────────────

def _retrieve_patient_previous_images(patient_id: str, image_type: str) -> str:
    """Retrieve a patient's prior imaging studies from the EHR/PACS.

    Args:
        patient_id: Patient identifier (e.g. RAD001)
        image_type: Filter by modality ('xray', 'ct', 'mri')

    Returns:
        JSON string of prior images with dates and reports, or None indicator
    """
    patient = RADIOLOGY_EHR_DB.get(patient_id.upper())
    if not patient:
        return json.dumps({
            "patient_id": patient_id,
            "previous_images": [],
            "message": f"Patient {patient_id} not found in database.",
        })

    prior = patient.get("prior_imaging", [])
    # Filter by image type
    filtered = [
        img for img in prior
        if img.get("type", "").lower() == image_type.lower()
    ]

    return json.dumps({
        "patient_id": patient_id,
        "patient_name": patient.get("name", "Unknown"),
        "num_prior_studies": len(filtered),
        "previous_images": filtered,
    })


# ─── Tool 6: run_longitudinal_review ─────────────────────────────────────────

def _run_longitudinal_review(
    patient_image: str, previous_images: str, verified_analysis: str
) -> str:
    """Compare the current study against prior imaging studies to identify
    interval changes: new findings, resolved findings, progressing disease.

    Args:
        patient_image: Current image path
        previous_images: JSON from retrieve_patient_previous_images
        verified_analysis: JSON from verify_few_shot_image_analysis

    Returns:
        String narrative of longitudinal comparison
    """
    try:
        priors = json.loads(previous_images) if isinstance(previous_images, str) else previous_images
        analysis = json.loads(verified_analysis) if isinstance(verified_analysis, str) else verified_analysis
    except (json.JSONDecodeError, TypeError):
        priors = {"previous_images": []}
        analysis = {}

    prior_list = priors.get("previous_images", [])
    if not prior_list:
        return (
            "LONGITUDINAL REVIEW: No prior imaging available for comparison. "
            "This is the baseline study. Current findings documented in the "
            "verified analysis should be used as the reference point for "
            "future comparisons."
        )

    # Build comparison prompt
    prior_summaries = "\n".join(
        f"  - {p.get('date', 'Unknown date')} [{p.get('type', '?')}]: "
        f"{p.get('report', 'No report')}"
        for p in prior_list
    )

    current_findings = "\n".join(
        f"  {k}: {v}" for k, v in analysis.items()
        if not k.startswith("_") and v is not None
    )

    prompt = (
        f"You are a Senior Radiologist performing a longitudinal comparison.\n\n"
        f"CURRENT STUDY FINDINGS:\n{current_findings}\n\n"
        f"PRIOR STUDIES:\n{prior_summaries}\n\n"
        f"Provide a structured comparison:\n"
        f"1. NEW FINDINGS: What is seen now but not before?\n"
        f"2. RESOLVED FINDINGS: What was seen before but not now?\n"
        f"3. PROGRESSING/WORSENING: What has gotten worse?\n"
        f"4. STABLE FINDINGS: What is unchanged?\n"
        f"5. OVERALL INTERVAL CHANGE ASSESSMENT: "
        f"[Improved / Stable / Worsened / Mixed]\n"
    )

    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    review = run_medgemma(messages=messages, max_new_tokens=1024, temperature=0.2)
    return review


# ─── Tool 7: revise_report ───────────────────────────────────────────────────

def _revise_report(
    clinician_critique: str,
    verified_analysis: str,
    longitudinal_review: str = None,
) -> str:
    """Revise the radiology report based on clinician feedback/critique.

    The clinician reviews the verified analysis and provides corrections,
    additions, or requests for clarification. This tool incorporates
    that feedback into an updated report.

    Args:
        clinician_critique: The clinician's feedback text
        verified_analysis: JSON from verify_few_shot_image_analysis
        longitudinal_review: Optional longitudinal comparison text

    Returns:
        Updated report as structured text
    """
    try:
        analysis = json.loads(verified_analysis) if isinstance(verified_analysis, str) else verified_analysis
    except (json.JSONDecodeError, TypeError):
        analysis = {}

    context_parts = [f"ORIGINAL ANALYSIS:\n{json.dumps(analysis, indent=2)}"]
    if longitudinal_review:
        context_parts.append(f"\nLONGITUDINAL REVIEW:\n{longitudinal_review}")

    prompt = (
        f"You are a Radiology report editor. A supervising clinician has "
        f"provided feedback on the draft report. Incorporate their critique "
        f"while maintaining clinical accuracy.\n\n"
        f"{''.join(context_parts)}\n\n"
        f"CLINICIAN CRITIQUE:\n{clinician_critique}\n\n"
        f"Generate the REVISED REPORT. Maintain all accurate original "
        f"findings, incorporate the clinician's corrections, and flag any "
        f"remaining uncertainties. Format as a standard radiology report with "
        f"sections: INDICATION, TECHNIQUE, FINDINGS, IMPRESSION."
    )

    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    revised = run_medgemma(messages=messages, max_new_tokens=1024, temperature=0.2)
    return revised


# ─── Tool 8: retrieve_ehr ────────────────────────────────────────────────────

def _retrieve_ehr(patient_id: str) -> str:
    """Retrieve the patient's Electronic Health Record including
    complaints, clinical notes, prescriptions, and allergies.

    Args:
        patient_id: Patient identifier (e.g. RAD001)

    Returns:
        Structured text of the patient's medical records
    """
    patient = RADIOLOGY_EHR_DB.get(patient_id.upper())
    if not patient:
        return (
            f"No EHR records found for patient {patient_id}. "
            f"Please verify the patient ID."
        )

    record = (
        f"PATIENT: {patient.get('name', 'Unknown')} | "
        f"Age: {patient.get('age', '?')} | Sex: {patient.get('sex', '?')}\n"
        f"CHIEF COMPLAINT: {patient.get('complaints', 'Not documented')}\n"
        f"CLINICAL NOTES: {patient.get('clinical_notes', 'None')}\n"
        f"CURRENT MEDICATIONS: {patient.get('prescriptions', 'None')}\n"
        f"ALLERGIES: {', '.join(patient.get('allergies', [])) or 'NKDA'}\n"
        f"PRIOR IMAGING: {len(patient.get('prior_imaging', []))} studies on file"
    )
    return record


# ─── Tool 9: generate_soap_report ────────────────────────────────────────────

def _generate_soap_report(
    medical_records: str,
    verified_analysis: str,
    updated_report: str = None,
    longitudinal_review: str = None,
) -> str:
    """Generate a SOAP (Subjective, Objective, Assessment, Plan) note
    integrating EHR data and radiology findings.

    Args:
        medical_records: Text from retrieve_ehr
        verified_analysis: JSON from verify_few_shot_image_analysis
        updated_report: Optional revised report from revise_report
        longitudinal_review: Optional longitudinal comparison

    Returns:
        Structured SOAP report text
    """
    # Use updated_report if available, otherwise verified_analysis
    radiology_data = updated_report or verified_analysis

    context_parts = [
        f"PATIENT RECORDS:\n{medical_records}",
        f"\nRADIOLOGY FINDINGS:\n{radiology_data}",
    ]
    if longitudinal_review:
        context_parts.append(f"\nLONGITUDINAL REVIEW:\n{longitudinal_review}")

    prompt = (
        f"Generate a comprehensive SOAP note for a radiology consultation.\n\n"
        f"{''.join(context_parts)}\n\n"
        f"Format:\n"
        f"SUBJECTIVE: Patient's presenting complaints and relevant history\n"
        f"OBJECTIVE: Imaging findings (from the radiology analysis), vitals, labs\n"
        f"ASSESSMENT: Clinical interpretation integrating history + imaging\n"
        f"PLAN: Recommended next steps, follow-up imaging, referrals\n\n"
        f"Be concise but thorough. Flag any critical findings requiring "
        f"immediate action."
    )

    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    soap = run_medgemma(messages=messages, max_new_tokens=1024, temperature=0.2)
    return soap


# ─── Tool 10: build_pdf_soap_report ──────────────────────────────────────────

def _build_pdf_soap_report(
    soap_report: str, localized_image: str, patient_image: str
) -> str:
    """Build a PDF document containing the SOAP report and annotated image.

    Args:
        soap_report: Text from generate_soap_report
        localized_image: JSON from localize_abnormalities (contains image path)
        patient_image: Original patient image path

    Returns:
        File path to the generated PDF
    """
    try:
        loc_data = json.loads(localized_image) if isinstance(localized_image, str) else localized_image
    except (json.JSONDecodeError, TypeError):
        loc_data = {}

    localized_path = loc_data.get("localized_image_path", patient_image)
    bboxes = loc_data.get("bounding_boxes", [])

    # Generate PDF using reportlab or simple text-based fallback
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_filename = f"radiology_report_{timestamp}.pdf"

    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import mm
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Image as RLImage,
        )
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

        doc = SimpleDocTemplate(pdf_filename, pagesize=A4)
        styles = getSampleStyleSheet()

        title_style = ParagraphStyle(
            "ReportTitle", parent=styles["Title"], fontSize=16,
            spaceAfter=12,
        )
        heading_style = ParagraphStyle(
            "SectionHead", parent=styles["Heading2"], fontSize=12,
            spaceAfter=6, spaceBefore=12,
        )
        body_style = ParagraphStyle(
            "ReportBody", parent=styles["Normal"], fontSize=10,
            spaceAfter=4, leading=14,
        )

        story = []
        story.append(Paragraph("RADIOLOGY CONSULTATION REPORT", title_style))
        story.append(Paragraph(
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            body_style,
        ))
        story.append(Spacer(1, 12))

        # Add SOAP report sections
        for line in soap_report.split("\n"):
            line = line.strip()
            if not line:
                story.append(Spacer(1, 6))
            elif line.startswith(("SUBJECTIVE", "OBJECTIVE", "ASSESSMENT", "PLAN")):
                story.append(Paragraph(line, heading_style))
            else:
                # Escape XML-sensitive characters for reportlab
                safe = line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                story.append(Paragraph(safe, body_style))

        # Add localization info
        if bboxes:
            story.append(Spacer(1, 12))
            story.append(Paragraph("LOCALIZED FINDINGS", heading_style))
            for box in bboxes:
                label = box.get("label", "Unknown")
                coords = box.get("box_2d", "N/A")
                story.append(Paragraph(
                    f"• {label}: coordinates {coords}", body_style
                ))

        # Add image if file exists
        if os.path.exists(localized_path):
            story.append(Spacer(1, 12))
            story.append(Paragraph("ANNOTATED IMAGE", heading_style))
            story.append(RLImage(localized_path, width=150 * mm, height=150 * mm))

        doc.build(story)

    except ImportError:
        # Fallback: write as text file with .pdf extension
        with open(pdf_filename, "w") as f:
            f.write("=" * 60 + "\n")
            f.write("  RADIOLOGY CONSULTATION REPORT\n")
            f.write(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
            f.write("=" * 60 + "\n\n")
            f.write(soap_report)
            f.write("\n\n" + "-" * 40 + "\n")
            f.write("LOCALIZED FINDINGS\n")
            for box in bboxes:
                f.write(f"  {box.get('label', '?')}: {box.get('box_2d', 'N/A')}\n")
            f.write(f"\nImages: {patient_image}, {localized_path}\n")

    return json.dumps({
        "generated_pdf_path": pdf_filename,
        "includes_localization": len(bboxes) > 0,
        "timestamp": datetime.now().isoformat(),
    })


# ─── Tool 11: retrieve_session_memory (ADDITIONAL) ───────────────────────────

def _retrieve_session_memory(session_id: str) -> str:
    """Retrieve accumulated context from a prior session to enable
    follow-up questions and iterative refinement.

    Args:
        session_id: Session identifier

    Returns:
        Text summary of all prior context in this session
    """
    return SESSION_STORE.get_session_summary(session_id)


# ─── Tool 12: qa_followup (ADDITIONAL) ───────────────────────────────────────

def _qa_followup(followup_question: str, session_context: str) -> str:
    """Answer a follow-up question about a previous radiology analysis
    using the accumulated session context.

    This tool enables conversational interaction: after the initial
    image review, the clinician can ask "What about the effusion?"
    or "Is cardiomegaly progressing?" without re-running the full pipeline.

    Args:
        followup_question: The clinician's follow-up question
        session_context: Text summary from retrieve_session_memory

    Returns:
        Clinically grounded answer based on prior context
    """
    prompt = (
        f"You are an expert Co-Radiologist answering a follow-up question "
        f"about a radiology case you previously reviewed.\n\n"
        f"SESSION CONTEXT (prior analysis data):\n{session_context}\n\n"
        f"CLINICIAN'S FOLLOW-UP QUESTION: {followup_question}\n\n"
        f"Answer the question using ONLY information from the session context. "
        f"If the information needed is not available in context, say so clearly "
        f"and suggest what additional analysis might be needed. "
        f"Be concise and clinically precise."
    )

    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    answer = run_medgemma(messages=messages, max_new_tokens=512, temperature=0.2)
    return answer


# ─── Tool 13: store_session_memory (ADDITIONAL) ──────────────────────────────

def _store_session_memory(session_id: str, data_to_store: str) -> str:
    """Persist analysis results into the session store for future
    follow-up queries.

    Args:
        session_id: Session identifier
        data_to_store: Stringified data to store

    Returns:
        Confirmation message
    """
    SESSION_STORE.save_context(session_id, f"saved_{datetime.now().isoformat()}", data_to_store)
    return f"Session {session_id}: data stored successfully."


# ─── Tool 14: classify_image_modality (ADDITIONAL) ───────────────────────────

def _classify_image_modality(patient_image: str) -> str:
    """Auto-detect the imaging modality and body region from the image
    when not explicitly provided by the clinician.

    Args:
        patient_image: Path/URL/b64 of the image

    Returns:
        JSON with detected image_type and body_part
    """
    prompt = (
        f"Identify the medical imaging modality and body region.\n"
        f"Image: {patient_image}\n\n"
        f"Return ONLY a JSON object with:\n"
        f'{{"image_type": "xray|ct|mri", "body_part": "chest|head|abdomen|brain|other"}}'
    )
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    # In production, include image block
    response = run_medgemma(messages=messages, max_new_tokens=100, temperature=0.0)
    parsed = extract_and_validate_json(response)

    if "image_type" not in parsed:
        parsed["image_type"] = "xray"
    if "body_part" not in parsed:
        parsed["body_part"] = "chest"

    return json.dumps(parsed)


print("All 14 tool functions defined")