"""
nanoathens.inference — Model Inference Adapter
═══════════════════════════════════════════════
Wraps any HuggingFace image-text-to-text pipeline into
the DDA-compatible LLM caller interface.
"""

import json
import re
from typing import List, Optional

# ── Pipeline State ────────────────────────────────────────────────────────────

_pipeline = None


def set_pipeline(pipe):
    """Register any HF pipeline as the inference backend."""
    global _pipeline
    _pipeline = pipe


def get_pipeline():
    """Return the current pipeline (or None)."""
    return _pipeline


def load_medgemma(device: str = "auto"):
    """Convenience loader for MedGemma. Requires torch + transformers."""
    global _pipeline
    try:
        import torch
        from transformers import pipeline as hf_pipeline

        dev = "cuda" if torch.cuda.is_available() else "cpu"
        _pipeline = hf_pipeline(
            "image-text-to-text",
            model="google/medgemma-1.5-4b-it",
            device=dev,
            torch_dtype=torch.bfloat16 if dev == "cuda" else torch.float32,
        )
        print(f"✓ MedGemma loaded on {dev}")
    except Exception as e:
        print(f"[WARN] MedGemma not available: {e}")


# ── Unified LLM Caller ───────────────────────────────────────────────────────

def _extract_text(content) -> str:
    """Flatten content blocks and extract text."""
    flat = []
    for item in (content if isinstance(content, list) else [content]):
        if isinstance(item, list):
            flat.extend(item)
        else:
            flat.append(item)
    return "\n".join(
        c["text"] for c in flat if isinstance(c, dict) and c.get("type") == "text"
    )


def _reformat_messages(messages):
    """Convert DDA message format to HF pipeline format."""
    reformatted = []
    for msg in messages:
        new_msg = {"role": msg["role"]}
        content = msg.get("content", "")
        if isinstance(content, str):
            new_msg["content"] = [
                {"type": "text", "source_lang_code": "en",
                 "target_lang_code": "en", "text": content}
            ]
        elif isinstance(content, list):
            new_content = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    new_content.append(
                        {"type": "text", "source_lang_code": "en",
                         "target_lang_code": "en", "text": block.get("text", "")}
                    )
                else:
                    new_content.append(block)
            new_msg["content"] = new_content
        else:
            new_msg["content"] = content
        reformatted.append(new_msg)
    return reformatted


def run_llm(messages=None, max_new_tokens=512, temperature=0.1, **kw) -> str:
    """Call the registered pipeline with DDA-compatible message format.

    Falls back to stub_llm if no pipeline is loaded.
    """
    if messages is None:
        return ""
    if _pipeline is None:
        return _stub_llm(messages, max_new_tokens, temperature)

    try:
        out = _pipeline(
            text=_reformat_messages(messages),
            max_new_tokens=max_new_tokens,
            do_sample=(temperature > 0),
            temperature=temperature if temperature > 0 else None,
            pad_token_id = _pipeline.tokenizer.eos_token_id,  # ← silences the warning
        )
        generated = out[0]["generated_text"][-1]["content"]
        if isinstance(generated, list):
            return _extract_text(generated)
        return str(generated)
    except Exception as e:
        print(f"[LLM] Error: {e}")
        # Return first text block as fallback
        for msg in messages:
            c = msg.get("content", "")
            if isinstance(c, list):
                for b in c:
                    if isinstance(b, dict) and "text" in b:
                        return b["text"]
        return ""


# Backward compatibility alias
run_medgemma = run_llm


# ── Minimal Stub (for testing without GPU) ────────────────────────────────────

def _stub_llm(messages, max_new_tokens=512, temperature=0.1) -> str:
    """Bare-minimum stub. Returns empty JSON for extraction prompts,
    generic text for everything else. Override with set_stub() for
    domain-specific testing."""
    text = ""
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            text += content + " "
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text += block.get("text", "") + " "
    text = text.lower()

    if "which single key" in text:
        return "verified_analysis"
    if "json only" in text:
        return "{}"
    if "fill these parameters" in text:
        return "{}"
    return "Analysis complete. Review tool outputs for findings."


_custom_stub = None



def _stub_llm(messages, max_new_tokens=512, temperature=0.1) -> str:
    """Deterministic stub LLM for testing without GPU."""
    text_parts = []
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            text_parts.append(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
    full_text = " ".join(text_parts).lower()

    # Goal resolution
    if "which single key" in full_text:
        if "generated_pdf" in full_text or "pdf" in full_text:
            return "generated_pdf_path"
        if "soap" in full_text:
            return "soap_report"
        if "localize" in full_text or "bounding" in full_text:
            return "localized_image_path"
        if "verified" in full_text or "verify" in full_text:
            return "verified_analysis"
        if "few_shot" in full_text or "analysis" in full_text:
            return "few_shot_analysis"
        if "longitudinal" in full_text:
            return "longitudinal_review"
        if "report" in full_text and "revise" in full_text:
            return "updated_report"
        if "ehr" in full_text or "medical_records" in full_text:
            return "medical_records"
        if "follow" in full_text or "qa" in full_text or "question" in full_text:
            return "qa_response"
        if "similar" in full_text or "retriev" in full_text:
            return "knn_images"
        return "verified_analysis"

    # JSON extraction
    if "json only" in full_text:
        extracted = {}
        img_match = re.search(
            r"(?:patient image|image)[:\s]+(\S+\.(?:png|jpg|dcm|dicom))", full_text
        )
        if img_match:
            extracted["patient_image_input"] = img_match.group(1)
        for itype in ["xray", "x-ray", "ct", "mri"]:
            if itype in full_text:
                extracted["image_type_input"] = itype.replace("-", "")
                break
        pid_match = re.search(r"(RAD\d{3})", full_text, re.IGNORECASE)
        if pid_match:
            extracted["patient_id_input"] = pid_match.group(1).upper()
        sid_match = re.search(r"(RAD-[A-Z0-9]{8})", full_text)
        if sid_match:
            extracted["session_id_input"] = sid_match.group(1)
        crit_match = re.search(
            r"(?:critique|feedback)[:\s]+['\"](.+?)['\"]", full_text
        )
        if crit_match:
            extracted["clinician_critique_input"] = crit_match.group(1)
        if "follow" in full_text or "question" in full_text:
            for msg in messages:
                c = msg.get("content", "")
                if isinstance(c, list):
                    for b in c:
                        if isinstance(b, dict) and "text" in b:
                            txt = b["text"]
                            if "USER GOAL:" in txt or "TEXT:" in txt:
                                goal_match = re.search(
                                    r"(?:USER GOAL|TEXT):\s*(.+?)(?:\n|$)", txt
                                )
                                if goal_match:
                                    extracted["followup_question_input"] = (
                                        goal_match.group(1).strip()
                                    )
        if extracted:
            return json.dumps(extracted)
        return "{}"

    # Parameter filling
    if "fill these parameters" in full_text:
        filled = {}
        params_match = re.search(r"Parameters.*?({.*?})", full_text, re.DOTALL)
        if params_match:
            try:
                needed = json.loads(params_match.group(1))
                for param_name in needed:
                    pn = param_name.lower()
                    if "image" in pn and "type" not in pn:
                        img_m = re.search(r"patient_image_input:\s*(\S+)", full_text)
                        if img_m:
                            filled[param_name] = img_m.group(1)
                    elif "image_type" in pn:
                        for itype in ["xray", "ct", "mri"]:
                            if itype in full_text:
                                filled[param_name] = itype
                                break
                    elif "patient_id" in pn:
                        pid_m = re.search(r"(RAD\d{3})", full_text, re.IGNORECASE)
                        if pid_m:
                            filled[param_name] = pid_m.group(1).upper()
                    elif "session_id" in pn:
                        sid_m = re.search(r"(RAD-[A-Z0-9]{8})", full_text)
                        if sid_m:
                            filled[param_name] = sid_m.group(1)
                    elif "critique" in pn:
                        filled[param_name] = "Clinician feedback"
                    elif "question" in pn or "followup" in pn:
                        filled[param_name] = "Follow-up question"
            except (json.JSONDecodeError, KeyError):
                pass
        if filled:
            return json.dumps(filled)
        return "{}"

    # Default synthesis
    return (
        "Based on the clinical data provided, the analysis has been completed. "
        "Please review the tool outputs for detailed findings."
    )


def set_stub(fn):
    """Register a custom stub for domain-specific testing."""
    global _custom_stub, _stub_llm
    _custom_stub = fn
    _stub_llm = fn