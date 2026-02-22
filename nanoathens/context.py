"""
athena_dda.context — Value Extraction & Context Bank (Installable SDK)
═════════════════════════════════════════════════════
BaseValueExtractor, LLMValueExtractor, ContextBank.
"""

import json
import re
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional

from .core import ArgExtractorType


class BaseValueExtractor(ABC):
    @abstractmethod
    def extract_from_goal(self, goal, known_keys=None):
        pass

    @abstractmethod
    def extract_from_result(self, tool_name, result, output_keys=None):
        pass


class LLMValueExtractor(BaseValueExtractor):
    def __init__(self, llm_caller: Callable, verbose=True):
        self.llm = llm_caller
        self.verbose = verbose

    def _log(self, msg):
        if self.verbose:
            print(f"[LLMExtractor] {msg}")

    def _call_llm(self, prompt, max_tokens=512):
        messages = [
            {"role": "user", "content": [{"type": "text", "text": prompt}]}
        ]
        resp = self.llm(messages=messages, max_new_tokens=max_tokens, temperature=0.0)
        for st, et in [("<unused94>", "<unused95>"), ("<think>", "</think>")]:
            while st in resp:
                s = resp.find(st)
                e = resp.find(et)
                resp = resp[:s] + (resp[e + len(et):] if e > s else "")
        return resp.strip()

    def _parse_json(self, response):
        s = response.find("{")
        e = response.rfind("}")
        if s != -1 and e > s:
            try:
                return {
                    k: v
                    for k, v in json.loads(response[s : e + 1]).items()
                    if v not in (None, "null", "")
                }
            except Exception:
                pass
        return {}

    def extract_from_goal(self, goal, known_keys=None):
        keys_str = "\n".join(f"- {k}" for k in (known_keys or []))
        prompt = (
            f"Extract ONLY values explicitly stated. Use null for anything "
            f"not stated.\n\nTEXT: {goal}\n\nFields:\n{keys_str}\n\nJSON only:"
        )
        vals = self._parse_json(self._call_llm(prompt))
        self._log(f"From goal: {list(vals.keys())}")
        return vals

    def extract_from_result(self, tool_name, result, output_keys=None):
        keys_spec = "\n".join(
            f"- {k} ({t})" for k, t in (output_keys or {}).items()
        )
        prompt = (
            f"Extract these values from the tool result. Use null if not "
            f"found.\n\nTOOL: {tool_name}\nRESULT: {result}\n\nValues:\n"
            f"{keys_spec}\n\nJSON only:"
        )
        vals = self._parse_json(self._call_llm(prompt))
        vals[f"_raw_{tool_name}"] = result
        return vals


class ContextBank:
    """Accumulates context values across the DDA execution pipeline."""

    # Alias map for anatomical and clinical adjectives → canonical enum values
    _ENUM_ALIASES = {
        "abdominal": "abdomen", "thoracic": "thorax", "cervical": "head",
        "cranial": "head", "pelvic": "pelvis", "spinal": "spine",
        "standard": "standard", "comprehensive": "comprehensive",
        "urgent": "urgent", "routine": "routine",
        "chest": "chest", "brain": "brain", "frontal": "frontal",
        "lateral": "lateral", "xray": "xray", "x-ray": "xray",
        "ct": "ct", "mri": "mri",
    }

    def __init__(self, extractor: BaseValueExtractor, known_keys=None, registry=None):
        self.extractor = extractor
        self.known_keys = known_keys or []
        self.registry = registry
        self.user_goal = ""
        self.retrieved_data: Dict[str, str] = {}
        self.parsed_values: Dict[str, Any] = {}

    def set_goal(self, goal):
        self.user_goal = goal
        if self.registry:
            self.parsed_values.update(self._extract_from_schema(goal))
        for k, v in self.extractor.extract_from_goal(goal, self.known_keys).items():
            if k not in self.parsed_values:
                self.parsed_values[k] = v

    def _extract_from_schema(self, goal):
        """Extract values by matching against tool parameter schemas."""
        values = {}
        gl = goal.lower()
        for tool in self.registry._tools.values():
            # Try to match enum values from tool parameters
            parameters = getattr(tool, "parameters", None)
            if not parameters:
                continue
            for pname, pdesc in parameters.items():
                for part in str(pdesc).split():
                    if "|" in part:
                        for val in part.strip(".,;:()[]").split("|"):
                            canon = val.strip().lower()
                            ctx_key = tool.arg_sources.get(pname, pname)
                            if ctx_key in values:
                                continue
                            if canon in gl:
                                values[ctx_key] = canon
                            else:
                                for alias, canonical in self._ENUM_ALIASES.items():
                                    if canonical == canon and alias in gl:
                                        values[ctx_key] = canon
                                        break

        # Try arg_extractor based extraction
        try:
            extractors = self.registry.get_all_arg_extractors()
        except AttributeError:
            extractors = {}

        for ctx_key, (ext_type, cfg) in extractors.items():
            if ctx_key in values:
                continue
            if ext_type == ArgExtractorType.QUOTED:
                for q in ["'", '"']:
                    pts = goal.split(q)
                    if len(pts) >= 3 and len(pts[1]) > 2:
                        values[ctx_key] = pts[1]
                        break
            elif ext_type == ArgExtractorType.ALPHANUMERIC_ID:
                pattern = cfg.get("pattern", r"[A-Za-z]\d+")
                words = goal.split()
                preceding = [p.lower() for p in cfg.get("preceding_words", [])]
                compiled = re.compile(f"^{pattern}$", re.IGNORECASE)
                for i, w in enumerate(words):
                    cw = w.rstrip(".,;:!?")
                    if compiled.match(cw):
                        if (
                            i > 0
                            and words[i - 1].lower().rstrip(":#.,;") in preceding
                        ) or not preceding:
                            values[ctx_key] = cw.upper()
                            break
            elif ext_type == ArgExtractorType.NUMERIC_ID:
                words = goal.split()
                for i, w in enumerate(words):
                    cw = w.rstrip(".,;:!?")
                    if cw.isdigit() and len(cw) >= 2 and i > 0:
                        prev = words[i - 1].lower().rstrip(":#.,;")
                        if prev in cfg.get("preceding_words", []):
                            values[ctx_key] = cw
                            break
            elif ext_type == ArgExtractorType.NUMBER:
                for unit in cfg.get("units", []):
                    m = re.search(rf"(\d+\.?\d*)\s*{re.escape(unit)}", gl)
                    if m:
                        values[ctx_key] = float(m.group(1))
                        break
        return values

    def add_tool_result(self, tool_name, result, output_keys=None):
        self.retrieved_data[tool_name] = result
        self.parsed_values.update(
            self.extractor.extract_from_result(
                tool_name, result, output_keys or {}
            )
        )

    def has_value(self, key):
        return key in self.parsed_values

    def get_value(self, key, default=None):
        return self.parsed_values.get(key, default)

    def get_grounding_context(self):
        lines = [f"USER GOAL: {self.user_goal}"]
        if self.parsed_values:
            lines += ["EXTRACTED VALUES:"] + [
                f"  {k}: {str(v)[:120]}"
                for k, v in self.parsed_values.items()
            ]
        if self.retrieved_data:
            lines += ["TOOL RESULTS:"] + [
                f"  [{t}]: {r[:200]}"
                for t, r in self.retrieved_data.items()
            ]
        return "\n".join(lines)

    def reset(self):
        self.user_goal = ""
        self.retrieved_data = {}
        self.parsed_values = {}
