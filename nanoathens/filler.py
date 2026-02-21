"""
athena_dda.filler — Grounded Argument Filler
═════════════════════════════════════════════
Fills tool arguments from ContextBank, with optional LLM fallback.
"""

import json
from typing import Callable, Dict, Optional

from .core import ToolSchema
from .context import ContextBank


class GroundedArgumentFiller:
    def __init__(self, llm_caller: Callable = None, verbose=True):
        self.llm = llm_caller
        self.verbose = verbose

    def _log(self, msg):
        if self.verbose:
            print(f"[Filler] {msg}")

    def fill_arguments(
        self, tool: ToolSchema, context: ContextBank
    ) -> Optional[Dict]:
        arguments = {}
        unfilled = []
        grounded = 0

        for param in tool.parameters:
            ctx_key = tool.arg_sources.get(param, param)
            val = context.get_value(ctx_key)
            if val is not None:
                arguments[param] = val
                if param in tool.required:
                    grounded += 1
            elif param in tool.required:
                unfilled.append(param)

        if (
            grounded == 0
            and len(tool.required) >= 2
            and len(unfilled) == len(tool.required)
            and not self.llm
        ):
            self._log(
                f"SKIP {tool.name}: zero required params grounded "
                f"and no LLM available"
            )
            return None

        if unfilled and self.llm:
            param_specs = {p: tool.parameters.get(p, "?") for p in unfilled}
            prompt = (
                f"Fill these parameters with ACTUAL VALUES from context.\n"
                f"FUNCTION: {tool.name}\n"
                f"CONTEXT:\n{context.get_grounding_context()}\n"
                f"Parameters (return real values, not type descriptions):\n"
                f"{json.dumps(param_specs)}\nJSON only:"
            )
            messages = [
                {"role": "user", "content": [{"type": "text", "text": prompt}]}
            ]
            raw = self.llm(messages=messages, max_new_tokens=256, temperature=0.0)
            s = raw.find("{")
            e = raw.rfind("}")
            if s != -1 and e > s:
                try:
                    for k, v in json.loads(raw[s : e + 1]).items():
                        sv = str(v)
                        if (
                            sv.lower()
                            not in ("null", "none", "unknown", "n/a", "")
                            and "|" not in sv
                        ):
                            arguments[k] = v
                            unfilled = [x for x in unfilled if x != k]
                except Exception:
                    pass

        missing = [p for p in tool.required if p not in arguments]
        if missing:
            self._log(f"FAIL {tool.name}: missing {missing}")
            return None
        return arguments
