"""
athena_dda.resolver — Goal Key Resolver
════════════════════════════════════════
Maps a user query to the single target output key in the DAG.
"""

from typing import Callable, Optional

from .engine import DataFlowEngine


class GoalKeyResolver:
    """Resolves user queries to target context keys using LLM + keyword fallback."""

    # Original oncology few-shot (preserved as default for backward compat)
    _DEFAULT_FEW_SHOT = (
        "EXAMPLES:\n"
        "Query: Is the patient fit for surgery? -> surgical_readiness_status\n"
        "Query: What is the tumor volume? -> tumor_volume_cc\n"
        "Query: What anesthesia is needed? -> anesthesia_protocol\n"
        "Query: How is recovery going? -> recovery_trend\n"
        "Query: What is the BMI category? -> bmi_category\n"
        "Query: What is the metabolic status? -> metabolic_status\n"
        "Query: What is the cardiac risk? -> cardiac_risk_score\n"
        "Query: Finalize the medical record -> record_archive_number\n"
    )

    def __init__(
        self,
        llm_caller: Callable,
        engine: DataFlowEngine,
        verbose=True,
        few_shot_examples: str = None,
    ):
        self.llm = llm_caller
        self.engine = engine
        self.verbose = verbose
        self._all_keys = engine.get_all_output_keys()
        self.FEW_SHOT = few_shot_examples or self._DEFAULT_FEW_SHOT

    def _log(self, msg):
        if self.verbose:
            print(f"[GoalResolver] {msg}")

    def resolve(self, user_query: str) -> Optional[str]:
        keys_list = "\n".join(f"  - {k}" for k in self._all_keys)
        prompt = (
            f"{self.FEW_SHOT}\n"
            f"AVAILABLE KEYS:\n{keys_list}\n\n"
            f"USER QUERY: {user_query}\n\n"
            f"Which single key best represents the FINAL GOAL? "
            f"Reply with ONLY the key name:"
        )
        messages = [
            {"role": "user", "content": [{"type": "text", "text": prompt}]}
        ]
        try:
            raw = self.llm(
                messages=messages, max_new_tokens=40, temperature=0.0
            ).strip()
            for st, et in [
                ("<unused94>", "<unused95>"),
                ("<think>", "</think>"),
            ]:
                while st in raw:
                    s = raw.find(st)
                    e = raw.find(et)
                    raw = raw[:s] + (raw[e + len(et):] if e > s else "")
            raw = (
                raw.strip()
                .lower()
                .replace(" ", "_")
                .replace("-", "_")
                .split("\n")[0]
                .strip()
            )
            if raw in self._all_keys:
                self._log(f"LLM resolved: {raw!r}")
                return raw
            for k in self._all_keys:
                if raw in k or k in raw:
                    self._log(f"Fuzzy: {raw!r} -> {k!r}")
                    return k
        except Exception as e:
            self._log(f"LLM error: {e}")

        # Keyword fallback
        query_lower = user_query.lower()
        best_key, best_score = None, 0
        for k in self._all_keys:
            score = sum(
                1 for w in k.replace("_", " ").split() if w in query_lower
            )
            if score > best_score:
                best_score, best_key = score, k
        if best_key:
            self._log(
                f"Keyword fallback -> {best_key!r} (score={best_score})"
            )
        return best_key
