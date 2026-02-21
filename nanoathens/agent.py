"""
athena_dda.agent — Declarative DataFlow Agent
═══════════════════════════════════════════════
The main orchestrator that ties everything together:
  1. LLM extracts values only (not a planner)
  2. DataFlowEngine resolves exact execution path
  3. Returns explicit null_plan if registry gap detected
  4. Minimum LLM calls: 2 (extraction + goal resolution)
  5. Deterministic: same query always produces same plan
"""

import time
from typing import Callable, Dict, List, Optional

from .core import ToolRegistry
from .context import ContextBank, LLMValueExtractor
from .engine import DataFlowEngine
from .resolver import GoalKeyResolver
from .filler import GroundedArgumentFiller


class DeclarativeDataFlowAgent:
    """DDA Orchestrator — schema-resolved, data-driven execution."""

    # Original default (oncology) — preserved for backward compat
    _DEFAULT_SYSTEM_PROMPT = (
        "You are a precision oncology AI assistant. "
        "Provide a structured response."
    )

    def __init__(
        self,
        registry: ToolRegistry,
        reasoning_caller: Callable,
        verbose: bool = True,
        system_prompt: str = None,
        goal_few_shot_examples: str = None,
    ):
        self.registry = registry
        self.reasoning_llm = reasoning_caller
        self.verbose = verbose
        self.system_prompt = system_prompt or self._DEFAULT_SYSTEM_PROMPT
        self.extractor = LLMValueExtractor(reasoning_caller, verbose=False)
        self.engine = DataFlowEngine(registry, verbose=verbose)
        self.goal_resolver = GoalKeyResolver(
            reasoning_caller,
            self.engine,
            verbose=verbose,
            few_shot_examples=goal_few_shot_examples,
        )
        self.filler = GroundedArgumentFiller(reasoning_caller, verbose=False)

    def _log(self, msg, indent=0):
        if self.verbose:
            print(f"{'  ' * indent}{msg}")

    async def run(self, user_query: str, target_key: str = None) -> Dict:
        t0 = time.time()
        self._log(f'\n{"=" * 60}')
        self._log(f"[DDA] {user_query[:80]}")
        self._log(f'{"=" * 60}')

        # Step 1: Variable Extraction
        self._log("[Step 1] Variable Extraction")
        known_keys = self.registry.get_all_context_keys()
        context = ContextBank(self.extractor, known_keys, self.registry)
        context.set_goal(user_query)

        populated_keys = {
            k
            for k, v in context.parsed_values.items()
            if v is not None and v != "" and v != "None" and v != "unknown"
        }
        all_tool_produced = set(self.engine._producers.keys())
        source_keys = {k for k in known_keys if k not in all_tool_produced}
        available_keys = populated_keys | source_keys
        self._log(
            f"  {len(populated_keys)} populated: {sorted(populated_keys)}", 1
        )
        self._log(
            f"  {len(source_keys)} source keys (leaf inputs): {len(source_keys)}", 1
        )
        self._log(f"  {len(available_keys)} total available", 1)

        # Step 2: Autonomous Goal Key Identification
        self._log("[Step 2] Goal Key Identification")
        if target_key is None:
            target_key = self.goal_resolver.resolve(user_query)
        if target_key is None:
            return {
                "response": "Could not identify target goal from query.",
                "tools_executed": [],
                "latency_s": round(time.time() - t0, 2),
                "execution_plan": None,
                "target_key": None,
                "agent": "DDA",
                "status": "goal_resolution_failed",
            }
        self._log(f"  Target: '{target_key}'", 1)
        available_keys.discard(target_key)

        # Step 3: DataFlow Graph Resolution
        self._log("[Step 3] DataFlow Graph Resolution")
        plan = self.engine.resolve_execution_plan(available_keys, target_key)

        if plan is None:
            self._log(f"  NULL PLAN: no path to '{target_key}'")
            return {
                "response": (
                    f"Registry gap: no execution path to produce "
                    f"'{target_key}' from available context "
                    f"{sorted(available_keys)}. "
                    f"Missing tool or broken dependency chain."
                ),
                "tools_executed": [],
                "latency_s": round(time.time() - t0, 2),
                "execution_plan": None,
                "target_key": target_key,
                "agent": "DDA",
                "status": "null_plan",
            }

        self._log(f"  Plan ({len(plan)} tools): {plan}", 1)

        # Step 4: Deterministic Execution
        self._log("[Step 4] Execution")
        tools_executed = []
        failed_tools = []

        for i, tool_name in enumerate(plan):
            tool = self.registry.get_tool(tool_name)
            self._log(f"  [{i + 1}/{len(plan)}] {tool_name}", 1)
            args = self.filler.fill_arguments(tool, context)
            if args is None:
                self._log("    SKIP: cannot fill args", 2)
                failed_tools.append(tool_name)
                continue
            try:
                result = await self.registry.call(tool_name, args)
                context.add_tool_result(
                    tool_name, str(result), tool.output_keys
                )
                tools_executed.append(tool_name)
                self._log(f"    OK: {str(result)[:100]}", 2)
            except Exception as e:
                self._log(f"    ERR: {e}", 2)
                failed_tools.append(tool_name)

        # Step 5: Synthesis
        self._log("[Step 5] Synthesis")
        results_text = "\n".join(
            f"- {t}: {r[:250]}" for t, r in context.retrieved_data.items()
        )
        synth = (
            f"{self.system_prompt}\n\n"
            f"QUESTION: {user_query}\n\n"
            f"CLINICAL DATA:\n{results_text or 'No data collected.'}\n\n"
            f"RESPONSE:"
        )
        try:
            response = self.reasoning_llm(
                messages=[
                    {"role": "user", "content": [{"type": "text", "text": synth}]}
                ],
                max_new_tokens=512,
                temperature=0.2,
            )
        except Exception:
            response = f"Collected data:\n{results_text}"

        elapsed = round(time.time() - t0, 2)
        self._log(f"DDA complete in {elapsed}s | Executed: {tools_executed}")
        return {
            "response": response.strip(),
            "tools_executed": tools_executed,
            "failed_tools": failed_tools,
            "execution_plan": plan,
            "target_key": target_key,
            "latency_s": elapsed,
            "agent": "DDA",
            "status": "success" if not failed_tools else "partial",
            "context_keys": list(context.parsed_values.keys()),
        }


# Alias for user convenience
ConfigurableOrchestrator = DeclarativeDataFlowAgent
