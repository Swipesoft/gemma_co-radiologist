"""
athena_dda.engine — DataFlow Engine
════════════════════════════════════
Builds a static dependency DAG from the collision-free registry.
Resolves minimal execution paths using backward depth-first search.
"""

from collections import defaultdict
from typing import Dict, List, Optional, Set

from .core import ToolRegistry


class DataFlowEngine:
    """Builds a static dependency DAG from the collision-free registry.
    Resolves minimal execution paths using backward depth-first search.
    The collision-free property (output_keys disjoint from arg_sources)
    ensures no self-loops — backward DFS always terminates.
    """

    def __init__(self, registry: ToolRegistry, verbose: bool = True):
        self.registry = registry
        self.verbose = verbose
        self._producers: Dict[str, List[str]] = defaultdict(list)
        self._consumers: Dict[str, List[str]] = defaultdict(list)
        self._build_static_graph()

    def _build_static_graph(self):
        for tool_name, tool in self.registry._tools.items():
            for key in tool.output_keys:
                self._producers[key].append(tool_name)
            for _, ctx_key in tool.arg_sources.items():
                self._consumers[ctx_key].append(tool_name)
        if self.verbose:
            print(
                f"[DataFlowEngine] {len(self._producers)} producer keys, "
                f"{len(self.registry._tools)} tools registered"
            )

    def get_all_output_keys(self) -> List[str]:
        return list(self._producers.keys())

    def resolve_execution_plan(
        self, available_keys: Set[str], target_key: str, max_depth: int = 15
    ) -> Optional[List[str]]:
        visited_in_path: Set[str] = set()

        def _resolve_key(needed_key, depth, running_avail):
            if needed_key in running_avail:
                return []
            if depth > max_depth:
                return None
            if needed_key in visited_in_path:
                return None
            visited_in_path.add(needed_key)

            producers = self._producers.get(needed_key, [])
            if not producers:
                visited_in_path.discard(needed_key)
                return None

            for tool_name in producers:
                tool = self.registry.get_tool(tool_name)
                needed_inputs = set(tool.arg_sources.values())
                sub_plan = []
                feasible = True
                temp_avail = set(running_avail)

                for input_key in needed_inputs:
                    if input_key in temp_avail:
                        continue
                    sub = _resolve_key(input_key, depth + 1, temp_avail)
                    if sub is None:
                        feasible = False
                        break
                    for t in sub:
                        temp_avail.update(
                            self.registry.get_tool(t).output_keys.keys()
                        )
                    sub_plan.extend(sub)

                if feasible:
                    seen = set()
                    ordered = []
                    for t in sub_plan + [tool_name]:
                        if t not in seen:
                            seen.add(t)
                            ordered.append(t)
                    visited_in_path.discard(needed_key)
                    return ordered

            visited_in_path.discard(needed_key)
            return None

        return _resolve_key(target_key, 0, set(available_keys))

    def visualize_graph(self, highlight: List[str] = None) -> str:
        hl = set(highlight or [])
        lines = ["\nDataFlow Graph (tool -> inputs / outputs):"]
        for name, tool in self.registry._tools.items():
            marker = " ** ON PATH **" if name in hl else ""
            lines.append(f"  {name}{marker}")
            lines.append(f"    IN : {list(tool.arg_sources.values())}")
            lines.append(f"    OUT: {list(tool.output_keys.keys())}")
        return "\n".join(lines)

    def get_graph_stats(self) -> Dict:
        all_produced = set(self._producers.keys())
        source_tools = [
            name
            for name, tool in self.registry._tools.items()
            if not set(tool.arg_sources.values()).intersection(all_produced)
        ]
        return {
            "total_tools": len(self.registry._tools),
            "unique_producer_keys": len(all_produced),
            "source_tools": source_tools,
        }
