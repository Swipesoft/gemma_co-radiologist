"""
athena_dda.session — Session State Manager
════════════════════════════════════════════
Persistent in-memory store for multi-turn agent sessions.
"""

import uuid
from datetime import datetime
from typing import Any, Dict, List


class SessionStore:
    """Persistent in-memory store for agent session state.
    Keyed by session_id, stores all context from prior tool executions
    so users can ask follow-up questions or request revisions.
    """

    def __init__(self):
        self._sessions: Dict[str, Dict[str, Any]] = {}

    def create_session(self, session_id: str = None) -> str:
        sid = session_id or f"RAD-{uuid.uuid4().hex[:8].upper()}"
        if sid not in self._sessions:
            self._sessions[sid] = {
                "created_at": datetime.now().isoformat(),
                "history": [],
                "context": {},
                "reports": [],
                "images": {},
            }
        return sid

    def save_context(self, session_id: str, key: str, value: Any):
        if session_id not in self._sessions:
            self.create_session(session_id)
        self._sessions[session_id]["context"][key] = value

    def save_run_result(self, session_id: str, query: str, result: Dict):
        if session_id not in self._sessions:
            self.create_session(session_id)
        self._sessions[session_id]["history"].append({
            "query": query,
            "result_summary": result.get("response", "")[:500],
            "tools_executed": result.get("tools_executed", []),
            "timestamp": datetime.now().isoformat(),
        })
        for key in result.get("context_keys", []):
            if key not in self._sessions[session_id]["context"]:
                self._sessions[session_id]["context"][key] = "available"

    def get_context(self, session_id: str) -> Dict:
        return self._sessions.get(session_id, {}).get("context", {})

    def get_full_session(self, session_id: str) -> Dict:
        return self._sessions.get(session_id, {})

    def get_session_summary(self, session_id: str) -> str:
        """Returns a text summary of the session for LLM consumption."""
        session = self._sessions.get(session_id)
        if not session:
            return "No prior session data available."

        parts = [f"Session ID: {session_id}"]
        parts.append(f"Created: {session.get('created_at', 'unknown')}")

        ctx = session.get("context", {})
        if ctx:
            parts.append("\nAccumulated Context:")
            for k, v in ctx.items():
                val_str = str(v)[:200] if not k.startswith("_raw_") else "[raw data]"
                parts.append(f"  {k}: {val_str}")

        history = session.get("history", [])
        if history:
            parts.append(f"\nConversation History ({len(history)} turns):")
            for h in history[-5:]:
                parts.append(f"  Q: {h['query'][:100]}")
                parts.append(f"  A: {h['result_summary'][:150]}")
                parts.append(f"  Tools: {h['tools_executed']}")

        return "\n".join(parts)

    def list_sessions(self) -> List[str]:
        return list(self._sessions.keys())


# Global session store singleton
SESSION_STORE = SessionStore()
