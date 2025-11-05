# agents.py
import json
import uuid
from datetime import datetime
from typing import Any, Dict, List, Tuple

from django.conf import settings
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool

# Pull only what we need from utils
from .utils import analyze_query, hybrid_search, _norm  # reuse your normalization


# ──────────────────────────────────────────────────────────────────────────────
# Helpers to serialize history (unchanged, with tiny polish)
# ──────────────────────────────────────────────────────────────────────────────

def _content_to_str(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, dict) and p.get("type") == "text":
                parts.append(p.get("text", ""))
        return "\n".join(parts).strip()
    try:
        return json.dumps(content, ensure_ascii=False)
    except Exception:
        return str(content)

def _role_of(msg: BaseMessage) -> str:
    t = getattr(msg, "type", "").lower()
    if t == "human": return "user"
    if t == "ai":    return "assistant"
    if t == "tool":  return "tool"
    if t == "system":return "system"
    return t or msg.__class__.__name__.replace("Message", "").lower()

def serialize_history(messages: List[BaseMessage]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for m in messages:
        role = _role_of(m)
        content = _content_to_str(m.content)
        # Drop blank assistant stubs (tool-call scaffolding)
        if role == "assistant" and not (content or "").strip():
            continue
        # If assistant content is JSON {"response": "..."} keep only response
        if role == "assistant":
            try:
                payload = json.loads(content)
                if isinstance(payload, dict) and "response" in payload:
                    content = payload["response"]
            except Exception:
                pass
        out.append({"role": role, "content": content})
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Base LLM
# ──────────────────────────────────────────────────────────────────────────────

llm = ChatOpenAI(
    model="gpt-4.1",
    temperature=0,
    api_key=settings.OPENAI_API_KEY,
    timeout=60.0,
    max_retries=5,
)


# ──────────────────────────────────────────────────────────────────────────────
# Tool: car_spec_rag
# Returns ONLY the primary (asked) model’s docs + urls + page_ids.
# Also performs a strict post-guard so off-model chunks are dropped.
# ──────────────────────────────────────────────────────────────────────────────

def _norm_set(vals: List[str]) -> List[str]:
    """Return a de-duped list of normalized variants (keeps both en/fa digits)."""
    seen = {}
    for v in vals:
        for nv in _norm(v):
            if nv not in seen:
                seen[nv] = True
    return list(seen.keys())

def _choose_primary_model(query: str, analysis: Dict[str, Dict[str, List[str]]], docs_by_model: Dict[str, List[Tuple]]) -> str | None:
    q = next(iter(_norm(query.lower())))
    # 1) If any alias appears in the raw query, pick that model
    for m, info in analysis.items():
        aliases = (info.get("aliases") or []) + [m]
        for a in _norm_set(aliases):
            if a and a in q:
                return m
    # 2) Fallback: the model with the most docs
    if docs_by_model:
        return max(docs_by_model.items(), key=lambda kv: len(kv[1]))[0]
    return None

def _passes_model_guard(tup: Tuple, aliases_norm: List[str]) -> bool:
    """
    Accept a chunk only if title/content/car_model meta contains one of the normalized aliases.
    Tuple layout from hybrid_search:
      0 id, 1 title, 2 content, 3 fused, 4 imgs, 5 model_id, 6 url, 7 source_name, 8 page_id, 9 car_model(meta)
    """
    title = (tup[1] or "").strip()
    content = (tup[2] or "").strip()
    car_model_meta = (tup[9] if len(tup) >= 10 else "") or ""
    hay_raw = f"{title}\n{content}\n{car_model_meta}"
    hay = next(iter(_norm(hay_raw)))
    return any(a in hay for a in aliases_norm if a)

@tool
def car_spec_rag(query: str) -> str:
    """Returns relevant documents/specs for the *asked* car model only.
    If nothing is confidently found, returns empty arrays.
    Not for prices."""
    analysis = analyze_query(query) or {}
    docs_by_model = hybrid_search(analysis, top_k=20)

    primary = _choose_primary_model(query, analysis, docs_by_model)
    chosen = docs_by_model.get(primary, []) if primary else []

    # Build alias set for strict guard
    aliases = []
    if primary and analysis.get(primary):
        aliases = (analysis[primary].get("aliases") or []) + [primary]
    aliases_norm = _norm_set(aliases)

    # Strict post-filter: drop any chunk that doesn't look like the primary model
    if aliases_norm:
        guarded = [t for t in chosen if _passes_model_guard(t, aliases_norm)]
    else:
        guarded = chosen

    # Stable de-dupe, preserving ranking
    flat_urls: List[str] = []
    seen_url = set()
    for t in guarded:
        u = t[6]
        if u and u not in seen_url:
            seen_url.add(u)
            flat_urls.append(u)

    page_ids: List[int] = []
    seen_pid = set()
    for t in guarded:
        pid = t[8] if len(t) >= 9 else None
        if isinstance(pid, int) and pid not in seen_pid:
            seen_pid.add(pid)
            page_ids.append(pid)

    payload = {
        "primary_model": primary,                 # str | None
        "model_aliases": list(dict.fromkeys(aliases)) if aliases else [],
        "docs": {primary: guarded} if primary else {},
        "urls": flat_urls,
        "page_ids": page_ids,
    }
    return json.dumps(payload, ensure_ascii=False)


@tool
def car_price(car: str) -> str:
    """Return price for a given car (mocked)."""
    return "Price for BMW X5 in the US starts at $54 200."


tools = [car_spec_rag, car_price]


# ──────────────────────────────────────────────────────────────────────────────
# React agent: force pure-JSON final message (no extra text)
# The view will read "response" only; JSON never leaks to UI.
# ──────────────────────────────────────────────────────────────────────────────

agent = create_react_agent(
    model=llm,
    tools=tools,
    prompt=(
        "You are a helpful automotive assistant. You ONLY answer questions about car specifications and comparisons.\n"
        "• Do not answer non-car questions.\n"
        "• Do not discuss NSFW/sexual/ideological/political topics (even about cars).\n"
        "• If asked for recommendations, say you only answer specs/comparisons.\n"
        "• You may call at most ONE tool.\n"
        "• For spec/comparison questions, you MUST call `car_spec_rag` first.\n"
        "• `car_spec_rag` returns JSON with `primary_model`, `docs`, `urls`, `page_ids`.\n"
        "After the tool returns, reply with EXACTLY this JSON and nothing else:\n"
        "{\n"
        '  "response": "<short Farsi answer based ONLY on tool docs; if nothing relevant, say: متأسفانه اطلاعات درخواستی در مستندات ارائه‌شده موجود نیست.>",\n'
        '  "urls": <paste the `urls` array from car_spec_rag exactly>,\n'
        '  "page_ids": <paste the `page_ids` array from car_spec_rag exactly>\n"'
        "}\n"
    ),
)


# ──────────────────────────────────────────────────────────────────────────────
# Graph + one-turn runner
# ──────────────────────────────────────────────────────────────────────────────

graph = StateGraph(MessagesState)

def _chatbot(state: MessagesState):
    response_state = agent.invoke({"messages": state["messages"]})
    return {"messages": response_state["messages"]}

graph.add_node("chatbot", _chatbot)
graph.add_edge(START, "chatbot")
graph.add_edge("chatbot", END)

_compiled_graph = graph.compile(checkpointer=MemorySaver())

def run_turn(user_text: str, *, thread_id: str | None = None) -> tuple[str, List[str], str, List[dict]]:
    thread_id = thread_id or str(uuid.uuid4())
    cfg = {"configurable": {"thread_id": thread_id}}

    result = _compiled_graph.invoke({"messages": [{"role": "user", "content": user_text}]}, config=cfg)
    full_history = serialize_history(result["messages"])

    raw = result["messages"][-1].content  # final assistant content
    urls: List[str] = []
    try:
        parsed = json.loads(raw)
        answer = parsed.get("response", "").strip() or raw
        urls = parsed.get("urls", []) or []
    except Exception:
        # In the unlikely case the model violates the JSON rule
        answer = raw

    return answer, urls, thread_id, full_history


def get_thread_messages(thread_id: str):
    cfg = {"configurable": {"thread_id": thread_id}}
    snapshot = _compiled_graph.get_state(cfg)
    if snapshot is None:
        return []
    return snapshot.values.get("messages", [])
