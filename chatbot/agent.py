# agents.py
import json
import uuid
from typing import Any, Dict, List, Tuple

from django.conf import settings
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool

from .utils import analyze_query, hybrid_search, _norm  # reuse your normalizer


# ——— history helpers (unchanged except we drop empty assistant stubs) ———

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
    if t == "system": return "system"
    return t or msg.__class__.__name__.replace("Message", "").lower()


def serialize_history(messages: List[BaseMessage]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for m in messages:
        role = _role_of(m)
        content = _content_to_str(m.content)
        if role == "assistant" and not (content or "").strip():
            continue
        if role == "assistant":
            try:
                payload = json.loads(content)
                if isinstance(payload, dict) and "response" in payload:
                    content = payload["response"]
            except Exception:
                pass
        out.append({"role": role, "content": content})
    return out


# ——— base LLM ———

llm = ChatOpenAI(
    model="gpt-4.1",
    temperature=0,
    api_key=settings.OPENAI_API_KEY,
    timeout=60.0,
    max_retries=5,
)


# ——— strict equality guard helpers ———

def _norm_set(vals: List[str]) -> set[str]:
    """All normalized variants across values (fa/en digits)."""
    out: set[str] = set()
    for v in vals or []:
        out |= _norm(v)
    return out


def _eq_norm(val: str, allowed: set[str]) -> bool:
    """True if normalized(val) equals one of the allowed normalized aliases."""
    if not val:
        return False
    return bool(_norm(val) & allowed)


# ——— TOOL: ONLY return docs/attachments whose metadata car_model equals the asked model ———

@tool
def car_spec_rag(query: str) -> str:
    """
    Returns relevant documents/specs for the *asked* car model only.
    Attachments (urls/page_ids) and docs are STRICTLY filtered by metadata equality on car_model.
    If nothing matches strictly, returns empty arrays (no wrong images).
    Not for prices.
    """
    analysis = analyze_query(query) or {}
    docs_by_model = hybrid_search(analysis, top_k=20)

    # choose primary by alias presence in query; fallback to largest bucket
    q_norm = next(iter(_norm(query.lower())))
    primary = None
    for m, info in analysis.items():
        aliases = (info.get("aliases") or []) + [m]
        if any(a and a in q_norm for a in _norm_set(aliases)):
            primary = m
            break
    if primary is None and docs_by_model:
        primary = max(docs_by_model.items(), key=lambda kv: len(kv[1]))[0]

    chosen = docs_by_model.get(primary, []) if primary else []

    # strict guard by metadata equality on car_model
    aliases = ((analysis.get(primary) or {}).get("aliases") or []) + ([primary] if primary else [])
    allowed = _norm_set(aliases)

    # tuple layout from hybrid_search: ... 9 = car_model_meta
    strict_docs = [t for t in chosen if _eq_norm((t[9] if len(t) >= 10 else ""), allowed)]

    # stable, ranked, de-duped attachments from strict_docs only
    urls: List[str] = []
    seen_u = set()
    pids: List[int] = []
    seen_p = set()
    for t in strict_docs:
        u = t[6]
        if u and u not in seen_u:
            seen_u.add(u)
            urls.append(u)
        pid = t[8] if len(t) >= 9 else None
        if isinstance(pid, int) and pid not in seen_p:
            seen_p.add(pid)
            pids.append(pid)

    payload = {
        "primary_model": primary,
        "docs": {primary: strict_docs} if primary else {},
        "urls": urls,
        "page_ids": pids,
    }
    return json.dumps(payload, ensure_ascii=False)


@tool
def car_price(car: str) -> str:
    """Return price for a given car (mocked)."""
    return "Price for BMW X5 in the US starts at $54 200."


tools = [car_spec_rag, car_price]

# ——— agent: respond with JSON only (fixed template) ———

agent = create_react_agent(
    model=llm,
    tools=tools,
    prompt=(
        "You are a helpful automotive assistant. You ONLY answer questions about car specifications and comparisons.\n"
        "• Always call `car_spec_rag` before answering specs/comparisons and base your answer ONLY on its docs.\n"
        "• You may call at most ONE tool, and you must return EXACTLY this JSON (no prose, no extra text):\n"
        "{\n"
        '  "response": "<short Farsi answer based ONLY on tool docs; if none, write: متأسفانه اطلاعات درخواستی در مستندات ارائه‌شده موجود نیست.>",\n'
        '  "urls": <paste the `urls` array from car_spec_rag exactly>,\n'
        '  "page_ids": <paste the `page_ids` array from car_spec_rag exactly>\n'
        "}\n"
    ),
)

# ——— graph + one-turn runner (unchanged) ———

graph = StateGraph(MessagesState)


def _chatbot(state: MessagesState):
    response_state = agent.invoke({"messages": state["messages"]})
    return {"messages": response_state["messages"]}


graph.add_node("chatbot", _chatbot)
graph.add_edge(START, "chatbot")
graph.add_edge("chatbot", END)

_compiled_graph = graph.compile(checkpointer=MemorySaver())


def run_turn(user_text: str, *, thread_id: str | None = None):
    thread_id = thread_id or str(uuid.uuid4())
    cfg = {"configurable": {"thread_id": thread_id}}
    result = _compiled_graph.invoke({"messages": [{"role": "user", "content": user_text}]}, config=cfg)
    full_history = serialize_history(result["messages"])

    raw = result["messages"][-1].content
    urls: List[str] = []
    try:
        parsed = json.loads(raw)
        answer = parsed.get("response", "").strip() or raw
        urls = parsed.get("urls", []) or []
    except Exception:
        answer = raw

    return answer, urls, thread_id, full_history


def get_thread_messages(thread_id: str):
    cfg = {"configurable": {"thread_id": thread_id}}
    snapshot = _compiled_graph.get_state(cfg)
    if snapshot is None:
        return []
    return snapshot.values.get("messages", [])
