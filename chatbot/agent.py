# agent.py
import json, uuid
from typing import Any, Dict, List
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, START, MessagesState, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from django.conf import settings

from .utils import analyze_query, hybrid_search

# ───────── Helpers ─────────
def _content_to_str(content: Any) -> str:
    if isinstance(content, str): return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, dict) and p.get("type") == "text":
                parts.append(p.get("text",""))
        return "\n".join(parts).strip()
    return str(content)

def _role_of(msg: BaseMessage) -> str:
    t = getattr(msg, "type", "").lower()
    if t == "human": return "user"
    if t == "ai":    return "assistant"
    if t == "tool":  return "tool"
    if t == "system":return "system"
    return t or msg.__class__.__name__.replace("Message","").lower()

def serialize_history(messages: List[BaseMessage]) -> List[Dict[str, str]]:
    out = []
    for m in messages:
        role = _role_of(m)
        content = _content_to_str(m.content)
        if role == "assistant":
            if not (content or "").strip(): continue
            try:
                payload = json.loads(content)
                if isinstance(payload, dict) and "response" in payload:
                    content = payload["response"]
            except Exception:
                pass
        out.append({"role": role, "content": content})
    return out

# ───────── Base LLM (ثابت) ─────────
llm = ChatOpenAI(
    model="gpt-4.1",
    temperature=0,
    api_key=settings.OPENAI_API_KEY,
    timeout=60.0,
    max_retries=5,
)

# ───────── Tools ─────────
from langchain_core.tools import tool

@tool
def car_spec_rag(query: str) -> str:
    """Returns relevant documents/specs for specific car(s). Not for price."""
    analysis = analyze_query(query)                  # JSON-mode + cache
    docs = hybrid_search(analysis, top_k=20)         # Hybrid (embed+bm25)

    if not any(docs.values()):
        return json.dumps({"docs": {}, "urls": [], "page_ids": []}, ensure_ascii=False)

    flat_urls = list({
        tup[6] for model_docs in docs.values() for tup in model_docs if tup[6]
    })
    page_ids = sorted({
        tup[8] for model_docs in docs.values() for tup in model_docs if len(tup) >= 9 and tup[8] is not None
    })
    return json.dumps({"docs": docs, "urls": flat_urls, "page_ids": page_ids}, ensure_ascii=False)

@tool
def car_price(car: str) -> str:
    """Return price for a given car (mocked)."""
    return "Price for BMW X5 in the US starts at $54 200."

tools = [car_spec_rag, car_price]

# ───────── Agent (همان create_react_agent) ─────────
agent = create_react_agent(
    model=llm,
    tools=tools,
    prompt=(
        "You are a helpful automotive assistant.\n\n"
        "• You may call at most one tool per question.\n"
        "• `car_spec_rag` returns JSON with keys: `docs`, `urls`, `page_ids`.\n"
        "• Answer ONLY from tool results; if nothing relevant, say you don't know.\n"
        "• Finish with EXACTLY this JSON and nothing else:\n"
        "{\n"
        '  "response": "<پاسخ شما به فارسی>",\n'
        '  "urls": [<لیست یکتای URLهایی که برای پاسخ استفاده شد>]\n'
        "}"
    ),
)

# ───────── Graph ─────────
graph = StateGraph(MessagesState)

def _chatbot(state: MessagesState):
    response_state = agent.invoke({"messages": state["messages"]})
    return {"messages": response_state["messages"]}

graph.add_node("chatbot", _chatbot)
graph.add_edge(START, "chatbot")
graph.add_edge("chatbot", END)

_compiled_graph = graph.compile(checkpointer=MemorySaver())

# ───────── Turn Helpers ─────────
def run_turn(user_text: str, *, thread_id: str | None = None):
    thread_id = thread_id or str(uuid.uuid4())
    cfg = {"configurable": {"thread_id": thread_id}}
    result = _compiled_graph.invoke({"messages": [{"role": "user", "content": user_text}]}, config=cfg)

    full_history = serialize_history(result["messages"])
    raw = result["messages"][-1].content
    urls = []
    try:
        parsed = json.loads(raw)
        answer = parsed.get("response", raw)
        urls   = parsed.get("urls", []) or []
    except Exception:
        answer = raw
    return answer, urls, thread_id, full_history

def get_thread_messages(thread_id: str):
    cfg = {"configurable": {"thread_id": thread_id}}
    snapshot = _compiled_graph.get_state(cfg)
    if snapshot is None: return []
    return snapshot.values.get("messages", [])
