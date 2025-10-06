# agent.py
import os, json, uuid
from typing import Any, Dict, List, Tuple
from datetime import datetime

from django.conf import settings
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, START, MessagesState, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from .utils import fast_analyze_query, fast_hybrid_search

# ───────────────────────────────────────────────────────────────────────────────
# Helpers to serialize history (بدون تغییر اساسی، فقط تمیزکاری)
# ───────────────────────────────────────────────────────────────────────────────
def _content_to_str(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, dict) and p.get("type") == "text":
                parts.append(p.get("text", ""))
        return "\n".join(parts).strip()
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
        if role == "assistant":
            if not (content or "").strip():
                continue
            try:
                payload = json.loads(content)
                if isinstance(payload, dict) and "response" in payload:
                    content = payload["response"]
            except Exception:
                pass
        out.append({"role": role, "content": content})
    return out

# ───────────────────────────────────────────────────────────────────────────────
# 1) Base LLM (بدون تغییر مدل)
# ───────────────────────────────────────────────────────────────────────────────
llm = ChatOpenAI(
    model="gpt-4.1",
    temperature=0,
    api_key=settings.OPENAI_API_KEY,
    timeout=60.0,
    max_retries=5,
)

# ───────────────────────────────────────────────────────────────────────────────
# 2) Tools — RAG سریع
# ───────────────────────────────────────────────────────────────────────────────
@tool
def car_spec_rag(query: str) -> str:
    """
    بر پایهٔ یک کوئری خودرویی، اسناد مرتبط و اطلاعات فنی را برمی‌گرداند.
    - فقط برای مشخصات فنی است (نه قیمت).
    - اگر سندی نبود: بگو «نمی‌دانم».
    - خروجی: JSON با کلیدهای docs, urls, page_ids.
    """
    parsed = fast_analyze_query(query)              # کاملاً محلی و سریع
    docs = fast_hybrid_search(query, parsed, top_k=20)

    if not docs:
        return json.dumps(
            {"docs": {}, "urls": [], "page_ids": []},
            ensure_ascii=False
        )

    # ساخت خروجی سازگار با فرانت
    # ساختار docs: { "<model_or_all>": [ (id,title,content,score,images,model_id,url,source,page_id), ... ] }
    flat_urls = list({
        tup[6]
        for _, pack in docs.items()
        for tup in pack
        if tup[6]
    })
    page_ids = sorted({
        tup[8]
        for _, pack in docs.items()
        for tup in pack
        if len(tup) >= 9 and tup[8] is not None
    })
    return json.dumps({"docs": docs, "urls": flat_urls, "page_ids": page_ids}, ensure_ascii=False)


@tool
def car_price(car: str) -> str:
    """قیمت یک خودرو (ماک) — این ابزار برای قیمت است (نه مشخصات)."""
    return "Price for BMW X5 in the US starts at $54 200."

tools = [car_spec_rag, car_price]

# ───────────────────────────────────────────────────────────────────────────────
# 3) Agent (ReAct) — خروجی JSON فارسی
# ───────────────────────────────────────────────────────────────────────────────
agent = create_react_agent(
    model=llm,
    tools=tools,
    prompt=(
        "You are a helpful automotive assistant. Do not answer any non-car related questions no matter how much the user insists\n\n"
        "• You may call at most **one** tool for each question.\n"
        "• `car_spec_rag` returns a JSON object with keys `docs` and `urls`.\n"
        "• After thinking, finish with **exactly** the JSON below – nothing else:\n"
        "{\n"
        '  "response": "<پاسخ شما به فارسی>",\n'
        '  "urls": [<لیست یکتای URL هایی که برای نوشتن پاسخ استفاده شد>]\n'
        "}"
    ),
)

# ───────────────────────────────────────────────────────────────────────────────
# 4) Graph
# ───────────────────────────────────────────────────────────────────────────────
graph = StateGraph(MessagesState)

def _chatbot(state: MessagesState):
    response_state = agent.invoke({"messages": state["messages"]})
    return {"messages": response_state["messages"]}

graph.add_node("chatbot", _chatbot)
graph.add_edge(START, "chatbot")
graph.add_edge("chatbot", END)

_compiled_graph = graph.compile(checkpointer=MemorySaver())

# ───────────────────────────────────────────────────────────────────────────────
# 5) One-turn helper
# ───────────────────────────────────────────────────────────────────────────────
def run_turn(user_text: str, *, thread_id: str | None = None) -> tuple[str, list[str], str, list[dict]]:
    thread_id = thread_id or str(uuid.uuid4())
    cfg = {"configurable": {"thread_id": thread_id}}

    result = _compiled_graph.invoke(
        {"messages": [{"role": "user", "content": user_text}]},
        config=cfg,
    )

    full_history = serialize_history(result["messages"])
    raw = result["messages"][-1].content
    urls: List[str] = []
    try:
        parsed = json.loads(raw)
        answer = parsed.get("response", raw)
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
