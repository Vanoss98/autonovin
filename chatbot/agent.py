import os
from pydantic import BaseModel
from datetime import datetime
import uuid
from typing import Annotated
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, MessagesState, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from django.conf import settings
from .utils import analyze_query, hybrid_search, generate_answer
import json
from typing import Any, Dict, List
from langchain_core.messages import BaseMessage

def _content_to_str(content: Any) -> str:
    # LangChain message content can be str or list[{"type":"text","text":...}, ...]
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
    # Map LangChain message types to friendly roles
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
            # drop empty assistant stubs (e.g., tool-call scaffolding)
            if not (content or "").strip():
                continue
            # if assistant content is JSON with {"response": "...", ...}, keep only the response text
            try:
                payload = json.loads(content)
                if isinstance(payload, dict) and "response" in payload:
                    content = payload["response"]
            except Exception:
                pass  # leave as-is if not JSON

        out.append({"role": role, "content": content})
    return out


# ── 1. base LLM ───────────────────────────────────────────────────────────
llm = ChatOpenAI(model="gpt-4.1", temperature=0, api_key=settings.OPENAI_API_KEY, timeout=60.0, max_retries=5)


# ── 2. tools ──────────────────────────────────────────────────────────────
@tool
def car_spec_rag(query: str) -> str:
    """Returns relevant documents and specification information for a specific car or multiple cars at once.
        if the documents were none, say i dont know the answer. make the answer short but complete and to the point
        this is not the tool used for car price. do not use it for price"""
    analysis = analyze_query(query)
    docs = hybrid_search(analysis, top_k=20)

    flat_urls = list({
        tup[6]
        for model_docs in docs.values()
        for tup in model_docs
        if tup[6]
    })

    # NEW: collect page_ids from tuple index 8
    page_ids = sorted({
        tup[8]
        for model_docs in docs.values()
        for tup in model_docs
        if len(tup) >= 9 and tup[8] is not None
    })

    payload = {"docs": docs, "urls": flat_urls, "page_ids": page_ids}
    return json.dumps(payload, ensure_ascii=False)


@tool
def car_price(car: str) -> str:
    """Return price for a given car (mocked)."""
    return "Price for BMW X5 in the US starts at $54 200."


tools = [car_spec_rag, car_price]

# ── 3. agent (React) ──────────────────────────────────────────────────────
agent = create_react_agent(
    model=llm,
    tools=tools,
    prompt=(
        "You are a helpful automotive assistant.\n\n"
        "• You may call at most **one** tool for each question.\n"
        "• `car_spec_rag` returns a JSON object with keys `docs` and `urls`.\n"
        "• After thinking, finish with **exactly** the JSON below – nothing else:\n"
        "{\n"
        '  "response": "<پاسخ شما به فارسی>",\n'
        '  "urls": [<لیست یکتای URL هایی که برای نوشتن پاسخ استفاده شد>]\n'
        "}"
    ),
)

# ── 4. graph ──────────────────────────────────────────────────────────────
graph = StateGraph(MessagesState)


def _chatbot(state: MessagesState):
    response_state = agent.invoke({"messages": state["messages"]})
    return {"messages": response_state["messages"]}


graph.add_node("chatbot", _chatbot)
graph.add_edge(START, "chatbot")
graph.add_edge("chatbot", END)

_compiled_graph = graph.compile(checkpointer=MemorySaver())  # in-process memory


# ── 5. helper for one turn ────────────────────────────────────────────────
def run_turn(user_text: str, *, thread_id: str | None = None) -> tuple[str, list[str], str, list[dict]]:
    thread_id = thread_id or str(uuid.uuid4())
    cfg = {"configurable": {"thread_id": thread_id}}

    result = _compiled_graph.invoke(
        {"messages": [{"role": "user", "content": user_text}]},
        config=cfg,
    )

    # result["messages"] is the entire conversation for this thread after this turn
    full_history = serialize_history(result["messages"])

    raw = result["messages"][-1].content  # final assistant content
    urls: list[str] = []

    try:
        parsed = json.loads(raw)
        answer = parsed.get("response", raw)
        urls = parsed.get("urls", [])
    except Exception:
        answer = raw

    return answer, urls, thread_id, full_history
