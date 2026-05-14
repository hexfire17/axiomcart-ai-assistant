"""
Graph node functions.

Architecture:

  Main graph:
    START → orchestrator ─┬─ menu_agent ──→ synthesizer → END
                          └─ order_agent ──↗

  Each agent is internally a compiled subgraph with a model ⇄ tools loop:
    START → model ─┬─ tools → model (loop back)
                   └─ END    (no tool calls → done)

Key concepts:
  • Tool nodes: tools are separate graph nodes, not manual loops.
    LangGraph controls the model ⇄ tools cycle natively.
  • Parallel dispatch via Send()
  • HITL via conversation persistence — if an agent needs info
    (e.g. order ID), it simply asks. The answer arrives on the
    next turn with full conversation history.
  • Response synthesis for multi-agent queries
"""

from __future__ import annotations

import operator
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Send, interrupt
from sympy.multipledispatch.conflict import ordering

from src.config import get_logger, llm
from src.state import SnackStackState, ClassificationResult, WorkerInput
from src.tools import (
    get_order_status,
    search_snackstack_menu,
)

logger = get_logger("nodes")


# ── Agent Prompts ────────────────────────────────────────────

MENU_PROMPT = """\
You are the Menu Search & reccomendation Agent for SnackStack.

ROLE: Help customers find food to eat. You also handle
general conversation (greetings, thanks, chitchat).

TOOLS:
  search_snackstack_menu – semantic search over our menu db

GUIDELINES:
- For menu questions, always search the snackstack menu first.
- If an item is out of stock, suggest alternatives.
- If a customer is looking for similar food items or food items at all for reccomendation, ALWAYS search the snackstack menu
  and only recommend items that are actually provided by the tool. Do NOT pretend we have e.g. indian options that we do not in the menu
- If the search returns menu items the customer has already seen or that
  don't match what they asked for (wrong cuisine, type, dietary restriction, etc.),
  be honest and say we don't currently have what they're looking for.
  Do NOT present irrelevant items as if they match the request.
- Keep responses concise and helpful.
- Pay extra close attention to dietary restrictions if provided.
- Sometimes customers may want either alternative or similar items to those in their order. Check prior message context
  from the tool call from the order agent and use that context to look up options similar to their food item, category,
  cuisine, or dietary restrictions. Do NOT ask them for more details in this case.
- For greetings or general chat, respond warmly without calling tools.
  """

ORDER_PROMPT = f"""\
You are the Order Support Agent for SnackStack.

ROLE: Handle order enquiries.

TOOLS:
  get_order_status   – look up an order by Order ID (e.g. ORD-201), Tracking ID (e.g. SS201TRK), or email.
                       this tool attempts to normalize customer output and can handle many different formats
                       for the identifier. If all else fails it will search for a number provided.

GUIDELINES:
- If the customer has NOT provided an order ID, tracking #, or email, you MUST ask
  for it before calling any tools. Say something like: "Could you
  please provide your order ID (e.g. ORD201), Tracking ID (e.g. SS201TRK), or email used when placing the order
  so I can help it up? Remember that there is normalization, so if you see any sort of number and asking for tracking
  or status, you can attempt to make the tool call.
- Be empathetic and professional.
- After retrieving information, respond directly to the customer.
"""


# ── Tool bindings ────────────────────────────────────────────

menu_tools = [search_snackstack_menu]
menu_tools_by_name = {t.name: t for t in menu_tools}

order_tools = [get_order_status]
order_tools_by_name = {t.name: t for t in order_tools}

menu_llm = llm.bind_tools(menu_tools)
order_llm   = llm.bind_tools(order_tools)


# ═══════════════════════════════════════════════════════════
#  Agent Subgraphs (model ⇄ tools)
# ═══════════════════════════════════════════════════════════

class AgentState(TypedDict):
    """Minimal state for the model ⇄ tools subgraph loop."""
    messages: Annotated[list[AnyMessage], operator.add]

def should_continue(state: AgentState) -> str:
    """Route after model node: tool_calls → tools, otherwise → END."""
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return END


# ── Product subgraph ─────────────────────────────────────────

def menu_model(state: AgentState) -> dict:
    """Call the product LLM (with tools bound)."""
    response = menu_llm.invoke(state["messages"])
    logger.info("[menu:model] tool_calls=%s", bool(response.tool_calls))
    return {"messages": [response]}


def menu_tools(state: AgentState) -> dict:
    """Execute tool calls from the menu LLM."""
    last = state["messages"][-1]
    results = []
    for tc in last.tool_calls:
        name, args = tc["name"], tc["args"]
        logger.info("[menu:tools] %s(%s)", name, args)
        out = menu_tools_by_name[name].invoke(args) if name in menu_tools_by_name else f"Unknown tool: {name}"
        results.append(ToolMessage(content=str(out), tool_call_id=tc["id"]))
    return {"messages": results}


pb = StateGraph(AgentState)
pb.add_node("model", menu_model)
pb.add_node("tools", menu_tools)
pb.add_edge(START, "model")
pb.add_conditional_edges("model", should_continue)
pb.add_edge("tools", "model")
menu_subgraph = pb.compile()


# ── Support subgraph ─────────────────────────────────────────

def order_model(state: AgentState) -> dict:
    """Call the support LLM. If it asks for info without calling tools,
    use interrupt() to pause the graph and collect user input."""
    response = order_llm.invoke(state["messages"])
    logger.info("[support:model] tool_calls=%s", bool(response.tool_calls))

    # If no tool calls and no tools have been called yet,
    # the agent is asking for missing info — interrupt for HITL
    if not response.tool_calls:
        any_tools_called = any(isinstance(m, ToolMessage) for m in state["messages"])
        if not any_tools_called:
            logger.info("[support:model] HITL: interrupting to collect user info")
            user_reply = interrupt(response.content)
            logger.info("[support:model] HITL: user replied %r", user_reply)
            return {"messages": [response, HumanMessage(content=str(user_reply))]}

    return {"messages": [response]}


def order_tools(state: AgentState) -> dict:
    """Execute tool calls from the support LLM."""
    last = state["messages"][-1]
    results = []
    for tc in last.tool_calls:
        name, args = tc["name"], tc["args"]
        logger.info("[support:tools] %s(%s)", name, args)
        out = order_tools_by_name[name].invoke(args) if name in order_tools_by_name else f"Unknown tool: {name}"
        results.append(ToolMessage(content=str(out), tool_call_id=tc["id"]))
    return {"messages": results}


def support_should_continue(state: AgentState) -> str:
    """Route after support model node. If the last message is a
    HumanMessage (user answered via HITL interrupt), loop back to model."""
    last = state["messages"][-1]
    if isinstance(last, HumanMessage):
        return "model"
    # tools_calls att is handled internally by langgraph library, dictated by LLM response
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return END


sb = StateGraph(AgentState)
sb.add_node("model", order_model)
sb.add_node("tools", order_tools)
sb.add_edge(START, "model")
sb.add_conditional_edges("model", support_should_continue)
sb.add_edge("tools", "model")
order_subgraph = sb.compile()


# ── Conversation context helper ──────────────────────────────

def build_context(messages: list[AnyMessage]) -> str:
    """Format prior conversation turns as text for agent context."""
    if not messages:
        return ""
    parts = []
    for m in messages:
        if isinstance(m, HumanMessage):
            parts.append(f"Customer: {m.content}")
        elif isinstance(m, AIMessage):
            parts.append(f"Assistant: {m.content}")
    if not parts:
        return ""
    return "CONVERSATION SO FAR:\n" + "\n".join(parts) + "\n\n"


# ═══════════════════════════════════════════════════════════
#  NODE 1 — Orchestrator
# ═══════════════════════════════════════════════════════════

def orchestrator_node(state: SnackStackState) -> Command[Literal["menu_agent", "order_agent", "synthesizer"]]:
    """Classify the user query and dispatch to the right agent(s)."""
    user_query = state.get("user_query", "")
    if not user_query and state.get("messages"):
        user_query = state["messages"][-1].content

    logger.info("Orchestrator  query=%r", user_query)

    prompt = (
        f'Analyse this customer query and decide which agent(s) should handle it.\n\n'
        f'QUERY: "{user_query}"\n\n'
        'AGENTS:\n'
        '  menu_agent – recommendations, similar food options,\n'
        '                  AND general conversation (greetings, thanks, chitchat)\n'
        '  order_agent   – order status\n\n'
        'RULES:\n'
        '1. Greetings, chitchat, general questions (hi, hello, thanks, how are you)\n'
        '   → menu_agent only\n'
        '2. Menu item only queries  → menu_agent only\n'
        '3. Order/support queries → order_agent only\n'
        '4. Mixed queries         → BOTH agents, requires_synthesis = true\n'
        '\nIMPORTANT: Only route to order_agent when the query clearly involves\n'
        'an order status or complaint. When in doubt, use menu_agent.\n'
    )

    classifier = llm.with_structured_output(ClassificationResult)
    try:
        classification = classifier.invoke(prompt)
    except Exception:
        logger.exception("Classification failed — defaulting to order_agent")
        classification = ClassificationResult(
            tasks=[], requires_synthesis=False,
            reasoning="Fallback: classification error",
        )

    logger.info("  routing=%s  synthesis=%s",
                [t.agent for t in classification.tasks],
                classification.requires_synthesis)

    return {
        "tasks": classification.tasks,
        "user_query": user_query,
    }

def parallel_run(state: SnackStackState) -> Command[Literal["menu_agent", "order_agent", "synthesizer"]]:
    targets: list[Send] = []
    for task in state.get("tasks", []):
        targets.append(Send(task.agent, {
            "messages": state.get("messages", []),
            "user_query": state.get("user_query", ""),
            "task_description": task.task_description,
            "is_parallel": True,
        }))

    if not targets:
        targets = [Send("synthesizer", {})]

    return Command(
        update={ # update the main axion state
            "tasks": state.get("tasks", []),
            "requires_synthesis": state.get("requires_synthesis", False),
            "user_query": state.get("user_query", ""),
            "agent_results": [],  # reset stale results from prior turns
        },
        goto=targets, # execute the Send List which sends Arg(s) to Node(s)
    )


def sequential_run(state: SnackStackState) -> Command[Literal["menu_agent", "order_agent", "synthesizer"]]:
    tasks = state.get("tasks", [])

    if not tasks:
        return Command(goto="synthesizer")

    current_task = tasks[0]
    remaining_tasks = tasks[1:]

    return Command(
        update={
            "tasks": remaining_tasks,  # Just update the queue
        },
        goto=Send(current_task.agent, {
            "messages": state.get("messages", []),
            "user_query": state.get("user_query", ""),
            "task_description": current_task.task_description,
        })
    )

# ═══════════════════════════════════════════════════════════
#  NODE 2 — Product Agent
# ═══════════════════════════════════════════════════════════

def menu_agent(state: WorkerInput) -> Command[Literal["synthesizer"]]:
    """Run the product-discovery agent via its model ⇄ tools subgraph."""
    user_query = state.get("user_query", "")
    task_desc  = state.get("task_description", user_query)
    logger.info("Menu Agent  task=%r", task_desc)

    context = build_context(state.get("messages", []))

    # calling subgraph within the node! - different schema
    # basic conversion to just messages below
    result = menu_subgraph.invoke({"messages": [
        SystemMessage(content=MENU_PROMPT),
        HumanMessage(content=f"{context}Task: {task_desc}\nCustomer query: {user_query}"),
    ]})

    answer = result["messages"][-1].content

    # simple insertion to transform answer back to parent type
    # could also just return the state update and add edge menu_agent->synth if desired since this is not dynamic
    # e.g. return {"agent_results": [{"source": "product_discovery", "response": answer}]}
    goto_node = "synthesizer"
    if not state.get("is_parallel", False):
        goto_node = "executor" # run remaining tasks

    return Command(
        update={"agent_results": [{"source": "menu_discovery", "response": answer}],
                "messages":[AIMessage(content=answer)]},

        goto=goto_node,
    )


# ═══════════════════════════════════════════════════════════
#  NODE 3 — Support Agent
# ═══════════════════════════════════════════════════════════

def order_agent(state: WorkerInput) -> Command[Literal["synthesizer"]]:
    """Run the sales-support agent via its model ⇄ tools subgraph.

    HITL is handled through conversation persistence: if the agent
    needs info (e.g. order ID), it responds with a question. The
    user's answer arrives on the next turn via the message history.
    """
    user_query = state.get("user_query", "")
    task_desc  = state.get("task_description", user_query)
    logger.info("Order Agent  task=%r", task_desc)

    context = build_context(state.get("messages", []))

    result = order_subgraph.invoke({"messages": [
        SystemMessage(content=ORDER_PROMPT),
        HumanMessage(content=f"{context}Task: {task_desc}\nCustomer query: {user_query}"),
    ]})

    answer = result["messages"][-1].content

    goto_node = "synthesizer"
    if not state.get("is_parallel", False):
        logger.info("running sequentially")
        goto_node = "executor" # run remaining tasks

    return Command(
        update={"agent_results": [{"source": "menu_discovery", "response": answer}],
                "messages":[AIMessage(content=answer)]},

        goto=goto_node,
    )

# ═══════════════════════════════════════════════════════════
#  NODE 4 — Synthesizer
# ═══════════════════════════════════════════════════════════

def synthesizer_node(state: SnackStackState) -> dict:
    """Merge results from one or more agents into a single user-facing reply."""
    results = state.get("agent_results", [])
    user_query = state.get("user_query", "")

    if not results:
        logger.warning("Synthesizer received no agent results")
        return {"final_answer": "Sorry, I couldn't process that request. Please try again."}

    if len(results) == 1:
        logger.info("Synthesizer  single-agent pass-through")
        return {"final_answer": results[0]["response"]}

    logger.info("Synthesizer  merging %d agent responses", len(results))

    parts = "\n\n".join(
        f"[{r['source'].upper()}]:\n{r['response']}" for r in results
    )
    prompt = (
        f"You are combining responses from multiple specialist agents.\n\n"
        f"CUSTOMER QUERY: {user_query}\n\n"
        f"AGENT RESPONSES:\n{parts}\n\n"
        "Write a single, coherent reply that addresses every part of the "
        "customer's query. Be concise. Speak as 'SnackStack Assistant'. Do NOT"
        "include anything in the response food item / order-wise that was not explicitly"
        "looked up via a tool."
    )

    merged = llm.invoke(prompt)
    return {"final_answer": merged.content}