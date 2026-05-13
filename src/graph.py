"""
Build and compile the LangGraph StateGraph.

Graph topology:

  START → orchestrator ─┬─ menu_agent ──→ synthesizer → END
                        └─ order_agent ──↗

Each agent is internally a subgraph with a model ⇄ tools loop.
The MemorySaver checkpointer persists conversation history across
turns, enabling multi-turn HITL (agent asks a question on one turn,
user answers on the next).
"""

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from src.config import get_logger
from src.nodes import orchestrator_node, menu_agent, order_agent, synthesizer_node
from src.state import SnackStackState

logger = get_logger("graph")


def build_graph() -> StateGraph:
    """Create, wire, and compile the SnackStack multi-agent graph."""

    builder = StateGraph(SnackStackState)

    # ── Add nodes ────────────────────────────────────────
    builder.add_node("orchestrator", orchestrator_node)
    builder.add_node("menu_agent", menu_agent) # call subgraph within node wrapper example
    builder.add_node("order_agent", order_agent)
    builder.add_node("synthesizer", synthesizer_node)

    # ── Add edges ────────────────────────────────────────
    builder.add_edge(START, "orchestrator")
    builder.add_edge("synthesizer", END)

    # ── Compile with checkpointer ────────────────────────
    # MemorySaver persists graph state so that interrupt()-based
    # HITL can pause and resume, and conversation history carries
    # forward between queries.
    memory = MemorySaver()
    graph = builder.compile(checkpointer=memory)

    logger.info("Graph compiled  (with MemorySaver for conversation persistence)")
    return graph


# Module-level singleton
snackstack_graph = build_graph()
