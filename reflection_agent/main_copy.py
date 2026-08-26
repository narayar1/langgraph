from typing import TypedDict, Annotated

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, StateGraph, MessagesState
from langgraph.graph.message import add_messages

from chains import generate_chain, reflect_chain

class MessageGraph(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

import os

load_dotenv()

REFLECT = "reflect"
GENERATE = "generate"


def generation_node(state: MessagesState):
    """
    Generate a new tweet based on the messages in the state.
    """
    response = generate_chain.invoke({
        "messages": state["messages"]
    })

    return {
        "messages": [response]
    }


def reflection_node(state: MessagesState):
    """
    Generate a critique of the current tweet.
    """
    response = reflect_chain.invoke({
        "messages": state["messages"]
    })

    return {
        "messages": [
            HumanMessage(content=response.content)
        ]
    }


builder = StateGraph(MessagesState)

builder.add_node(GENERATE, generation_node)
builder.add_node(REFLECT, reflection_node)
builder.set_entry_point(GENERATE)


def should_continue(state: MessagesState):
    if len(state["messages"]) > 6:
        return END

    return REFLECT


builder.add_conditional_edges(
    GENERATE,
    should_continue,
    {
        END: END,
        REFLECT: REFLECT
    }
)

builder.add_edge(REFLECT, GENERATE)

graph = builder.compile()


# print(graph.get_graph().draw_mermaid())
# graph.get_graph().print_ascii()


if __name__ == "__main__":
    print("Hello LangGraph")
    inputs = HumanMessage(content= """Make this tweet better:"
    @lanhchainAI
    -new tool calling feature is seriously underrated.
    After a long wait its here making the implementatation of agents cross
    different models with function calling - super easy.
    Make a video covering their newest blog post""")
    response = graph.invoke(inputs)
    print(response)