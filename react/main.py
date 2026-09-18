from dotenv import load_dotenv
import os
from typing import List

from pydantic import BaseModel, Field

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch


# ============================================================
# 1. Load environment variables
# ============================================================

load_dotenv(override=True)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")


# ============================================================
# 2. Validate API keys
# ============================================================

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is not set in .env")

if not TAVILY_API_KEY:
    raise ValueError("TAVILY_API_KEY is not set in .env")


print("Google API key found:", bool(GOOGLE_API_KEY))
print("Tavily API key found:", bool(TAVILY_API_KEY))


# ============================================================
# 3. Pydantic schemas
# ============================================================

class Source(BaseModel):
    """Schema for a source used by the agent."""

    title:str = Field(description="The title of the source used to answer the question.")

    url: str = Field(
        description="The URL of the source used to answer the question."
    )

    relavance:float = Field(
        description="The relevance score of the source to the question."
    )

    published_date:str = Field(
        description="The published date of the source used to answer the question."
    )



class AgentResponse(BaseModel):
    """Schema for the agent's final response."""

    answer: str = Field(
        description="The answer to the user's question."
    )

    sources: List[Source] = Field(
        default_factory=list,
        description="A list of sources used to generate the answer."
    )


# ========cd====================================================
# 4. Create Gemini LLM
# ============================================================

llm = ChatGoogleGenerativeAI(
    google_api_key=GOOGLE_API_KEY,
    model="gemini-2.5-flash-lite",
    temperature=0,
    max_tokens=500,
)


# ============================================================
# 5. Create Tavily Search Tool
# ============================================================

search_tool = TavilySearch(
    tavily_api_key=TAVILY_API_KEY,
    max_results=5,
    topic="general",
)


tools = [search_tool]


# ============================================================
# 6. Create Agent
# ============================================================

agent = create_agent(
    model=llm,
    tools=tools,
    response_format=AgentResponse,
)


# ============================================================
# 7. Main
# ============================================================

def main():

    print("\n======================================")
    print("Starting LangChain Agent")
    print("======================================")

    question = (
       "Use the Tavily search tool to find the current weather in Kochi, Kerala. "
    "Then provide a concise answer and include the sources used. "
    "For each source provide the title, URL, Tavily relevance score, "
    "and publication date if explicitly available in the search result. "
    "If only a 'Last Updated' date is available, use that date and label it "
    "as the last updated date. Do not invent dates."
    )

    print("\nUser question:")
    print(question)

    print("\nCalling agent...")

    result = agent.invoke(
        {
            "messages": [
                HumanMessage(content=question)
            ]
        }
    )

    # ========================================================
    # Print structured response
    # ========================================================

    print("\n======================================")
    print("STRUCTURED RESPONSE")
    print("======================================")

    structured_response = result.get("structured_response")

    if structured_response:

        print("\nAnswer:")
        print(structured_response.answer)

        print("\nSources:")

        for source in structured_response.sources:
            print(f"- {source.url}")

    else:
        print("No structured response returned.")

    # ========================================================
    # Print complete agent execution
    # ========================================================

    print("\n======================================")
    print("AGENT MESSAGE TRACE")
    print("======================================")

    for i, message in enumerate(result["messages"], start=1):

        print(f"\n--- Message {i} ---")

        print("Type:")
        print(type(message).__name__)

        # ----------------------------------------------------
        # AI tool calls
        # ----------------------------------------------------

        if hasattr(message, "tool_calls") and message.tool_calls:

            print("\nTool calls:")

            for tool_call in message.tool_calls:
                print(tool_call)

        # ----------------------------------------------------
        # Tool result
        # ----------------------------------------------------

        if type(message).__name__ == "ToolMessage":

            print("\nTool result:")
            print(message.content)

        # ----------------------------------------------------
        # Message content
        # ----------------------------------------------------

        if hasattr(message, "content"):

            print("\nContent:")

            if isinstance(message.content, list):

                for item in message.content:
                    print(item)

            else:
                print(message.content)

        # ----------------------------------------------------
        # Token usage
        # ----------------------------------------------------

        if hasattr(message, "usage_metadata"):

            if message.usage_metadata:

                print("\nToken usage:")
                print(message.usage_metadata)


# ============================================================
# 8. Run
# ============================================================

if __name__ == "__main__":
    main()