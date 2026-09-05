from dotenv import load_dotenv

# VERY IMPORTANT:
# Load .env before importing LangChain
load_dotenv(override=True)

import os

from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI


def main():

    print("====================================")
    print("LangSmith Configuration")
    print("====================================")

    print("Endpoint :", os.getenv("LANGSMITH_ENDPOINT"))
    print("Project  :", os.getenv("LANGSMITH_PROJECT"))
    print("Tracing  :", os.getenv("LANGSMITH_TRACING"))
    print("API Key  :", bool(os.getenv("LANGSMITH_API_KEY")))

    print("\n====================================")
    print("Creating LangChain chain")
    print("====================================")

    information = """
    Elon Musk is the CEO of Tesla and SpaceX.
    He is also associated with several other technology companies.
    """

    prompt = PromptTemplate(
        input_variables=["information"],
        template="""
        Given the following information:

        {information}

        Provide a short summary in 3 sentences.
        """
    )

    llm = ChatGoogleGenerativeAI(
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        model="gemini-2.5-flash-lite",
        temperature=0,
        max_tokens=500,
    )

    chain = prompt | llm

    print("\nInvoking chain...")

    result = chain.invoke(
        {
            "information": information
        }
    )

    print("\n====================================")
    print("RESULT")
    print("====================================")

    print(result.content)

    print("\n====================================")
    print("Chain completed")
    print("====================================")
    print("Check LangSmith project: langgraphcourse")


if __name__ == "__main__":
    main()