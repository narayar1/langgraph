from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import os


load_dotenv()




def main():
    print("Hello World")
    information = """ Elon Reeve Musk (/ˈiːlɒn/ ⓘ EE-lon; born June 28, 1971) is a businessman and former public official who is the CEO and largest shareholder of Tesla and SpaceX. Musk has been the wealthiest person in the world since 2025, and briefly became the only trillionaire (in terms of US dollars) in June 2026; as of August 14, 2026, Forbes estimates his net worth to be US$864 billion."""
    summary_temmplate = """Given the information {information} about the person
    i want you to create 
    1. A short summary
    2. two interesting facts about the person"""

    summary_prompt_template = PromptTemplate(
        input_variables=["information"],
        template=summary_temmplate
    )

    llm = ChatGoogleGenerativeAI(
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    model="gemini-2.5-flash-lite",
    temperature=0,
    max_tokens=500,
)
    chain = summary_prompt_template | llm
    result = chain.invoke(input= {"information": information})
    print(result.content)
    


if __name__ == "__main__":
    main()