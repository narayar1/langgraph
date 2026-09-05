from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import os


load_dotenv()
llm = ChatGoogleGenerativeAI(
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    model="gemini-2.5-flash-lite",
    temperature=0,
    max_tokens=500,
)


result = llm.invoke("Here is a fact about pluto")
print(result.content)