from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv
load_dotenv()

reflection_prompt = ChatPromptTemplate.from_messages([
    ("system", 
     "You are a viral twitter influencer grading a tweet"
     "Generate a critique and recommendation for the user"
     " Always provide a detailed recommendation"
     "Include rquequests for length, virality, style etc.."
    ),
    MessagesPlaceholder(variable_name="messages")
]
)

generation_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a twitter techie influencer tasked wit writting excellent twitter posts"
     "Generate the best twitter post possible based on the user's request"
     "If the user provides critique, respond with a new tweet based on the critique"
    ),
    MessagesPlaceholder(variable_name="messages"),
]
)

llm = ChatGoogleGenerativeAI(
                google_api_key=os.getenv("GOOGLE_API_KEY"),
                model= "gemini-2.5-flash-lite",
                temperature=0.7, 
                max_tokens=500)

generate_chain = generation_prompt| llm
reflect_chain = reflection_prompt | llm






