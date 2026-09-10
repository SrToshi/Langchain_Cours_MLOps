import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from src.documents.tools import load_pdf_tool, clean_text_tool
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

TOOLS = [load_pdf_tool, clean_text_tool]

doc_agent = create_agent(
    model=os.getenv("CHAT_MODEL", "groq:openai/gpt-oss-120b"),
    tools=TOOLS,
    system_prompt=(
        "You are a document analysis specialist assistant. "
        "You use only the provided tools. "
        "You never invent a result absent from the document."
    ),
)

checkpointer = InMemorySaver()

chat_agent = create_agent(
    model=os.getenv("CHAT_MODEL", "groq:openai/gpt-oss-120b"),
    tools=[],
    system_prompt="You are a helpful and precise assistant.",
    checkpointer=checkpointer,
)