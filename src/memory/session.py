import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

checkpointer = InMemorySaver()

agent = create_agent(
    model=os.getenv("CHAT_MODEL", "groq:openai/gpt-oss-120b"),
    tools=[],
    system_prompt="You are an educational assistant.",
    checkpointer=checkpointer,
)

if __name__ == "__main__":
    config = {"configurable": {"thread_id": "alice"}}

    response = agent.invoke(
        {"messages": [{"role": "user", "content": "Hello, my name is Alice."}]},
        config=config,
    )
    print(response["messages"][-1].content)

    agent.invoke(
        {"messages": [{"role": "user", "content": "I study medicine."}]},
        config=config,
    )

    result = agent.invoke(
        {"messages": [{"role": "user", "content": "What is my field of study?"}]},
        config=config,
    )
    print(result["messages"][-1].content)