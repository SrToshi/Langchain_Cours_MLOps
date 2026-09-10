from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from dotenv import load_dotenv
import os

load_dotenv()

checkpointer = InMemorySaver()

agent = create_agent(
    model=os.getenv("CHAT_MODEL", "groq:openai/gpt-oss-120b"),
    tools=[],
    system_prompt="You are a helpful and precise assistant.",
    checkpointer=checkpointer,
)

if __name__ == "__main__":

    alice_config = {"configurable": {"thread_id": "alice"}}
    bob_config = {"configurable": {"thread_id": "bob"}}
    charlie_config = {"configurable": {"thread_id": "charlie"}}

    # Alice talks
    agent.invoke(
        {"messages": [{"role": "user", "content": "Hello, my name is Alice."}]},
        config=alice_config,
    )

    # Bob talks
    agent.invoke(
        {"messages": [{"role": "user", "content": "Hello Alice, my name is Bob. Nice to meet you"}]},
        config=bob_config,
    )

    # Alice talks again
    agent.invoke(
        {"messages": [{"role": "user", "content": "Hello Bob. I am a student of medicine. And you?"}]},
        config=alice_config,
    )

    # Charlie talks
    agent.invoke(
        {"messages": [{"role": "user", "content": "Hello, my name is Charlie."}]},
        config=charlie_config,
    )

    # Bob talks again
    agent.invoke(
        {"messages": [{"role": "user", "content": "Hi Charlie! I work in cybersecurity."}]},
        config=bob_config,
    )

    # Final question for Alice
    alice_result = agent.invoke(
        {"messages": [{"role": "user", "content": "What is my name and my field of study?"}]},
        config=alice_config,
    )

    # Final question for Bob
    bob_result = agent.invoke(
        {"messages": [{"role": "user", "content": "What is my name and my field?"}]},
        config=bob_config,
    )

    # Final question for Charlie
    charlie_result = agent.invoke(
        {"messages": [{"role": "user", "content": "What is my name?"}]},
        config=charlie_config,
    )
    print("Alice:", alice_result["messages"][-1].content)
    print("Bob:", bob_result["messages"][-1].content)
    print("Charlie:", charlie_result["messages"][-1].content)