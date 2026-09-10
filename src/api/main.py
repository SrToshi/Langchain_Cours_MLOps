from fastapi import FastAPI
from pydantic import BaseModel
from src.core.chains import summary_chain, translation_chain
from src.agents.doc_agent import doc_agent, chat_agent, checkpointer
from fastapi import HTTPException

app = FastAPI(title="LangChain Course API", version="4.0.0")

users = {}

class SignupInput(BaseModel):
    username: str
    password: str

class LoginInput(BaseModel):
    username: str
    password: str

class TextInput(BaseModel):
    text: str

class AgentInput(BaseModel):
    file_path: str
    query: str

class HistoryInput(BaseModel):
    session_id: str

@app.post("/summary")
def summarize_text(input: TextInput):
    result = summary_chain.invoke({"input": input.text})
    return {"summary": result.summary}


@app.post("/translate")
def translate_text(input: TextInput):
    result = translation_chain.invoke({"input": input.text})
    return {"translated_text": result.translated_text}

@app.post("/agent")
def run_agent(input: AgentInput):
    result = doc_agent.invoke({
        "messages": [
            {
                "role": "user",
                "content": (
                    f"Load the document located here: {input.file_path}. "
                    f"Then answer this question: {input.query}"
                ),
            }
        ]
    })
    return {"response": result["messages"][-1].content}
class ChatInput(BaseModel):
    session_id: str
    query: str

@app.post("/chat")
def chat(input: ChatInput):
    result = chat_agent.invoke(
        {"messages": [{"role": "user", "content": input.query}]},
        config={"configurable": {"thread_id": input.session_id}},
    )
    return {"response": result["messages"][-1].content}

@app.post("/history")
def history(input: HistoryInput):
    state = checkpointer.get({"configurable": {"thread_id": input.session_id}})
    return {"history": state}

@app.post("/signup")
def signup(input: SignupInput):
    users[input.username] = input.password
    return {"status": "created"}

@app.post("/login")
def login(input: LoginInput):
    if users.get(input.username) != input.password:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return {"session_id": input.username}