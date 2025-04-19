from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from rag_chain import run_graph

app = FastAPI()

# 允許 CORS（讓前端 Next.js 可以直接 call API）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 開發階段允許所有來源，正式上線要鎖定
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    thread_id: str


class ChatResponse(BaseModel):
    answer: str
    sources: list

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    answer, source_docs = run_graph(request.message, request.thread_id)

    sources = []

    for doc in source_docs:
        sources.append({
            "page_content": doc.page_content,
            "metadata": doc.metadata
        })
    
    return ChatResponse(answer=answer, sources=sources)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)