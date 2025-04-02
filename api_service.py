import json
import os
import dotenv
import requests
import re
from fastapi import FastAPI, UploadFile, File, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from typing import Optional
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import uuid

dotenv.load_dotenv()
API_KEY = os.getenv("DEEPSEEK_API_KEY")

app = FastAPI()

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 向量数据库基础目录
BASE_VECTOR_DB_DIR = "vector_dbs"
os.makedirs(BASE_VECTOR_DB_DIR, exist_ok=True)

class Question(BaseModel):
    question: str

def get_user_vector_store(user_id: str):
    """
    获取用户特定的向量数据库实例
    """
    user_db_dir = os.path.join(BASE_VECTOR_DB_DIR, user_id)
    os.makedirs(user_db_dir, exist_ok=True)
    
    embeddings = HuggingFaceEmbeddings(
        model_name="shibing624/text2vec-base-chinese"
    )
    return Chroma(
        persist_directory=user_db_dir,
        embedding_function=embeddings
    )

def semantic_search(vector_store, query, top_k=10):
    """
    语义搜索
    """
    query = re.sub(r'[是谁的吗什么啊了?？。,，!]', '', query).strip()
    try:
        relevant_docs = vector_store.similarity_search(query, k=top_k)
        return relevant_docs
    except Exception as e:
        print(f"语义搜索错误: {e}")
        return None

def generate_response(query, relevant_docs, api_key):
    """
    使用 SiliconFlow API 生成回答
    """
    context = "\n".join([doc.page_content for doc in relevant_docs])

    prompt = f"""基于以下背景信息回答问题：
背景：
{context}
问题：
{query}
请给出详细、准确的回答。如果背景信息中没有直接答案，请诚实地说明。"""

    url = "https://api.siliconflow.cn/v1/chat/completions"
    payload = {
        "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "stream": False,
        "max_tokens": 512,
        "temperature": 0.3,
        "top_p": 0.7,
        "stop": None,
        "top_k": 50,
        "frequency_penalty": 0.5,
        "n": 1,
        "response_format": {"type": "text"},
        "tools": [
            {
                "type": "function",
                "function": {
                    "description": "<string>",
                    "name": "<string>",
                    "parameters": {},
                    "strict": False
                }
            }
        ]
    }

    headers = {
        "Authorization": "Bearer sk-zthoxwuljgnwxpczemvtaxcalsxycxtzrpdzdcijjslxgsrg",
        "Content-Type": "application/json"
    }

    try:
        response = requests.request("POST", url, json=payload, headers=headers)
        response_json = json.loads(response.text)
        reasoning_content = response_json['choices'][0]['message'].get('reasoning_content', '')
        return reasoning_content
    except Exception as e:
        print(f"API 请求错误: {e}")
        return None

@app.post("/init")
async def init_session():
    """
    初始化用户会话，返回用户ID
    """
    user_id = str(uuid.uuid4())
    return {"user_id": user_id}

@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    user_id: str = Header(..., description="用户ID")
):
    try:
        # 确保上传的是txt文件
        if not file.filename.endswith('.txt'):
            raise HTTPException(status_code=400, detail="只支持txt文件上传")
        
        # 保存上传的文件
        file_path = f"uploads/{user_id}/{file.filename}"
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # 加载文档并构建向量数据库
        loader = TextLoader(file_path, encoding='utf-8')
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=64,
            length_function=len
        )
        
        chunks = text_splitter.split_documents(documents)
        print("完成分块")
        
        # 创建用户特定的向量数据库
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=HuggingFaceEmbeddings(model_name="shibing624/text2vec-base-chinese"),
            persist_directory=os.path.join(BASE_VECTOR_DB_DIR, user_id)
        )
        
        # 持久化保存
        vector_store.persist()
        
        return {
            "message": "文件上传成功，向量数据库构建完成",
            "chunks_count": len(chunks)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask")
async def ask_question(
    question: Question,
    user_id: str = Header(..., description="用户ID")
):
    try:
        # 获取用户特定的向量数据库实例
        vector_store = get_user_vector_store(user_id)
        
        # 语义搜索
        relevant_docs = semantic_search(vector_store, question.question)
        
        if not relevant_docs:
            return {"answer": "未找到相关文档"}
        
        # 生成回答
        response = generate_response(question.question, relevant_docs, API_KEY)
        
        if response:
            return {
                "answer": response,
                "references": [doc.page_content[:100] + "..." for doc in relevant_docs]
            }
        else:
            return {"answer": "生成回答时发生错误"}
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 
    