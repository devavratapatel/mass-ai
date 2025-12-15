from langgraph.graph import StateGraph, START, END
from typing import Any, Dict, TypedDict, List, Optional
import operator
import os
from dotenv import load_dotenv

load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

try:
    from prompts import SYSTEM_PROMPT, HUMAN_PROMPT
except ImportError:
    SYSTEM_PROMPT = "You are a helpful assistant."
    HUMAN_PROMPT = "Context: {context}\n\nQuestion: {input}"

class AgentState(TypedDict):
    input: str
    context: Dict[str, Any]
    history: List[BaseMessage]
    response: Any  

try:

    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

    docs_to_load = []
    if os.path.exists("pdf1.pdf"):
        docs_to_load.append("pdf1.pdf")
    if os.path.exists("pdf2.pdf"):
        docs_to_load.append("pdf2.pdf")

    if docs_to_load:
        documents = []
        for pdf_file in docs_to_load:
            loader = PyPDFLoader(pdf_file)
            documents.extend(loader.load())
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(documents)

        vectorstore = Chroma.from_documents(documents=texts, embedding=embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    else:
        print("Warning: PDF files not found. Initializing empty vectorstore.")
        retriever = None

except Exception as e:
    print(f"RAG initialization error: {e}")
    retriever = None


llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.3,
    streaming=True
)

def retrieve_context(state: AgentState):
    """Retrieves relevant context documents."""
    query = state["input"]
    
    if retriever:
        docs = retriever.invoke(query)
        context = "\n\n".join([doc.page_content for doc in docs])
    else:
        context = "No documents found or RAG not initialized."
        
    return {"context": context}

def generate_response(state: AgentState):
    """Generates the response using Gemini-2.0-Flash."""
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="history"),
        HumanMessage(content=HUMAN_PROMPT.replace("{input}", state["input"]).replace("{context}", str(state["context"])))
    ])

    messages = prompt.invoke({"history": state.get("history", [])})

    response_stream = llm.stream(messages)

    return {"response": response_stream}

def create_agent():
    """Compiles the LangGraph StateGraph."""
    graph = StateGraph(AgentState)
    graph.add_node("retrieve_context", retrieve_context)
    graph.add_node("generate_response", generate_response)
    
    graph.add_edge(START, "retrieve_context")
    graph.add_edge("retrieve_context", "generate_response")
    graph.add_edge("generate_response", END)
    
    return graph.compile()

def run_Agent(input_text: str):
    """
    Runs the agent and yields text chunks for streaming.
    This acts as a generator for the frontend.
    """
    agent = create_agent()
    
    stream_iterator = agent.stream({"input": input_text, "history": []}) 
    
    for chunk in stream_iterator:
        if "generate_response" in chunk:
            response_stream = chunk["generate_response"]["response"]
            
            for message_chunk in response_stream:
                content = message_chunk.content
                if content:
                    yield content