import os
import sys
from typing import Any, Dict, TypedDict, List
from dotenv import load_dotenv

# --- LangGraph & LangChain Imports ---
from langgraph.graph import StateGraph, START, END
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings  # Updated Import
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langchain_core.documents import Document
from prompts import SYSTEM_PROMPT, HUMAN_PROMPT
# --- OpenAI Import ---
from openai import OpenAI
from langchain_classic import hub

# 1. Load Environment Variables
load_dotenv()

# --- Configuration & Prompts (Mocked for single-file template) ---
# In a real app, these might come from a prompts.py file
# SYSTEM_PROMPT = """You are a helpful RAG assistant. 
# Use the provided context to answer the user's question. 
# If the context doesn't contain the answer, say you don't know."""

# HUMAN_PROMPT = """
# Context:
# {context}

# Question: 
# {input}
# """

# --- Agent State Definition ---
class AgentState(TypedDict):
    input: str
    context: Dict[str, Any]
    history: List[BaseMessage]
    response: Any  # response will hold the stream object

# --- RAG Setup (Robust Fallback) ---
# We try to load PDFs, but if they miss or libraries fail, we use a dummy retriever
# so the agent loop still works for testing.
retriever = None

try:
    # Check if files exist before trying to load
    pdf_files = ["pdf1.pdf", "pdf2.pdf"]
    existing_pdfs = [f for f in pdf_files if os.path.exists(f)]

    if existing_pdfs:
        print(f"Found PDFs: {existing_pdfs}. Loading...")
        documents = []
        for pdf in existing_pdfs:
            loader = PyPDFLoader(pdf)
            documents.extend(loader.load())
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(documents)
        
        # --- UPDATED EMBEDDINGS SETUP ---
        # Using HuggingFace Embeddings (runs locally, free, no API key required)
        # Requirement: pip install sentence-transformers
        try:
            print("Initializing Local Embeddings (all-MiniLM-L6-v2)...")
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        except ImportError:
            print("Error: 'sentence-transformers' not installed.")
            print("Please run: pip install sentence-transformers")
            raise

        vectorstore = Chroma.from_documents(documents=texts, embedding=embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        print("RAG System Initialized Successfully.")
    else:
        print("Warning: No PDF files found (pdf1.pdf, pdf2.pdf). using Dummy Retriever.")
        raise FileNotFoundError("No PDFs")

except Exception as e:
    print(f"RAG initialization skipped/failed: {e}")
    print("Using a dummy retriever for testing purposes.")
    
    # Mock retriever class to prevent crashes
    class DummyRetriever:
        def invoke(self, query):
            return [Document(page_content="This is dummy context because the RAG system failed to load.")]
    
    retriever = DummyRetriever()


# --- OpenAI Client Setup ---
# Ensure you have OPENROUTER_KIMI_API_KEY in your .env file
api_key = os.getenv("OPENROUTER_KIMI_API_KEY")
if not api_key:
    print("WARNING: OPENROUTER_KIMI_API_KEY not found in .env. Agent calls will likely fail.")

client = OpenAI(
    api_key=api_key,
    base_url="https://openrouter.ai/api/v1",
)

# --- Node Functions ---

def retrieve_context(state: AgentState):
    """Retrieves relevant context documents."""
    query = state["input"]
    try:
        docs = retriever.invoke(query)
        context = "\n\n".join([doc.page_content for doc in docs])
    except Exception as e:
        context = "Error retrieving context."
    return {"context": context}

def generate_response(state: AgentState):
    """Generates the response using the LLM in streaming mode."""
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="history"),
        HumanMessage(content=HUMAN_PROMPT.replace("{input}", state["input"]).replace("{context}", str(state["context"])))
    ])

    # Construct messages list
    # Note: We default history to empty list if not present
    messages = prompt.invoke({"history": state.get("history", [])}).messages

    # IMPORTANT: Stream=True returns a generator
    try:
        response_stream = client.chat.completions.create(
            model="moonshotai/kimi-k2:free", # Ensure this model name is correct for your provider
            messages=[{"role": m.type, "content": m.content} for m in messages],
            temperature=0.3,
            stream=True
        )
        return {"response": response_stream}
    except Exception as e:
        return {"response": [type('obj', (object,), {'choices': [type('obj', (object,), {'delta': type('obj', (object,), {'content': f"Error calling LLM: {str(e)}"})})]})]}

def create_agent():
    """Compiles the LangGraph StateGraph."""
    graph = StateGraph(AgentState)
    graph.add_node("retrieve_context", retrieve_context)
    graph.add_node("generate_response", generate_response)
    
    graph.add_edge(START, "retrieve_context")
    graph.add_edge("retrieve_context", "generate_response")
    graph.add_edge("generate_response", END)
    
    return graph.compile()

# --- Generator for the Frontend/Terminal ---
def run_Agent(input_text: str):
    """
    Runs the agent and yields text chunks for streaming.
    """
    agent = create_agent()
    
    # Initialize with empty history for this test
    initial_state = {"input": input_text, "history": []}
    
    stream_iterator = agent.stream(initial_state)
    
    for chunk in stream_iterator:
        if "generate_response" in chunk:
            response_payload = chunk["generate_response"]["response"]
            
            # Check if it's a valid stream object or our error mock
            if hasattr(response_payload, '__iter__'):
                for token in response_payload:
                    if hasattr(token, 'choices') and len(token.choices) > 0:
                        content = token.choices[0].delta.content
                        if content:
                            yield content
            else:
                yield "Error: Invalid response format from LLM."

# --- Main Execution Block (Terminal Interface) ---
if __name__ == "__main__":
    print("\n" + "="*50)
    print("🤖 Local RAG Agent CLI")
    print("Type 'quit', 'exit', or 'q' to stop.")
    print("="*50 + "\n")

    while True:
        try:
            user_input = input("\033[1;34mYou:\033[0m ") # Blue text for "You:"
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break
            
            print("\033[1;32mAgent:\033[0m ", end="", flush=True) # Green text for "Agent:"
            
            # Iterate over the generator from run_Agent
            for text_chunk in run_Agent(user_input):
                print(text_chunk, end="", flush=True)
            
            print("\n") # New line after stream finishes
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")