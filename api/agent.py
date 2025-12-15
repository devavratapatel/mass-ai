from langgraph.graph import StateGraph, START, END
from typing import TypedDict, List, Any  # Added Any
import os
import time
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai

load_dotenv()

# --- 1. Configure Google GenAI directly ---
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# Global cache for the uploaded file handle
UPLOADED_FILE = None

def get_file_handle():
    """Uploads the PDF to Google and returns the file handle."""
    global UPLOADED_FILE
    if UPLOADED_FILE:
        return UPLOADED_FILE
    
    target_file = None
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # Ensure this matches your actual file name
    pdf1 = os.path.join(BASE_DIR, "pdf1.pdf")
    if os.path.exists(pdf1):
        target_file = pdf1
    
    if target_file:
        print(f"Uploading {target_file} to Google...")
        myfile = genai.upload_file(target_file)
        
        while myfile.state.name == "PROCESSING":
            print("Processing file on Google servers...")
            time.sleep(2)
            myfile = genai.get_file(myfile.name)
            
        print(f"File ready: {myfile.uri}")
        UPLOADED_FILE = myfile
        return UPLOADED_FILE
    
    return None

# --- 2. Setup LLM ---
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash", 
    temperature=0.3,
    streaming=True
)

SYSTEM_PROMPT = "You are a helpful assistant. Use the provided document to answer questions."

# --- FIX IS HERE ---
class AgentState(TypedDict):
    input: str
    history: List[BaseMessage]
    response: Any  # <--- This was missing! Required to pass the stream back.

def generate_response(state: AgentState):
    file_handle = get_file_handle()
    
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
    ]
    
    if file_handle:
        content_parts = [
            {"type": "text", "text": f"User Question: {state['input']}"},
            {
                "type": "media",
                "file_uri": file_handle.uri,
                "mime_type": file_handle.mime_type
            }
        ]
        messages.append(HumanMessage(content=content_parts))
    else:
        messages.append(HumanMessage(content=state['input']))

    messages.extend(state.get("history", []))

    # We return the generator. 
    # Note: In production graphs, we usually shouldn't store generators in state,
    # but for this direct streaming setup, it works if the key exists in AgentState.
    response_stream = llm.stream(messages)
    return {"response": response_stream}

def create_agent():
    graph = StateGraph(AgentState)
    graph.add_node("generate_response", generate_response)
    graph.add_edge(START, "generate_response")
    graph.add_edge("generate_response", END)
    return graph.compile()

def run_Agent(input_text: str):
    agent = create_agent()
    # Initialize with empty response key to be safe
    stream_iterator = agent.stream({"input": input_text, "history": [], "response": None}) 
    
    for chunk in stream_iterator:
        if "generate_response" in chunk:
            # This check ensures we have data before accessing ["response"]
            node_output = chunk["generate_response"]
            if node_output and "response" in node_output:
                response_stream = node_output["response"]
                for message_chunk in response_stream:
                    content = message_chunk.content
                    if content:
                        yield content