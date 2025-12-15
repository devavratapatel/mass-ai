from langgraph.graph import StateGraph, START, END
from typing import TypedDict, List
import os
import time
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai

load_dotenv()

# --- 1. Configure Google GenAI directly ---
# We use the raw SDK to upload files, as it's cleaner than LangChain for this specific task
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# Global cache for the uploaded file handle
UPLOADED_FILE = None

def get_file_handle():
    """Uploads the PDF to Google and returns the file handle."""
    global UPLOADED_FILE
    if UPLOADED_FILE:
        return UPLOADED_FILE
    
    # Check for your PDF file (Adjust filename as needed)
    # We prioritize the largest file or merge them if you prefer, 
    # but for simplicity, let's load the main one.
    target_file = None
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    pdf1 = os.path.join(BASE_DIR, "pdf1.pdf")
    if os.path.exists(pdf1):
        target_file = pdf1
    
    # Note: If you have multiple files, Gemini supports uploading multiple,
    # but let's start with one to fix the crash.
    
    if target_file:
        print(f"Uploading {target_file} to Google...")
        # This uploads the file to Google's cloud. 0 RAM used on Render.
        myfile = genai.upload_file(target_file)
        
        # Wait for processing to complete
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

class AgentState(TypedDict):
    input: str
    history: List[BaseMessage]

def generate_response(state: AgentState):
    file_handle = get_file_handle()
    
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
    ]
    
    # If we have a file, we send it as a 'media' block
    if file_handle:
        # LangChain-Google-GenAI specific format for passing file handles
        # We construct a message that contains the file reference
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

    # Add history
    messages.extend(state.get("history", []))

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
    stream_iterator = agent.stream({"input": input_text, "history": []}) 
    
    for chunk in stream_iterator:
        if "generate_response" in chunk:
            response_stream = chunk["generate_response"]["response"]
            for message_chunk in response_stream:
                content = message_chunk.content
                if content:
                    yield content