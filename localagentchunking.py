import os
import sys
import time
from typing import Any, Dict, TypedDict, List, Optional
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic import hub
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser

from agentic_chunker import AgenticChunker

load_dotenv()

SYSTEM_PROMPT = """You are a helpful RAG assistant. 
Use the provided context to answer the user's question. 
If the context doesn't contain the answer, say you don't know."""

HUMAN_PROMPT = """
Context:
{context}

Question: 
{input}
"""

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    print("WARNING: GOOGLE_API_KEY not found in .env.")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=api_key,
    temperature=0.3,
    max_retries=2
)

class Sentences(BaseModel):
    sentences: List[str] = Field(description="List of distinct propositions/sentences derived from the text.")

parser = PydanticOutputParser(pydantic_object=Sentences)

proposition_prompt = ChatPromptTemplate.from_messages([
    ("system", """
    Decompose the following text into clear, concise propositions (simple sentences). 
    Each sentence should contain a single atomic fact.
    {format_instructions}
    """),
    ("user", "{input}")
]).partial(format_instructions=parser.get_format_instructions())

proposition_chain = proposition_prompt | llm | parser

def get_propositions(text: str) -> List[str]:
    try:
        result = proposition_chain.invoke({"input": text})
        return result.sentences
    except Exception as e:
        print(f"Error getting propositions: {e}")
        return [text] 

class AgentState(TypedDict):
    input: str
    context: Dict[str, Any]
    history: List[BaseMessage]
    response: Any

retriever = None

try:
    pdf_files = ["pdf1.pdf", "pdf2.pdf"]
    existing_pdfs = [f for f in pdf_files if os.path.exists(f)]

    if existing_pdfs:
        print(f"Found PDFs: {existing_pdfs}. Processing with Agentic Chunker (this may take time)...")
        documents = []
        for pdf in existing_pdfs:
            loader = PyPDFLoader(pdf)
            documents.extend(loader.load())
        
        all_propositions = []
        print("Extracting propositions from documents...")
        for doc in documents:
            paragraphs = doc.page_content.split("\n\n")
            for i, para in enumerate(paragraphs):
                if len(para.strip()) > 20: 
                    print(f"Processing paragraph {i+1}...")
                    props = get_propositions(para)
                    all_propositions.extend(props)
                    
                    time.sleep(0.5) 

        print(f"Grouping {len(all_propositions)} propositions...")
        ac = AgenticChunker() 
        
        try:
            ac.add_propositions(all_propositions)
            ac.pretty_print_chunks()
            chunks = ac.get_chunks(get_type='list_of_strings')
        except Exception as e:
            print(f"Agentic Chunker failed: {e}. Using flat propositions.")
            chunks = all_propositions

        rag_docs = [Document(page_content=chunk, metadata={"source": "agentic_chunker"}) for chunk in chunks]

        try:
            print("Initializing Local Embeddings (all-MiniLM-L6-v2)...")
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        except ImportError:
            print("Error: 'sentence-transformers' not installed. Run: pip install sentence-transformers")
            raise

        vectorstore = Chroma.from_documents(documents=rag_docs, embedding=embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        print("RAG System Initialized Successfully.")
    else:
        print("Warning: No PDF files found (pdf1.pdf, pdf2.pdf). Using Dummy Retriever.")
        raise FileNotFoundError("No PDFs")

except Exception as e:
    print(f"RAG initialization skipped/failed: {e}")
    print("Using a dummy retriever for testing purposes.")
    
    class DummyRetriever:
        def invoke(self, query):
            return [Document(page_content="This is dummy context because the RAG system failed to load.")]
    
    retriever = DummyRetriever()

def retrieve_context(state: AgentState):
    query = state["input"]
    try:
        docs = retriever.invoke(query)
        context = "\n\n".join([doc.page_content for doc in docs])
    except Exception as e:
        context = "Error retrieving context."
    return {"context": context}

def generate_response(state: AgentState):
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="history"),
        HumanMessage(content=HUMAN_PROMPT.replace("{input}", state["input"]).replace("{context}", str(state["context"])))
    ])

    chain = prompt | llm

    response_stream = chain.stream({"history": state.get("history", [])})
    return {"response": response_stream}

def create_agent():
    graph = StateGraph(AgentState)
    graph.add_node("retrieve_context", retrieve_context)
    graph.add_node("generate_response", generate_response)
    
    graph.add_edge(START, "retrieve_context")
    graph.add_edge("retrieve_context", "generate_response")
    graph.add_edge("generate_response", END)
    
    return graph.compile()

def run_Agent(input_text: str):
    agent = create_agent()
    initial_state = {"input": input_text, "history": []}
    
    stream_iterator = agent.stream(initial_state)
    
    for chunk in stream_iterator:
        if "generate_response" in chunk:
            response_stream = chunk["generate_response"]["response"]
            for token in response_stream:
                if hasattr(token, 'content'):
                     yield token.content
                else:
                     yield str(token)

if __name__ == "__main__":
    while True:
        try:
            user_input = input("\033[1;34mYou:\033[0m ")
            if user_input.lower() in ["quit", "exit", "q"]:
                break
            
            print("\033[1;32mAgent:\033[0m ", end="", flush=True)
            for text_chunk in run_Agent(user_input):
                print(text_chunk, end="", flush=True)
            print("\n")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")