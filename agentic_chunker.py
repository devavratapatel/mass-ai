import uuid
import os
from typing import Optional, List, Dict
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from rich import print

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser

load_dotenv()

class AgenticChunker:
    def __init__(self, google_api_key: Optional[str] = None):
        self.chunks: Dict = {}
        self.id_truncate_limit = 5

        self.generate_new_metadata_ind = True
        self.print_logging = True

        if google_api_key is None:
            google_api_key = os.getenv("GOOGLE_API_KEY")

        if google_api_key is None:
            raise ValueError("API key is not provided and GOOGLE_API_KEY not found in environment variables")

        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=google_api_key,
            temperature=0,
            max_retries=5,
        )

    def add_propositions(self, propositions: List[str]):
        for proposition in propositions:
            self.add_proposition(proposition)
    
    def add_proposition(self, proposition: str):
        if self.print_logging:
            print(f"\nAdding: '{proposition}'")

        if len(self.chunks) == 0:
            if self.print_logging:
                print("No chunks, creating a new one")
            self._create_new_chunk(proposition)
            return

        chunk_id = self._find_relevant_chunk(proposition)

        if chunk_id:
            if self.print_logging:
                print(f"Chunk Found ({self.chunks[chunk_id]['chunk_id']}), adding to: {self.chunks[chunk_id]['title']}")
            self.add_proposition_to_chunk(chunk_id, proposition)
            return
        else:
            if self.print_logging:
                print("No chunks found")
            self._create_new_chunk(proposition)
        

    def add_proposition_to_chunk(self, chunk_id, proposition):
        self.chunks[chunk_id]['propositions'].append(proposition)

        if self.generate_new_metadata_ind:
            self.chunks[chunk_id]['summary'] = self._update_chunk_summary(self.chunks[chunk_id])
            self.chunks[chunk_id]['title'] = self._update_chunk_title(self.chunks[chunk_id])

    def _update_chunk_summary(self, chunk):
        PROMPT = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    You are the steward of a group of chunks.
                    A new proposition was just added to one of your chunks, generate a very brief 1-sentence summary.
                    Input: Proposition: Greg likes to eat pizza
                    Output: This chunk contains information about the types of food Greg likes to eat.
                    Only respond with the chunk new summary, nothing else.
                    """,
                ),
                ("user", "Chunk's propositions:\n{proposition}\n\nCurrent chunk summary:\n{current_summary}"),
            ]
        )
        runnable = PROMPT | self.llm | StrOutputParser()
        return runnable.invoke({
            "proposition": "\n".join(chunk['propositions']),
            "current_summary" : chunk['summary']
        })
    
    def _update_chunk_title(self, chunk):
        PROMPT = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    You are the steward of a group of chunks.
                    Generate a very brief updated chunk title.
                    Input: Summary: This chunk is about dates and times that the author talks about
                    Output: Date & Times
                    Only respond with the new chunk title, nothing else.
                    """,
                ),
                ("user", "Chunk's propositions:\n{proposition}\n\nChunk summary:\n{current_summary}\n\nCurrent chunk title:\n{current_title}"),
            ]
        )
        runnable = PROMPT | self.llm | StrOutputParser()
        return runnable.invoke({
            "proposition": "\n".join(chunk['propositions']),
            "current_summary" : chunk['summary'],
            "current_title" : chunk['title']
        })

    def _get_new_chunk_summary(self, proposition):
        PROMPT = ChatPromptTemplate.from_messages(
            [
                ("system", "Generate a 1-sentence summary for this proposition's new chunk."),
                ("user", "Proposition:\n{proposition}"),
            ]
        )
        runnable = PROMPT | self.llm | StrOutputParser()
        return runnable.invoke({"proposition": proposition})
    
    def _get_new_chunk_title(self, summary):
        PROMPT = ChatPromptTemplate.from_messages(
            [
                ("system", "Generate a brief title for this chunk based on the summary."),
                ("user", "Summary:\n{summary}"),
            ]
        )
        runnable = PROMPT | self.llm | StrOutputParser()
        return runnable.invoke({"summary": summary})

    def _create_new_chunk(self, proposition):
        new_chunk_id = str(uuid.uuid4())[:self.id_truncate_limit]
        new_chunk_summary = self._get_new_chunk_summary(proposition)
        new_chunk_title = self._get_new_chunk_title(new_chunk_summary)

        self.chunks[new_chunk_id] = {
            'chunk_id' : new_chunk_id,
            'propositions': [proposition],
            'title' : new_chunk_title,
            'summary': new_chunk_summary,
            'chunk_index' : len(self.chunks)
        }
        if self.print_logging:
            print(f"Created new chunk ({new_chunk_id}): {new_chunk_title}")
    
    def get_chunk_outline(self):
        chunk_outline = ""
        for chunk_id, chunk in self.chunks.items():
            single_chunk_string = f"""Chunk ID: {chunk['chunk_id']}\nTitle: {chunk['title']}\nSummary: {chunk['summary']}\n\n"""
            chunk_outline += single_chunk_string
        return chunk_outline

    def _find_relevant_chunk(self, proposition):
        current_chunk_outline = self.get_chunk_outline()

        class ChunkDecision(BaseModel):
            chunk_id: Optional[str] = Field(description="The chunk ID if a relevant chunk is found, otherwise null/None")

        parser = PydanticOutputParser(pydantic_object=ChunkDecision)

        PROMPT = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    Determine if the proposition belongs to any existing chunk.
                    Return the chunk_id if it matches, otherwise null.
                    {format_instructions}
                    """,
                ),
                ("user", "Current Chunks:\n{current_chunk_outline}\n\nProposition:\n{proposition}"),
            ]
        )

        prompt_with_format = PROMPT.partial(format_instructions=parser.get_format_instructions())
        runnable = prompt_with_format | self.llm | parser

        try:
            decision = runnable.invoke({
                "proposition": proposition,
                "current_chunk_outline": current_chunk_outline
            })
            if decision.chunk_id and decision.chunk_id not in self.chunks:
                return None
            return decision.chunk_id
        except Exception:
            return None
    
    def get_chunks(self, get_type='dict'):
        if get_type == 'dict':
            return self.chunks
        if get_type == 'list_of_strings':
            chunks = []
            for chunk_id, chunk in self.chunks.items():
                chunks.append(" ".join([x for x in chunk['propositions']]))
            return chunks
    
    def pretty_print_chunks(self):
        print(f"\nYou have {len(self.chunks)} chunks\n")
        for chunk_id, chunk in self.chunks.items():
            print(f"Chunk #{chunk['chunk_index']} ({chunk['title']})")