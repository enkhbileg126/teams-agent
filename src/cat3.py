import asyncio
import json
import os
import re
from typing import Any, Dict, List, TypedDict

from dotenv import load_dotenv

# Import for Vector Search
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langfuse import Langfuse, get_client, observe
from langgraph.graph import END, StateGraph

from src.cat1 import run_cat1
from src.cat2 import run_cat2

# --- 1. SETUP AND CONFIGURATION ---

load_dotenv()
if "GOOGLE_API_KEY" not in os.environ:
    raise ValueError("GOOGLE_API_KEY not found in .env file.")
langfuse = Langfuse(
    public_key=os.getenv('LANGFUSE_PUBLIC_KEY'),
    secret_key=os.getenv('LANGFUSE_SECRET_KEY'),
    host=os.getenv('LANGFUSE_HOST'),
)
langfuse = get_client()


def load_l3_categories(filepath: str) -> Dict[str, Dict[str, List[str]]]:
    """Loads the nested L1 -> L2 -> L3 categories from a JSON file."""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: The category file was not found at {filepath}")
        return {}
    except json.JSONDecodeError:
        print(f"Error: The file at {filepath} is not a valid JSON.")
        return {}


def extract_json_from_string(raw_string: str) -> str:
    """Uses regex to find and extract the first valid JSON object from a messy string."""
    match = re.search(r'\{.*\}', raw_string, re.DOTALL)
    if match:
        return match.group(0)
    return raw_string


# --- 2. VECTOR SEARCH CACHING & RETRIEVAL ---
# This section is crucial for performance. We build a vector store for each L2 category
# list ONCE and then cache it in memory to avoid re-computing it on every call.

all_l3_categories = load_l3_categories("data/cat3.json")
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vector_store_cache: Dict[str, VectorStoreRetriever] = {}


def get_retriever_for_category(l1: str, l2: str) -> VectorStoreRetriever | None:
    """
    Creates and caches a FAISS vector store retriever for a given L1/L2 category pair.
    """
    cache_key = f"{l1}::{l2}"
    if cache_key in vector_store_cache:
        return vector_store_cache[cache_key]

    # Navigate the nested dictionary to get the list of L3 categories
    l3_texts = all_l3_categories.get(l1, {}).get(l2)
    if not l3_texts:
        print(f"Warning: No L3 categories found for {l1} -> {l2}")
        return None

    try:
        # Create a new vector store and retriever for this specific list
        vectorstore = FAISS.from_texts(texts=l3_texts, embedding=embedding_model)
        retriever = vectorstore.as_retriever(
            search_kwargs={"k": 5}
        )  # Retrieve top 5 candidates
        vector_store_cache[cache_key] = retriever
        return retriever
    except Exception as e:
        print(f"Error creating vector store for {cache_key}: {e}")
        return None


# --- 3. LANGGRAPH STATE AND NODE DEFINITION ---


class GraphStateL3(TypedDict):
    chat_message: str
    category_l1: str
    category_l2: str
    category_l3_result: Dict[
        str, Any
    ]  # Will hold {"name": "...", "is_new": True/False}


def categorize_l3_node(state: GraphStateL3) -> dict:
    """
    Retrieves relevant candidates and uses an LLM to make a final L3 classification
    or generate a new category.
    """
    chat_message = state['chat_message']
    category_l1 = state['category_l1']
    category_l2 = state['category_l2']

    # Step 1: Retrieve top 5 semantically similar L3 candidates
    retriever = get_retriever_for_category(category_l1, category_l2)
    if not retriever:
        return {"category_l3_result": {"name": "L2_CATEGORY_NOT_FOUND", "is_new": True}}

    relevant_candidates = retriever.invoke(chat_message)
    candidate_list = [doc.page_content for doc in relevant_candidates]

    # Step 2: Reason with the LLM using the retrieved candidates
    prompt_template = """
    You are a highly precise technical support analyst. A ticket has been categorized as **{category_l1} -> {category_l2}**.
    Your final task is to assign the most specific issue or create a new one if no existing category is a perfect fit.

    **Support Ticket Text:**
    "{ticket_text}"

    **Analysis of similar past issues suggests these potential categories:**
    - {candidate_list}

    **Instructions:**
    1. Read the ticket and the candidate categories carefully.
    2. If the ticket's issue is a **perfect and specific match** for one of the candidates, choose that one.
    3. If **none** of the candidates are a good fit, you MUST create a new, concise (3-6 word) category name that accurately summarizes the specific, actionable issue.
    4. You MUST reply with ONLY a single, valid JSON object and nothing else. Use the structure provided below.

    **JSON Output Format (if you choose an existing category):**
    ```json
    {{
      "chosen_category": "THE_EXISTING_CATEGORY_NAME_YOU_CHOSE",
      "is_new": false,
      "reason": "A brief justification for why this category is a perfect match."
    }}
    ```

    **JSON Output Format (if you create a NEW category):**
    ```json
    {{
      "chosen_category": "YOUR_NEWLY_GENERATED_CATEGORY_NAME",
      "is_new": true,
      "reason": "A brief justification for why none of the existing categories were specific enough."
    }}
    ```
    """

    prompt = ChatPromptTemplate.from_template(prompt_template)
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash", temperature=0.1
    )  # Slightly higher temp for creativity if needed
    parser = StrOutputParser()
    chain = prompt | llm | parser

    result_str = chain.invoke(
        {
            "ticket_text": chat_message,
            "category_l1": category_l1,
            "category_l2": category_l2,
            "candidate_list": "\n- ".join(candidate_list),
        }
    )

    clean_json_str = extract_json_from_string(result_str)

    try:
        result_json = json.loads(clean_json_str)
        final_result = {
            "name": result_json.get("chosen_category", "PARSE_ERROR"),
            "is_new": result_json.get("is_new", True),
        }
    except (json.JSONDecodeError, AttributeError):
        print("Error: LLM did not return a valid JSON object.")
        final_result = {"name": "INVALID_JSON_RESPONSE", "is_new": True}

    return {"category_l3_result": final_result}


# --- 4. BUILD AND COMPILE THE GRAPH ---

workflow_l3 = StateGraph(GraphStateL3)
workflow_l3.add_node("categorize_l3", categorize_l3_node)
workflow_l3.set_entry_point("categorize_l3")
workflow_l3.add_edge("categorize_l3", END)
app_l3 = workflow_l3.compile()

# --- 5. EXPORTABLE ASYNC FUNCTION ---


async def run_cat3(input_from_l2: dict) -> dict:
    """Runs the Tier 3 categorization graph asynchronously."""
    if not all_l3_categories:
        raise RuntimeError(
            "L3 categories are not loaded. Check 'data/categories_l3.json'."
        )

    inputs = {
        "chat_message": input_from_l2["chat"],
        "category_l1": input_from_l2["category_1"],
        "category_l2": input_from_l2["category_2"],
    }

    final_state = await app_l3.ainvoke(inputs)

    output_json = {
        "category_1": final_state['category_l1'],
        "category_2": final_state['category_l2'],
        "category_3": final_state['category_l3_result'],  # This is already a dict
        "chat": final_state['chat_message'],
    }

    return output_json


@observe
async def categorize(ticket: str) -> dict:
    first = await run_cat1(ticket)
    second = await run_cat2(first)
    final = await run_cat3(second)
    langfuse.update_current_trace(name="categorizer", metadata="")
    langfuse.flush()
    return final
