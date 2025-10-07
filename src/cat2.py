import asyncio
import json
import os
import re
from typing import Dict, List, TypedDict

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langfuse import observe
from langgraph.graph import END, StateGraph

load_dotenv()
if "GOOGLE_API_KEY" not in os.environ:
    raise ValueError("GOOGLE_API_KEY not found in .env file.")


def load_l2_categories(filepath: str) -> Dict[str, List[str]]:
    """Loads the nested L2 categories from a JSON file."""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: The category file was not found at {filepath}")
        return {}
    except json.JSONDecodeError:
        print(f"Error: The file at {filepath} is not a valid JSON.")
        return {}


# --- 2. LANGGRAPH STATE AND NODE DEFINITION ---
def extract_json_from_string(raw_string: str) -> str:
    """
    Uses regex to find and extract the first valid JSON object within a string.
    This is useful for cleaning LLM output that includes markdown code blocks.

    Args:
        raw_string: The potentially messy string from the LLM.

    Returns:
        A clean string containing only the JSON object, or the original string if no JSON is found.
    """
    # This regex looks for a substring that starts with '{' and ends with '}'
    # re.DOTALL makes '.' match newlines, which is crucial for formatted JSON
    match = re.search(r'\{.*\}', raw_string, re.DOTALL)

    if match:
        # If a match is found, return the matched group (the JSON object)
        return match.group(0)

    # If no JSON object is found, return the original string
    return raw_string


class GraphStateL2(TypedDict):
    """Represents the state for the Tier 2 categorization graph."""

    chat_message: str
    category_l1: str
    category_l2: str


# Load category data globally within the module so it's ready for the node
all_l2_categories = load_l2_categories("data/cat2.json")


def categorize_l2_node(state: GraphStateL2) -> dict:
    """Invokes an LLM to categorize the ticket into a Tier 2 category."""

    chat_message = state['chat_message']
    category_l1 = state['category_l1']

    l2_options = all_l2_categories.get(category_l1)
    if not l2_options:
        print(f"Warning: L1 category '{category_l1}' not found in L2 category map.")
        return {"category_l2": "L1_CATEGORY_NOT_FOUND"}

    prompt_template = """
    You are a precise ticket classification engine. A ticket has already been classified into the `{category_l1}` domain.
    Your task is to determine the specific problem category from the list provided and return a single, valid JSON object.

    **Support Ticket Text:**
    "{ticket_text}"

    **Available Categories for `{category_l1}`:**
    - {l2_options}

    **Instructions:**
    1. Analyze the ticket text.
    2. Choose the single most appropriate category from the "Available Categories" list.
    3. You MUST reply with ONLY a single, valid JSON object and nothing else. Do not add any text before or after the JSON.
    4. Use the following structure, filling in the `category_2` value with your choice.

    ```json
    {{
      "category_1": "{category_l1}",
      "category_2": "YOUR_CHOSEN_CATEGORY_HERE",
      "chat": "{ticket_text_escaped}"
    }}
    ```
    """

    prompt = ChatPromptTemplate.from_template(prompt_template)
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash", temperature=0
    )  # Using flash for speed/cost
    parser = StrOutputParser()
    chain = prompt | llm | parser

    escaped_chat = chat_message.replace('"', '\\"').replace('\n', '\\n')

    result_str = chain.invoke(
        {
            "ticket_text": chat_message,
            "ticket_text_escaped": escaped_chat,
            "category_l1": category_l1,
            "l2_options": "\n- ".join(l2_options),
        }
    )
    # *** APPLY THE REGEX FIX HERE ***
    clean_json_str = extract_json_from_string(result_str)

    try:
        # Parse the cleaned string instead of the raw one
        result_json = json.loads(clean_json_str)
        final_category_l2 = result_json.get("category_2", "PARSE_ERROR")
    except (json.JSONDecodeError, AttributeError):
        print("Error: Failed to parse JSON even after cleaning.")
        final_category_l2 = "INVALID_JSON_RESPONSE"

    return {"category_l2": final_category_l2}


# --- 3. BUILD AND COMPILE THE GRAPH ---

workflow_l2 = StateGraph(GraphStateL2)
workflow_l2.add_node("categorize_l2", categorize_l2_node)
workflow_l2.set_entry_point("categorize_l2")
workflow_l2.add_edge("categorize_l2", END)
app_l2 = workflow_l2.compile()


# --- 4. EXPORTABLE ASYNC FUNCTION ---


@observe
async def run_cat2(input_from_l1: dict) -> dict:
    """
    Runs the Tier 2 categorization graph asynchronously.

    Args:
        input_from_l1: A dictionary containing 'category_l1' and 'chat'.
                       Example: {"category_l1": "IPTV", "chat": "My TV is not working"}

    Returns:
        A dictionary with the final, structured output.
    """
    if "category_l1" not in input_from_l1 or "chat" not in input_from_l1:
        raise ValueError("Input dictionary must contain 'category_l1' and 'chat' keys.")

    if not all_l2_categories:
        raise RuntimeError(
            "L2 categories are not loaded. Check 'data/categories_l2_new.json'."
        )

    inputs = {
        "chat_message": input_from_l1["chat"],
        "category_l1": input_from_l1["category_l1"],
    }

    final_state = await app_l2.ainvoke(inputs)

    output_json = {
        "category_1": final_state['category_l1'],
        "category_2": final_state['category_l2'],
        "chat": final_state['chat_message'],
    }

    return output_json
