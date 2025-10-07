import os
from typing import Annotated, List, TypedDict

import dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langfuse import Langfuse, get_client, observe
from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

dotenv.load_dotenv()
langfuse = get_client()

gemini_llm = os.getenv("GOOGLE_API_KEY")
if not gemini_llm:
    raise ValueError("GOOGLE_API_KEY environment variable not set.")
# Initialize Langfuse
langfuse = Langfuse(
    public_key=os.getenv('LANGFUSE_PUBLIC_KEY'),
    secret_key=os.getenv('LANGFUSE_SECRET_KEY'),
    host=os.getenv('LANGFUSE_HOST'),
)


def load_categories(filepath: str) -> List[str]:
    """Loads categories from a text file, one per line."""
    try:
        with open(filepath, 'r') as f:
            return [line.strip() for line in f.readlines() if line.strip()]
    except FileNotFoundError:
        print(f"Error: The category file was not found at {filepath}")
        print("Please make sure the file exists and the path is correct.")
        return []


# --- 2. Define the State for the Graph ---


# The state is the "memory" of our agent. It's what gets passed between nodes.
class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        chat: The initial ticket text.
        categories: The list of possible L1 categories.
        category_l1: The final categorized L1 label.
    """

    chat: str
    categories: List[str]
    category_l1: str


# --- 3. Define the Nodes of the Graph ---


def categorize_l1(state: GraphState) -> dict:
    """
    This node invokes an LLM to categorize the ticket into a Tier 1 category.
    """

    chat = state['chat']
    categories = state['categories']

    # This is a crucial prompt engineering step.
    # We constrain the LLM to only choose from the list we provide.
    prompt_template = """
    You are an expert ticket classifier. Your task is to categorize the following support ticket into ONE of the predefined domains.

    **Support Ticket Text:**
    "{ticket_text}"

    **Available Domains:**
    - {categories}

    Analyze the ticket and respond with ONLY the name of the single most appropriate domain from the list above. Do not add any explanation or punctuation.
    """

    prompt = ChatPromptTemplate.from_template(prompt_template)

    # Initialize the LLM. Using a low temperature for classification tasks is best practice.
    # We use gemini-1.5-flash for its speed and low cost, as requested.
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)

    # The output parser ensures we just get the text string back from the LLM.
    parser = StrOutputParser()

    # Create the chain of operations
    chain = prompt | llm | parser

    # Invoke the chain with the required inputs
    result = chain.invoke(
        {
            "ticket_text": chat,
            "categories": "\n- ".join(
                categories
            ),  # Format for better readability in the prompt
        }
    )

    # Clean the output just in case the LLM adds extra whitespace
    clean_result = result.strip()

    # Update the state with the result
    return {"category_l1": clean_result}


# --- 4. Build and Compile the Graph ---

# Initialize the state graph
workflow = StateGraph(GraphState)

# Add the node(s) to the graph
workflow.add_node("categorize_l1", categorize_l1)

# Set the entry point of the graph
workflow.set_entry_point("categorize_l1")

# Add the final edge: after categorization, the process is done.
workflow.add_edge("categorize_l1", END)

# Compile the graph into a runnable application
app = workflow.compile()


# --- 5. Run the Agent ---
cat1 = load_categories("data/cat1.txt")

ticket = "Hi, my mobile data is suddenly super slow, I can't even stream music. I've tried restarting my phone but it didn't help. Can you check my account?"


@observe
async def run_cat1(ticket: str):

    inputs = {"chat": ticket, "categories": cat1}

    # Run the graph
    final_state = app.invoke(inputs)

    # Print the final result
    if final_state:
        return final_state
