import os
from typing import List, TypedDict

import dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph

# Load environment variables from .env file
dotenv.load_dotenv()

# --- 1. Initialize the Language Model ---
# Ensure the Google API key is set
if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY environment variable not set.")

# Use Gemini 2.0 Flash as requested
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)


# --- 2. Define the State for the Graph ---
class GraphState(TypedDict):
    """
    Represents the state of our categorization graph.

    Attributes:
        conversation: The initial ticket conversation.
        domain: The identified domain.
        ticket_type: The identified ticket type.
        category_l1: The identified Level 1 category.
        category_l2: The identified Level 2 category.
        category_l3: The final identified Level 3 problem.
    """

    conversation: str
    domain: str
    ticket_type: str
    category_l1: str
    category_l2: str
    category_l3: str


# --- 3. Define Categorizer Nodes ---


# Agent 1: Domain and Ticket Type
async def run_domain_and_ticket_type(state: GraphState) -> dict:
    """Categorizes the conversation into a domain and ticket type."""
    conversation = state['conversation']
    domain_list = [
        "Mobile",
        "IPTV",
        "LookTV",
        "Premium",
        "Toki",
        "Outbound",
        "MDealer",
        "Corp-IPTV",
        "Ger Internet",
        "U-Point",
    ]
    ticket_type_list = ["complaint", "order", "service"]

    prompt = ChatPromptTemplate.from_template(
        """
        You are an expert ticket classifier. Your task is to identify the domain and ticket type from the following support ticket.

        **Support Ticket Text:**
        "{conversation}"

        **Available Domains:**
        - {domains}

        **Available Ticket Types:**
        - {ticket_types}

        Analyze the ticket and respond with the domain and ticket type in the format 'Domain: [domain_name], Ticket Type: [ticket_type_name]'.
        Choose only from the provided lists.
        """
    )
    chain = prompt | llm | StrOutputParser()
    result = await chain.ainvoke(
        {
            "conversation": conversation,
            "domains": "\n- ".join(domain_list),
            "ticket_types": "\n- ".join(ticket_type_list),
        }
    )

    # Simple parsing of the output
    domain = result.split("Domain:")[1].split(",")[0].strip()
    ticket_type = result.split("Ticket Type:")[1].strip()

    print(f"Agent 1 -> Domain: {domain}, Ticket Type: {ticket_type}")
    return {"domain": domain, "ticket_type": ticket_type}


# Agent 2: Category 1
async def run_cat1(state: GraphState) -> dict:
    """Categorizes the conversation into a Level 1 category."""
    conversation = state['conversation']
    domain = state['domain']
    # In a real system, you would fetch these categories based on the domain
    cat1_list = ["Data", "Voice", "SMS", "Billing", "Device"]

    prompt = ChatPromptTemplate.from_template(
        """
        Given the ticket conversation and its domain, classify it into one of the following broad categories.

        **Support Ticket Text:**
        "{conversation}"

        **Domain:**
        {domain}

        **Available Categories:**
        - {categories}

        Respond with ONLY the name of the single most appropriate category.
        """
    )
    chain = prompt | llm | StrOutputParser()
    result = await chain.ainvoke(
        {
            "conversation": conversation,
            "domain": domain,
            "categories": "\n- ".join(cat1_list),
        }
    )
    clean_result = result.strip()
    print(f"Agent 2 -> Category 1: {clean_result}")
    return {"category_l1": clean_result}


# Agent 3: Category 2
async def run_cat2(state: GraphState) -> dict:
    """Categorizes the conversation into a Level 2 category."""
    conversation = state['conversation']
    category_l1 = state['category_l1']
    # Categories would be fetched based on category_l1
    cat2_list = [
        "Regarding data consumption",
        "Slow data speed",
        "No data connection",
        "International roaming data",
    ]

    prompt = ChatPromptTemplate.from_template(
        """
        Given the ticket conversation and its broad category, identify the type of problem the user is experiencing.

        **Support Ticket Text:**
        "{conversation}"

        **Broad Category:**
        {category_l1}

        **Available Problem Types:**
        - {categories}

        Respond with ONLY the name of the single most appropriate problem type.
        """
    )
    chain = prompt | llm | StrOutputParser()
    result = await chain.ainvoke(
        {
            "conversation": conversation,
            "category_l1": category_l1,
            "categories": "\n- ".join(cat2_list),
        }
    )
    clean_result = result.strip()
    print(f"Agent 3 -> Category 2: {clean_result}")
    return {"category_l2": clean_result}


# Agent 4: Category 3 (Using Embeddings - Simplified for this example)
async def run_cat3(state: GraphState) -> dict:
    """Identifies the exact problem from a list using a simplified matching approach."""
    # In a real implementation, you would use a vector store (e.g., FAISS, Chroma)
    # and embeddings for this step.
    conversation = state['conversation']
    category_l2 = state['category_l2']

    # This list would come from your nested JSON, filtered by previous categories
    cat3_list = [
        "Cannot see consumption log",
        "Data usage not updating in real-time",
        "Incorrect data usage reported",
    ]

    # For this example, we'll use another LLM call to simulate vector search.
    # This is less efficient but demonstrates the principle.
    prompt = ChatPromptTemplate.from_template(
        """
        You are a precise problem identifier. Based on the user's conversation and the problem type,
        select the exact problem from the list below that best matches the user's issue.

        **Support Ticket Text:**
        "{conversation}"

        **Problem Type:**
        {category_l2}

        **Exact Problem List:**
        - {categories}

        Respond with ONLY the exact problem description from the list.
        """
    )
    chain = prompt | llm | StrOutputParser()
    result = await chain.ainvoke(
        {
            "conversation": conversation,
            "category_l2": category_l2,
            "categories": "\n- ".join(cat3_list),
        }
    )
    clean_result = result.strip()
    print(f"Agent 4 -> Category 3: {clean_result}")
    return {"category_l3": clean_result}


# --- 4. Build and Compile the Graph ---
workflow = StateGraph(GraphState)

# Add the nodes to the graph
workflow.add_node("domain_ticket_type", run_domain_and_ticket_type)
workflow.add_node("category_1", run_cat1)
workflow.add_node("category_2", run_cat2)
workflow.add_node("category_3", run_cat3)

# Define the sequence of execution
workflow.set_entry_point("domain_ticket_type")
workflow.add_edge("domain_ticket_type", "category_1")
workflow.add_edge("category_1", "category_2")
workflow.add_edge("category_2", "category_3")
workflow.add_edge("category_3", END)


# Compile the graph
app = workflow.compile()


# --- 5. Run the Categorizer ---
async def run_categorizer(ticket: str):
    """
    Runs the full categorization pipeline for a given ticket.
    """
    inputs = {"conversation": ticket}
    final_state = await app.ainvoke(inputs)
    return final_state


# Example Usage
if __name__ == "__main__":
    import asyncio

    example_ticket = "Hi, my mobile data is suddenly super slow, and I can't even stream music. I also noticed I can't find the page to see my data usage log. Can you check my account?"

    final_categorization = asyncio.run(run_categorizer(example_ticket))

    print("\n--- Final Categorization ---")
    print(f"Domain: {final_categorization.get('domain')}")
    print(f"Ticket Type: {final_categorization.get('ticket_type')}")
    print(f"Category 1: {final_categorization.get('category_l1')}")
    print(f"Category 2: {final_categorization.get('category_l2')}")
    print(f"Category 3: {final_categorization.get('category_l3')}")
