import json
import os

import google.generativeai as genai
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()
# --- Configuration ---
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
ALLOWED_DOMAIN = "unegui.mn"  # Changed the domain to our new target

# --- Gemini Model Initialization ---
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    print("WARNING: GEMINI_API_KEY is not set. The agent will not function.")

# --- Main Functions ---


def generate_search_query_with_gemini(user_input: str) -> str:
    """
    Uses Gemini to convert a natural language question into an optimal product search query.
    """
    print("-> Asking Gemini to generate an effective search query...")

    if not GEMINI_API_KEY:
        return user_input

    try:
        model = genai.GenerativeModel('gemini-2.5-flash')

        # Updated prompt for product searching on a classifieds site
        prompt = f"""
        You are an expert search query creator for a classifieds website like unegui.mn.
        Your task is to convert a user's question about a product into a concise Google search query.
        Extract only the essential keywords like the product name, model, brand, condition, and location.
        Do not include conversational words like "find me", "are there", "how much", etc.
        The output must be ONLY the search query string and nothing else.

        Here are some examples:
        User question: "Find me a used Toyota Prius from 2018 in Ulaanbaatar."
        Your output: used 2018 Toyota Prius Ulaanbaatar

        User question: "Are there any 4-person tents for sale?"
        Your output: 4 person tent for sale

        User question: "How much does a 256GB iPhone 13 Pro Max cost on the site?"
        Your output: iPhone 13 Pro Max 256GB price

        Now, generate a search query for the following user question:
        User question: "{user_input}"
        Your output:
        """

        response = model.generate_content(prompt)
        search_query = response.text.strip().replace('\n', ' ')
        print(f"-> Gemini generated query: '{search_query}'")
        return search_query

    except Exception as e:
        print(f"ERROR: Could not generate search query with Gemini: {e}")
        return user_input


def search_for_products(query: str) -> list:
    """
    Searches for a product query, strictly limited to the ALLOWED_DOMAIN.
    """
    print(f"-> Searching for: '{query}' on {ALLOWED_DOMAIN}...")

    if not SERPER_API_KEY:
        print("ERROR: SERPER_API_KEY environment variable not set.")
        return []

    url = "https://google.serper.dev/search"
    search_payload = json.dumps({"q": f"{query} site:{ALLOWED_DOMAIN}"})
    headers = {'X-API-KEY': SERPER_API_KEY, 'Content-Type': 'application/json'}

    try:
        response = requests.post(url, headers=headers, data=search_payload)
        response.raise_for_status()
        results = response.json()

        if "organic" in results:
            print(f"-> Found {len(results['organic'])} potential results.")
            return results['organic']
        else:
            print("-> No organic results found.")
            return []

    except requests.exceptions.RequestException as e:
        print(f"ERROR: An error occurred during the search request: {e}")
        return []


def scrape_page_content(url: str) -> str:
    """
    Scrapes the text content from a given URL using BeautifulSoup.
    """
    print(f"-> Reading content from: {url}...")
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        # This generic cleaner is usually effective for classifieds sites
        for element in soup(
            ["script", "style", "header", "footer", "nav", "aside", "form"]
        ):
            element.decompose()

        text = soup.get_text(separator='\n', strip=True)
        print("-> Successfully extracted page content.")
        return text

    except requests.exceptions.RequestException as e:
        print(f"ERROR: Could not retrieve content from the URL: {e}")
        return None


def get_answer_from_gemini(context: str, question: str) -> str:
    """
    Uses the Gemini API to answer a user's question based on the scraped product info.
    """
    print("-> Asking Gemini for the final summary...")

    if not GEMINI_API_KEY:
        return "ERROR: GEMINI_API_KEY environment variable not set."

    try:
        model = genai.GenerativeModel('gemini-2.5-flash')

        # Updated prompt to summarize product details
        prompt = f"""
        Based *only* on the following text from a product listing on unegui.mn, please provide a clear and concise answer to the user's original question.
        Extract key details like product name, price, condition, description, and location.
        If the information is not in the text, clearly state that. Do not make up information.
        Format the key details using bullet points.

        --- TEXT CONTENT ---
        {context}
        --- END OF TEXT CONTENT ---

        USER'S ORIGINAL QUESTION: "{question}"

        ANSWER:
        """
        response = model.generate_content(prompt)
        return response.text

    except Exception as e:
        return f"ERROR: An error occurred with the Gemini API: {e}"


def run_agent():
    """
    Main loop for the intelligent product search agent.
    """
    print("--- Uneguui.mn Intelligent Product Search Agent ---")
    print("Type 'quit' or 'exit' to stop.")

    while True:
        user_query = input("\nAsk me about a product on Uneguui.mn: ")
        if user_query.lower() in ['quit', 'exit']:
            break

        search_query = generate_search_query_with_gemini(user_query)
        search_results = search_for_products(search_query)

        if not search_results:
            print(
                "\nI couldn't find any relevant product listings for your query on Uneguui.mn."
            )
            continue

        top_result = search_results[0]
        page_url = top_result.get('link')

        if not page_url:
            print("\nSearch result did not contain a valid URL.")
            continue

        content = scrape_page_content(page_url)

        if not content or len(content) < 50:
            print("\nI was unable to read the main content of the product page.")
            continue

        final_answer = get_answer_from_gemini(content, user_query)

        print("\n--- Agent's Answer ---")
        print(final_answer)
        print(f"\nSource: {page_url}")
        print("----------------------")


if __name__ == "__main__":
    if not SERPER_API_KEY or not GEMINI_API_KEY:
        print(
            "FATAL ERROR: Make sure you have set both SERPER_API_KEY and GEMINI_API_KEY environment variables before running."
        )
    else:
        run_agent()
