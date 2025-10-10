from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langfuse import get_client, observe
from langfuse.langchain import CallbackHandler

from src.settings import sequential_agent_llm

langfuse = get_client()


@observe
async def run_agent_1_domain(ticket: str, domains: list[str]) -> str:
    prompt = ChatPromptTemplate.from_template(
        """
        You are an expert ticket classifier. Your task is to categorize the following support ticket into ONE of the predefined domains.

        **Support Ticket Text:**
        "{ticket}"

        **Available Domains:**
        - {domains}

        Analyze the ticket and respond with ONLY the name of the single most appropriate domain from the list.
        """
    )
    handler = CallbackHandler()
    llm = sequential_agent_llm
    chain = prompt | llm | StrOutputParser()

    result = await chain.ainvoke(
        {"ticket": ticket, "domains": "\n- ".join(domains)},
        config={"callbacks": [handler]},
    )
    return result.strip()
