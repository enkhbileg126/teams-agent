from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langfuse import get_client, observe
from langfuse.langchain import CallbackHandler

from src.settings import sequential_agent_llm

langfuse = get_client()


@observe
async def run_agent_3_cat2(
    ticket: str, domain: str, cat1: str, cat2_options: list[str]
) -> str:
    prompt = ChatPromptTemplate.from_template(
        """
        A support ticket is classified as **{domain} -> {cat1}**.
        Your task is to choose the best Tier 2 category.

        **Support Ticket:**
        {ticket}

        **Available Tier 2 Categories:**
        - {cat2_options}

        Respond with ONLY the name of the single most appropriate category from the list.
        """
    )
    llm = sequential_agent_llm
    handler = CallbackHandler()

    chain = prompt | llm | StrOutputParser()

    result = await chain.ainvoke(
        {
            "ticket": ticket,
            "domain": domain,
            "cat1": cat1,
            "cat2_options": "\n- ".join(cat2_options),
        },
        config={"callbacks": [handler]},
    )
    return result.strip()
