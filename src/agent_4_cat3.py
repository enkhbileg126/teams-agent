from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langfuse import get_client, observe
from langfuse.langchain import CallbackHandler

from src.settings import sequential_agent_llm

langfuse = get_client()


@observe
async def run_agent_4_cat3(
    ticket: str, domain: str, cat1: str, cat2: str, cat3_options: list[str]
) -> str:

    # NOTE: This is a simple classifier. For higher accuracy on this final, granular
    # step, this agent is the prime candidate to be upgraded with a semantic
    # retriever, as we discussed previously.

    prompt = ChatPromptTemplate.from_template(
        """
        A support ticket is classified as **{domain} -> {cat1} -> {cat2}**.
        Your final task is to choose the most specific Tier 3 category.

        **Support Ticket:**
        {ticket}

        **Available Tier 3 Categories:**
        - {cat3_options}

        Respond with ONLY the name of the single most appropriate category from the list. If none are a good fit, respond with "Other/Unknown".
        """
    )

    handler = CallbackHandler()
    llm = sequential_agent_llm
    chain = prompt | llm | StrOutputParser()

    result = await chain.ainvoke(
        {
            "ticket": ticket,
            "domain": domain,
            "cat1": cat1,
            "cat2": cat2,
            "cat3_options": "\n- ".join(cat3_options),
        },
        config={"callbacks": [handler]},
    )

    return result.strip()
