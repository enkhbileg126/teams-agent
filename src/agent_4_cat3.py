from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI


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
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
    chain = prompt | llm | StrOutputParser()

    result = await chain.ainvoke(
        {
            "ticket": ticket,
            "domain": domain,
            "cat1": cat1,
            "cat2": cat2,
            "cat3_options": "\n- ".join(cat3_options),
        }
    )

    return result.strip()
