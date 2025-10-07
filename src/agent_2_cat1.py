from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI


async def run_agent_2_cat1(ticket: str, domain: str, cat1_options: list[str]) -> str:
    prompt = ChatPromptTemplate.from_template(
        """
        A support ticket has been identified for the **{domain}** domain.
        Your task is to choose the best Tier 1 category for it.

        **Support Ticket:**
        {ticket}

        **Available Tier 1 Categories for {domain}:**
        - {cat1_options}

        Respond with ONLY the name of the single most appropriate category from the list.
        """
    )
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
    chain = prompt | llm | StrOutputParser()

    result = await chain.ainvoke(
        {"ticket": ticket, "domain": domain, "cat1_options": "\n- ".join(cat1_options)}
    )
    return result.strip()
