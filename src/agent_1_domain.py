from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI


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
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
    chain = prompt | llm | StrOutputParser()

    result = await chain.ainvoke({"ticket": ticket, "domains": "\n- ".join(domains)})
    return result.strip()
