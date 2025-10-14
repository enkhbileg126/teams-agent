from langchain_core.prompts import ChatPromptTemplate

PROMPT_DOMAIN = ChatPromptTemplate.from_template(
    """
You are an expert ticket classifier. Your task is to categorize the following support ticket into ONE of the predefined domains.

**Support Ticket Text:**
"{ticket}"

**Available Domains:**
- {domains}

Analyze the ticket and respond with ONLY the name of the single most appropriate domain from the list.
"""
)


PROMPT_CAT1 = ChatPromptTemplate.from_template(
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
PROMPT_CAT2 = ChatPromptTemplate.from_template(
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

PROMPT_CAT3 = ChatPromptTemplate.from_template(
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
