from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

OPENAI_MODEL_NAME = "gpt-5-mini"
GEMINI_MODEL_NAME = "gemini-2.0-flash"
openai_llm = ChatOpenAI(model=OPENAI_MODEL_NAME, reasoning_effort="minimal")
gemini_llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL_NAME, temperature=0)

# set the llm
sequential_agent_llm = openai_llm
rag_agent_llm = openai_llm
