import os
from typing import Any, Dict, List, Optional, TypedDict

import pandas as pd
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langfuse import get_client, observe
from langfuse.langchain import CallbackHandler
from langgraph.graph import END, StateGraph

from src.prompts.prompts import PROMPT_CAT1, PROMPT_CAT2, PROMPT_CAT3, PROMPT_DOMAIN
from src.settings import sequential_agent_llm

load_dotenv()
langfuse = get_client()

# ---------- State ----------


class PipelineState(TypedDict, total=False):
    ticket: str
    df_path: str
    df: pd.DataFrame

    # options at each tier
    domain_options: List[str]
    cat1_options: List[str]
    cat2_options: List[str]
    cat3_options: List[str]

    # chosen
    domain: Optional[str]
    category_1: Optional[str]
    category_2: Optional[str]
    category_3: Optional[str]

    # retriever cache for the current Domain->Cat1->Cat2 branch
    _cat3_vs: Any  # FAISS or None
    _cat3_id2label: Dict[str, str]

    # final payload
    final_result: Dict[str, Any]


# ---------- Shared LLM & helpers ----------


def llm_with_handler():
    handler = CallbackHandler()
    llm = sequential_agent_llm
    return llm, handler


def run_chain(prompt: ChatPromptTemplate, inputs: dict):
    llm, handler = llm_with_handler()
    chain = prompt | llm | StrOutputParser()
    return chain.invoke(inputs, config={"callbacks": [handler]}).strip()


def _build_cat3_retriever(state: PipelineState) -> PipelineState:
    """Builds (and caches) a FAISS vector store for Cat3 options scoped to the chosen Domain->Cat1->Cat2.
    Each Cat3 option becomes a Document with rich text created from available CSV columns.
    """
    df = state["df"]
    domain = state["domain"]
    cat1 = state["category_1"]
    cat2 = state["category_2"]

    # Filter rows to the current path
    sub = df[
        (df["Domain"] == domain) & (df["Cat1"] == cat1) & (df["Cat2"] == cat2)
    ].copy()
    # Handle missing descriptive column gracefully
    if "Cat3_Desc" not in sub.columns:
        sub["Cat3_Desc"] = ""

    # Create documents
    docs: List[Document] = []
    id2label: Dict[str, str] = {}

    for i, row in sub.iterrows():
        label = (row.get("Cat3") or "").strip()
        if not label:
            continue
        desc = (row.get("Cat3_Desc") or "").strip()

        # Rich text improves retrieval signals
        text = (
            f"Tier3: {label}\n"
            f"Domain: {domain} | Tier1: {cat1} | Tier2: {cat2}\n"
            f"Description: {desc if desc else 'N/A'}\n"
        )
        doc_id = f"{i}:{label}"
        docs.append(
            Document(page_content=text, metadata={"id": doc_id, "label": label})
        )
        id2label[doc_id] = label

    if not docs:
        # Nothing to index
        return {**state, "_cat3_vs": None, "_cat3_id2label": {}}

    # Build vector store (Google text-embedding-004)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vs = FAISS.from_documents(docs, embeddings)

    return {**state, "_cat3_vs": vs, "_cat3_id2label": id2label}


def _retrieve_cat3_context(ticket: str, state: PipelineState, k: int = 6) -> str:
    """Queries the cached FAISS index with the ticket and returns a formatted shortlist block.
    If the index is empty or missing, returns a fallback message.
    """
    vs = state.get("_cat3_vs")
    if not vs:
        return "(No semantic matches available.)"

    # Similarity search
    results = vs.similarity_search(ticket, k=k)
    if not results:
        return "(No semantic matches found.)"

    lines = []
    for rank, doc in enumerate(results, start=1):
        label = doc.metadata.get("label", "?")
        preview = doc.page_content.replace("\n", " ")
        # keep it compact
        if len(preview) > 240:
            preview = preview[:237] + "..."
        lines.append(f"{rank}. {label} — {preview}")

    return "\n".join(lines)


# ---------- Nodes ----------


@observe(name="load_taxonomy")
def node_load_taxonomy(state: PipelineState) -> PipelineState:
    df_path = state.get("df_path", "data/cleaned.csv")
    df = pd.read_csv(df_path).fillna("")
    if df.empty:
        raise ValueError("Taxonomy CSV is empty or failed to load.")
    domain_options = df["Domain"].unique().tolist()
    return {
        **state,
        "df": df,
        "domain_options": domain_options,
        "final_result": {"ticket": state["ticket"]},
    }


@observe(name="domain")
def node_domain(state: PipelineState) -> PipelineState:
    domain = run_chain(
        PROMPT_DOMAIN,
        {
            "ticket": state["ticket"],
            "domains": "\n- ".join(state["domain_options"]),
        },
    )
    df = state["df"]
    cat1_options = df[df["Domain"] == domain]["Cat1"].unique().tolist()
    return {**state, "domain": domain, "cat1_options": cat1_options}


@observe(name="cat1")
def node_cat1(state: PipelineState) -> PipelineState:
    cat1 = run_chain(
        PROMPT_CAT1,
        {
            "ticket": state["ticket"],
            "domain": state["domain"],
            "cat1_options": "\n- ".join(state["cat1_options"]),
        },
    )
    df = state["df"]
    cat2_df = df[(df["Domain"] == state["domain"]) & (df["Cat1"] == cat1)]
    cat2_options = cat2_df["Cat2"].unique().tolist()
    return {**state, "category_1": cat1, "cat2_options": cat2_options}


@observe(name="cat2")
def node_cat2(state: PipelineState) -> PipelineState:
    cat2 = run_chain(
        PROMPT_CAT2,
        {
            "ticket": state["ticket"],
            "domain": state["domain"],
            "cat1": state["category_1"],
            "cat2_options": "\n- ".join(state["cat2_options"]),
        },
    )
    df = state["df"]
    cat3_df = df[
        (df["Domain"] == state["domain"])
        & (df["Cat1"] == state["category_1"])
        & (df["Cat2"] == cat2)
    ]
    cat3_options = cat3_df["Cat3"].unique().tolist()

    # Build a retriever index for this branch (cached in state)
    tmp_state = {**state, "category_2": cat2, "cat3_options": cat3_options}
    tmp_state = _build_cat3_retriever(tmp_state)
    return tmp_state


@observe(name="cat3")
def node_cat3(state: PipelineState) -> PipelineState:
    retrieval_context = _retrieve_cat3_context(state["ticket"], state, k=6)

    cat3 = run_chain(
        PROMPT_CAT3,
        {
            "ticket": state["ticket"],
            "domain": state["domain"],
            "cat1": state["category_1"],
            "cat2": state["category_2"],
            "cat3_options": "\n- ".join(state["cat3_options"]),
            "retrieval_context": retrieval_context,
        },
    )
    return {**state, "category_3": cat3}


@observe(name="finalize")
def node_finalize(state: PipelineState) -> PipelineState:
    final = dict(state.get("final_result", {}))
    if "domain" in state:
        final["domain"] = state["domain"]
    if "category_1" in state:
        final["category_1"] = state["category_1"]
    if state.get("category_2"):
        final["category_2"] = state["category_2"]
    if state.get("category_3"):
        final["category_3"] = state["category_3"]
    return {**state, "final_result": final}


# ---------- Routing ----------


def should_go_cat1(state: PipelineState) -> str:
    if not state.get("cat1_options"):
        return "finalize"
    return "cat1"


def should_go_cat2(state: PipelineState) -> str:
    opts = state.get("cat2_options") or []
    if (not opts) or (opts == [""]):
        return "finalize"
    return "cat2"


def should_go_cat3(state: PipelineState) -> str:
    opts = state.get("cat3_options") or []
    if (not opts) or (opts == [""]):
        return "finalize"
    return "cat3"


# ---------- Build graph ----------


def build_graph():
    g = StateGraph(PipelineState)

    g.add_node("load_taxonomy", node_load_taxonomy)
    g.add_node("domain", node_domain)
    g.add_node("cat1", node_cat1)
    g.add_node("cat2", node_cat2)
    g.add_node("cat3", node_cat3)
    g.add_node("finalize", node_finalize)

    g.set_entry_point("load_taxonomy")
    g.add_edge("load_taxonomy", "domain")

    g.add_conditional_edges(
        "domain",
        should_go_cat1,
        {
            "cat1": "cat1",
            "finalize": "finalize",
        },
    )

    g.add_conditional_edges(
        "cat1",
        should_go_cat2,
        {
            "cat2": "cat2",
            "finalize": "finalize",
        },
    )

    g.add_conditional_edges(
        "cat2",
        should_go_cat3,
        {
            "cat3": "cat3",
            "finalize": "finalize",
        },
    )

    g.add_edge("cat3", "finalize")
    g.add_edge("finalize", END)

    return g.compile()


# ---------- Public API ----------


@observe(name="SequentialGraph")
def run_pipeline(ticket: str, csv_path: str = "data/cleaned.csv") -> Dict[str, Any]:
    app = build_graph()
    out: PipelineState = app.invoke({"ticket": ticket, "df_path": csv_path})
    langfuse.flush()
    return out["final_result"]


# ---------- Example ----------
if __name__ == "__main__":
    EXAMPLE_TICKET = """Get started
Toki
•
14:14
Сайн уу? 👋 Танд тусламж хэрэгтэй бол би туслахад бэлэн байна. Та асуух асуултаа бичээрэй 😊

У. Цогтбаатар
•
14:15
Лизингий гэрээ авах хэрэгтэй байна
urtnasan.a
•
14:15
Сайн байна уу? Би Солонгоо байна 🙋‍♀️ Та гар утаcны гэрээгээ файлаар авах бол дараах мэдээллийг илгээгээрэй.
1. Өөрийн РД
2. И-мэйл хаяг
3. Бүртгэлтэй дугаар
4. Иргэний үнэмлэхээ барьж авхуулсан селфи зураг

У. Цогтбаатар
•
14:17
Ел92032018
tsogtootsoogii58@gmail.com
80773696
urtnasan.a
•
14:18
Иргэний үнэмлэхээ барьж авхуулсан селфи зургаа илгээгээрэй.

attachment
urtnasan.a
•
14:19
Та 10-20 минутын дараагаар и-мэйл хаягаа шалгаарай. Таны и-мэйл хаягт тодорхойлолтыг илгээнэ 😊
"""
    result = run_pipeline(EXAMPLE_TICKET, "data/cleaned.csv")
    import json

    print("\n--- 🏁 FINAL GRAPH OUTPUT (with retriever) 🏁 ---")
    print(json.dumps(result, indent=2, ensure_ascii=False))
