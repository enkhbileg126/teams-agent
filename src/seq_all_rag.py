# graph_agent_taxonomy_rag.py
# Single-file LangGraph agent implementing taxonomy-grounded RAG at ALL tiers (Domain, Cat1, Cat2, Cat3)
# No model training required. Uses hybrid retrieval (BM25 + dense) and constrained choices.
#
# Requirements:
#   pip install langgraph>=0.2 langchain>=0.2 langchain-google-genai langchain-community pandas python-dotenv langfuse faiss-cpu rank_bm25
#
# CSV schema suggestions (NaN allowed; will be filled to ""):
#   Domain, Domain_Desc, Domain_Examples
#   Cat1,   Cat1_Desc,   Cat1_Examples
#   Cat2,   Cat2_Desc,   Cat2_Examples
#   Cat3,   Cat3_Desc,   Cat3_Examples
#
# Notes:
# - Descriptions/Examples are OPTIONAL but greatly improve retrieval.
# - The graph builds per-tier indices and retrieves K candidates and their mini-context
#   into the LLM prompt; generation is HARD-CONSTRAINED to the allowed labels at that tier.
# - Includes Reciprocal Rank Fusion (RRF) to merge BM25 + dense candidates.
# - Gracefully degrades if a tier has no options (early finalize) or no embeddings.
#
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, TypedDict

import pandas as pd
from dotenv import load_dotenv
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langfuse import get_client, observe
from langfuse.langchain import CallbackHandler
from langgraph.graph import END, StateGraph

load_dotenv()
langfuse = get_client()

# ----------------------------- Config ---------------------------------
K_DOMAIN = 20
K_CAT1 = 15
K_CAT2 = 12
K_CAT3 = 10
RRF_K = 60  # RRF constant

EMBED_MODEL = "text-embedding-004"  # Google Generative AI embeddings
LLM_MODEL = "gemini-2.0-flash"


# ----------------------------- State ----------------------------------
class PipelineState(TypedDict, total=False):
    ticket: str
    df_path: str
    df: pd.DataFrame

    # Options at each tier (full universe for that branch)
    domain_options: List[str]
    cat1_options: List[str]
    cat2_options: List[str]
    cat3_options: List[str]

    # Chosen outputs
    domain: Optional[str]
    category_1: Optional[str]
    category_2: Optional[str]
    category_3: Optional[str]

    # Retriever caches (per tier or branch)
    _vs_domain: Any
    _vs_cat1: Dict[str, Any]  # keyed by Domain
    _vs_cat2: Dict[Tuple[str, str], Any]  # keyed by (Domain, Cat1)
    _vs_cat3: Dict[Tuple[str, str, str], Any]  # keyed by (Domain, Cat1, Cat2)

    _bm25_domain: Any
    _bm25_cat1: Dict[str, Any]
    _bm25_cat2: Dict[Tuple[str, str], Any]
    _bm25_cat3: Dict[Tuple[str, str, str], Any]

    # For mapping doc ids -> labels
    _id2label_domain: Dict[str, str]
    _id2label_cat1: Dict[str, Dict[str, str]]
    _id2label_cat2: Dict[Tuple[str, str], Dict[str, str]]
    _id2label_cat3: Dict[Tuple[str, str, str], Dict[str, str]]

    # final payload
    final_result: Dict[str, Any]


# --------------------------- Helpers ----------------------------------


def llm_with_handler():
    handler = CallbackHandler()
    llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0)
    return llm, handler


def run_chain(prompt: ChatPromptTemplate, inputs: dict):
    llm, handler = llm_with_handler()
    chain = prompt | llm | StrOutputParser()
    return chain.invoke(inputs, config={"callbacks": [handler]}).strip()


def rrf_merge(bm25_ids: List[str], dense_ids: List[str], k: int) -> List[str]:
    # Reciprocal Rank Fusion over two ranked lists of string ids
    scores: Dict[str, float] = {}

    def add_list(lst):
        for rank, _id in enumerate(lst, start=1):
            scores[_id] = scores.get(_id, 0.0) + 1.0 / (RRF_K + rank)

    add_list(bm25_ids)
    add_list(dense_ids)
    return [i for i, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)][:k]


def build_docs(
    text_rows: List[Dict[str, str]], id_fmt: str
) -> Tuple[List[Document], Dict[str, str]]:
    docs: List[Document] = []
    id2label: Dict[str, str] = {}
    for idx, row in enumerate(text_rows):
        label = (row.get("label") or "").strip()
        if not label:
            continue
        desc = (row.get("desc") or "").strip()
        ex = (row.get("examples") or "").strip()
        txt = f"Label: {label}\nDescription: {desc if desc else 'N/A'}\nExamples: {ex if ex else 'N/A'}"
        _id = id_fmt.format(idx=idx, label=label)
        docs.append(Document(page_content=txt, metadata={"id": _id, "label": label}))
        id2label[_id] = label
    return docs, id2label


def make_bm25_corpus(docs: List[Document]):
    """Build a BM25 retriever that *preserves metadata* by indexing Documents directly.
    Returns (retriever), no parallel ids needed because returned docs keep metadata.
    """
    if not docs:
        return None
    return BM25Retriever.from_documents(docs)


def dense_search(vs: Any, query: str, k: int) -> List[str]:
    if not vs:
        return []
    res = vs.similarity_search_with_score(query, k=k)
    return [d.metadata.get("id") for d, _ in res]


def bm25_search(retr: Any, query: str, k: int) -> List[str]:
    if not retr:
        return []
    # New API: use invoke() instead of get_relevant_documents()
    docs = retr.invoke(query)
    out_ids: List[str] = []
    for d in docs[:k]:
        _id = d.metadata.get("id") if d.metadata else None
        if _id:
            out_ids.append(_id)
    return out_ids


# ------------------------ Prompts (constrained) ------------------------
PROMPT_CONSTRAINED = ChatPromptTemplate.from_template(
    """
You are a careful taxonomy classifier. Read the ticket and select ONE label from the Allowed Choices.

Ticket:
"""
    "{ticket}"
    """

Shortlist (retrieved context; may include summaries/examples):
{retrieval_context}

Allowed Choices:
- {choices}

Rules:
- Output EXACTLY one label string from Allowed Choices (verbatim). If nothing clearly fits, output "Other/Unknown".
- Do not add explanations.
"""
)


# ------------------------ Node: Load taxonomy -------------------------
@observe(name="load_taxonomy")
def node_load_taxonomy(state: PipelineState) -> PipelineState:
    df_path = state.get("df_path", "data/cleaned.csv")
    df = pd.read_csv(df_path).fillna("")

    # --- Normalize key columns to avoid whitespace/case mismatches ---
    for col in [
        "Domain",
        "Cat1",
        "Cat2",
        "Cat3",
        "Domain_Desc",
        "Domain_Examples",
        "Cat1_Desc",
        "Cat1_Examples",
        "Cat2_Desc",
        "Cat2_Examples",
        "Cat3_Desc",
        "Cat3_Examples",
    ]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    if df.empty:
        raise ValueError("Taxonomy CSV is empty or failed to load.")

    domain_options = df["Domain"].dropna().astype(str).str.strip().unique().tolist()

    # Build Domain docs from distinct domains (safe: handle missing row)
    domain_rows = []
    for dom in domain_options:
        sub = df[df["Domain"] == dom]
        if sub.empty:
            desc = examples = ""
        else:
            r = sub.iloc[0]
            desc = r.get("Domain_Desc", "")
            examples = r.get("Domain_Examples", "")
        domain_rows.append({"label": dom, "desc": desc, "examples": examples})

    dom_docs, id2label_dom = build_docs(domain_rows, id_fmt="dom:{idx}:{label}")

    embeddings = GoogleGenerativeAIEmbeddings(model=EMBED_MODEL)
    vs_domain = FAISS.from_documents(dom_docs, embeddings) if dom_docs else None
    bm25_domain = make_bm25_corpus(dom_docs) if dom_docs else None

    return {
        **state,
        "df": df,
        "domain_options": domain_options,
        "_vs_domain": vs_domain,
        "_bm25_domain": bm25_domain,
        "_id2label_domain": id2label_dom,
        "_vs_cat1": {},
        "_vs_cat2": {},
        "_vs_cat3": {},
        "_bm25_cat1": {},
        "_bm25_cat2": {},
        "_bm25_cat3": {},
        "_id2label_cat1": {},
        "_id2label_cat2": {},
        "_id2label_cat3": {},
        "final_result": {"ticket": state["ticket"]},
    }


# ------------------------ Index builders per tier ----------------------


def ensure_cat1_index(state: PipelineState, domain: str) -> PipelineState:
    if domain in state["_vs_cat1"]:
        return state
    df = state["df"]
    sub = df[df["Domain"] == domain]
    cat1_options = sub["Cat1"].dropna().astype(str).str.strip().unique().tolist()

    rows = []
    for c1 in cat1_options:
        s = sub[sub["Cat1"] == c1]
        if s.empty:
            desc = examples = ""
        else:
            r = s.iloc[0]
            desc = r.get("Cat1_Desc", "")
            examples = r.get("Cat1_Examples", "")
        rows.append({"label": c1, "desc": desc, "examples": examples})
    docs, id2label = build_docs(rows, id_fmt=f"c1:{domain}:{{idx}}:{{label}}")

    if docs:
        embeddings = GoogleGenerativeAIEmbeddings(model=EMBED_MODEL)
        vs = FAISS.from_documents(docs, embeddings)
        bm25 = make_bm25_corpus(docs)
    else:
        vs, bm25 = None, None

    state["_vs_cat1"][domain] = vs
    state["_bm25_cat1"][domain] = bm25
    state["_id2label_cat1"][domain] = id2label
    return state
    df = state["df"]
    sub = df[df["Domain"] == domain]
    cat1_options = sub["Cat1"].dropna().astype(str).str.strip().unique().tolist()

    rows = []
    for c1 in cat1_options:
        r = sub[sub["Cat1"] == c1].iloc[0]
        rows.append(
            {
                "label": c1,
                "desc": r.get("Cat1_Desc", ""),
                "examples": r.get("Cat1_Examples", ""),
            }
        )
    docs, id2label = build_docs(rows, id_fmt=f"c1:{domain}:{{idx}}:{{label}}")

    if docs:
        embeddings = GoogleGenerativeAIEmbeddings(model=EMBED_MODEL)
        vs = FAISS.from_documents(docs, embeddings)
        bm25, _ = make_bm25_corpus(docs)
    else:
        vs, bm25 = None, None

    state["_vs_cat1"][domain] = vs
    state["_bm25_cat1"][domain] = bm25
    state["_id2label_cat1"][domain] = id2label
    return state


def ensure_cat2_index(state: PipelineState, domain: str, cat1: str) -> PipelineState:
    key = (domain, cat1)
    if key in state["_vs_cat2"]:
        return state
    df = state["df"]
    sub = df[(df["Domain"] == domain) & (df["Cat1"] == cat1)]
    cat2_options = sub["Cat2"].dropna().astype(str).str.strip().unique().tolist()

    rows = []
    for c2 in cat2_options:
        s = sub[sub["Cat2"] == c2]
        if s.empty:
            desc = examples = ""
        else:
            r = s.iloc[0]
            desc = r.get("Cat2_Desc", "")
            examples = r.get("Cat2_Examples", "")
        rows.append({"label": c2, "desc": desc, "examples": examples})
    docs, id2label = build_docs(rows, id_fmt=f"c2:{domain}:{cat1}:{{idx}}:{{label}}")

    if docs:
        embeddings = GoogleGenerativeAIEmbeddings(model=EMBED_MODEL)
        vs = FAISS.from_documents(docs, embeddings)
        bm25 = make_bm25_corpus(docs)
    else:
        vs, bm25 = None, None

    state["_vs_cat2"][key] = vs
    state["_bm25_cat2"][key] = bm25
    state["_id2label_cat2"][key] = id2label
    return state
    df = state["df"]
    sub = df[(df["Domain"] == domain) & (df["Cat1"] == cat1)]
    cat2_options = sub["Cat2"].dropna().astype(str).str.strip().unique().tolist()

    rows = []
    for c2 in cat2_options:
        r = sub[sub["Cat2"] == c2].iloc[0]
        rows.append(
            {
                "label": c2,
                "desc": r.get("Cat2_Desc", ""),
                "examples": r.get("Cat2_Examples", ""),
            }
        )
    docs, id2label = build_docs(rows, id_fmt=f"c2:{domain}:{cat1}:{{idx}}:{{label}}")

    if docs:
        embeddings = GoogleGenerativeAIEmbeddings(model=EMBED_MODEL)
        vs = FAISS.from_documents(docs, embeddings)
        bm25, _ = make_bm25_corpus(docs)
    else:
        vs, bm25 = None, None

    state["_vs_cat2"][key] = vs
    state["_bm25_cat2"][key] = bm25
    state["_id2label_cat2"][key] = id2label
    return state


def ensure_cat3_index(
    state: PipelineState, domain: str, cat1: str, cat2: str
) -> PipelineState:
    key = (domain, cat1, cat2)
    if key in state["_vs_cat3"]:
        return state
    df = state["df"]
    sub = df[(df["Domain"] == domain) & (df["Cat1"] == cat1) & (df["Cat2"] == cat2)]
    cat3_options = sub["Cat3"].dropna().astype(str).str.strip().unique().tolist()

    rows = []
    for c3 in cat3_options:
        s = sub[sub["Cat3"] == c3]
        if s.empty:
            desc = examples = ""
        else:
            r = s.iloc[0]
            desc = r.get("Cat3_Desc", "")
            examples = r.get("Cat3_Examples", "")
        rows.append({"label": c3, "desc": desc, "examples": examples})
    docs, id2label = build_docs(
        rows, id_fmt=f"c3:{domain}:{cat1}:{cat2}:{{idx}}:{{label}}"
    )

    if docs:
        embeddings = GoogleGenerativeAIEmbeddings(model=EMBED_MODEL)
        vs = FAISS.from_documents(docs, embeddings)
        bm25 = make_bm25_corpus(docs)
    else:
        vs, bm25 = None, None

    state["_vs_cat3"][key] = vs
    state["_bm25_cat3"][key] = bm25
    state["_id2label_cat3"][key] = id2label
    return state
    df = state["df"]
    sub = df[(df["Domain"] == domain) & (df["Cat1"] == cat1) & (df["Cat2"] == cat2)]
    cat3_options = sub["Cat3"].dropna().astype(str).str.strip().unique().tolist()

    rows = []
    for c3 in cat3_options:
        r = sub[sub["Cat3"] == c3].iloc[0]
        rows.append(
            {
                "label": c3,
                "desc": r.get("Cat3_Desc", ""),
                "examples": r.get("Cat3_Examples", ""),
            }
        )
    docs, id2label = build_docs(
        rows, id_fmt=f"c3:{domain}:{cat1}:{cat2}:{{idx}}:{{label}}"
    )

    if docs:
        embeddings = GoogleGenerativeAIEmbeddings(model=EMBED_MODEL)
        vs = FAISS.from_documents(docs, embeddings)
        bm25, _ = make_bm25_corpus(docs)
    else:
        vs, bm25 = None, None

    state["_vs_cat3"][key] = vs
    state["_bm25_cat3"][key] = bm25
    state["_id2label_cat3"][key] = id2label
    return state


# ------------------------ Retrieval per tier ---------------------------


def retrieve_with_hybrid(
    ticket: str, bm25, vs, id2label: Dict[str, str], k_dense: int, k_final: int
) -> Tuple[List[str], str]:
    # Dense ids
    dense_ids = dense_search(vs, ticket, k=k_dense) if vs else []

    # BM25 now returns ids directly from metadata
    bm25_ids = bm25_search(bm25, ticket, k=k_dense) if bm25 else []

    merged = rrf_merge(bm25_ids, dense_ids, k=k_final)

    # Build a readable retrieval context block
    lines = []
    for rank, _id in enumerate(merged, start=1):
        label = id2label.get(_id, "?")
        lines.append(f"{rank}. {label}")
    retrieval_context = (
        "\n".join(lines) if lines else "(No semantic matches available.)"
    )
    labels = [id2label[_id] for _id in merged if _id in id2label]
    return labels, retrieval_context


# ----------------------------- Nodes ----------------------------------
from langgraph.graph import StateGraph


@observe(name="domain")
def node_domain(state: PipelineState) -> PipelineState:
    # Ensure indices
    # (already built in load_taxonomy)
    vs = state.get("_vs_domain")
    bm25 = state.get("_bm25_domain")
    id2 = state.get("_id2label_domain", {})

    # Hybrid retrieve and constrain choices to the global domain options
    shortlist, ctx = retrieve_with_hybrid(
        state["ticket"],
        bm25,
        vs,
        id2,
        k_dense=K_DOMAIN,
        k_final=min(K_DOMAIN, len(state["domain_options"])),
    )
    allowed = state["domain_options"]

    # Constrained prompt
    out = run_chain(
        PROMPT_CONSTRAINED,
        {
            "ticket": state["ticket"],
            "choices": "\n- ".join(allowed),
            "retrieval_context": ctx,
        },
    )
    domain = out if out in allowed else (shortlist[0] if shortlist else out)

    # Prepare Cat1 options
    df = state["df"]
    cat1_options = (
        df[df["Domain"] == domain]["Cat1"]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
        .tolist()
    )

    # Build Cat1 index for this domain
    state = ensure_cat1_index(state, domain)
    return {**state, "domain": domain, "cat1_options": cat1_options}


@observe(name="cat1")
def node_cat1(state: PipelineState) -> PipelineState:
    domain = state["domain"]
    vs = state["_vs_cat1"].get(domain)
    bm25 = state["_bm25_cat1"].get(domain)
    id2 = state["_id2label_cat1"].get(domain, {})

    shortlist, ctx = retrieve_with_hybrid(
        state["ticket"],
        bm25,
        vs,
        id2,
        k_dense=K_CAT1,
        k_final=min(K_CAT1, len(state["cat1_options"])),
    )
    allowed = state["cat1_options"]

    out = run_chain(
        PROMPT_CONSTRAINED,
        {
            "ticket": state["ticket"],
            "choices": "\n- ".join(allowed),
            "retrieval_context": ctx,
        },
    )
    cat1 = out if out in allowed else (shortlist[0] if shortlist else out)

    df = state["df"]
    cat2_options = (
        df[(df["Domain"] == domain) & (df["Cat1"] == cat1)]["Cat2"]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
        .tolist()
    )

    # Build Cat2 index
    state = ensure_cat2_index(state, domain, cat1)
    return {**state, "category_1": cat1, "cat2_options": cat2_options}


@observe(name="cat2")
def node_cat2(state: PipelineState) -> PipelineState:
    domain = state["domain"]
    cat1 = state["category_1"]

    vs = state["_vs_cat2"].get((domain, cat1))
    bm25 = state["_bm25_cat2"].get((domain, cat1))
    id2 = state["_id2label_cat2"].get((domain, cat1), {})

    shortlist, ctx = retrieve_with_hybrid(
        state["ticket"],
        bm25,
        vs,
        id2,
        k_dense=K_CAT2,
        k_final=min(K_CAT2, len(state["cat2_options"])),
    )
    allowed = state["cat2_options"]

    out = run_chain(
        PROMPT_CONSTRAINED,
        {
            "ticket": state["ticket"],
            "choices": "\n- ".join(allowed),
            "retrieval_context": ctx,
        },
    )
    cat2 = out if out in allowed else (shortlist[0] if shortlist else out)

    df = state["df"]
    cat3_options = (
        df[(df["Domain"] == domain) & (df["Cat1"] == cat1) & (df["Cat2"] == cat2)][
            "Cat3"
        ]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
        .tolist()
    )

    # Build Cat3 index
    state = ensure_cat3_index(state, domain, cat1, cat2)
    return {**state, "category_2": cat2, "cat3_options": cat3_options}


@observe(name="cat3")
def node_cat3(state: PipelineState) -> PipelineState:
    domain = state["domain"]
    cat1 = state["category_1"]
    cat2 = state["category_2"]

    vs = state["_vs_cat3"].get((domain, cat1, cat2))
    bm25 = state["_bm25_cat3"].get((domain, cat1, cat2))
    id2 = state["_id2label_cat3"].get((domain, cat1, cat2), {})

    shortlist, ctx = retrieve_with_hybrid(
        state["ticket"],
        bm25,
        vs,
        id2,
        k_dense=K_CAT3,
        k_final=min(K_CAT3, len(state["cat3_options"])),
    )
    allowed = state["cat3_options"]

    out = run_chain(
        PROMPT_CONSTRAINED,
        {
            "ticket": state["ticket"],
            "choices": "\n- ".join(allowed) if allowed else "(none)",
            "retrieval_context": ctx,
        },
    )
    cat3 = out if out in allowed else (shortlist[0] if shortlist else out)
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


# ----------------------------- Routing --------------------------------


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


# --------------------------- Build graph -------------------------------


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
        "domain", should_go_cat1, {"cat1": "cat1", "finalize": "finalize"}
    )
    g.add_conditional_edges(
        "cat1", should_go_cat2, {"cat2": "cat2", "finalize": "finalize"}
    )
    g.add_conditional_edges(
        "cat2", should_go_cat3, {"cat3": "cat3", "finalize": "finalize"}
    )

    g.add_edge("cat3", "finalize")
    g.add_edge("finalize", END)

    return g.compile()


# ----------------------------- Public API ------------------------------
@observe(name="SequentialGraph_RAG")
def run_pipeline(ticket: str, csv_path: str = "data/cleaned.csv") -> Dict[str, Any]:
    app = build_graph()
    out: PipelineState = app.invoke({"ticket": ticket, "df_path": csv_path})
    get_client().flush()
    return out["final_result"]


# ---------------------------- Example run ------------------------------
if __name__ == "__main__":
    EXAMPLE_TICKET = """Сайн уу, лизингийн гэрээний хуулбар хэрэгтэй байна. И-мэйл рүү явуулж өгнө үү."""
    result = run_pipeline(EXAMPLE_TICKET, "data/cleaned.csv")
    import json

    print("\n--- 🏁 FINAL GRAPH OUTPUT (taxonomy-RAG all tiers) 🏁 ---")
    print(json.dumps(result, indent=2, ensure_ascii=False))
