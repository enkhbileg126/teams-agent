import asyncio
import hashlib
import json
import os
import re
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langfuse import Langfuse, get_client, observe

# Pydantic v2
from pydantic import BaseModel, Field, ValidationError

# ======================== CONFIG ========================
load_dotenv()
if "GOOGLE_API_KEY" not in os.environ:
    raise ValueError("GOOGLE_API_KEY not found in .env file.")
langfuse = Langfuse(
    public_key=os.getenv('LANGFUSE_PUBLIC_KEY'),
    secret_key=os.getenv('LANGFUSE_SECRET_KEY'),
    host=os.getenv('LANGFUSE_HOST'),
)
langfuse = get_client()
EMBED_MODEL_NAME = "gemini-embedding-001"
LLM_MODEL_NAME = "gemini-2.0-flash"

# Per-level thresholds (tune with eval data)
TAU_L2 = 0.80  # Cat1
TAU_L3 = 0.80  # Cat2
TAU_L4 = 0.70  # Cat3

# Candidate caps (balance diversity & context size)
TOP_K_RAW = 100
TOP_K_TOTAL = 50
MAX_PER_L3 = 5  # max items per Cat2 parent
MAX_PER_L2 = 5  # max items per Cat1 parent

# Special IDs (logic only)
TERMINAL_ID = "TERMINAL"  # internal sentinel for terminal rows (no deeper level)
UNKNOWN_ID = "Other/Unknown"  # keep ID in English for logic
UNKNOWN_LABEL_MN = "Бусад/Тодорхойгүй"  # display label in Mongolian


# ================== Pydantic models (MN output) ==================
class PathLevel(BaseModel):
    id: str = Field(
        description="Тухайн түвшинд өгөгдсөн кандидатуудаас яг энэ ID-г сонго (эсвэл 'Other/Unknown')."
    )
    confidence: float = Field(description="Итгэл [0.0, 1.0].")


class BestPath(BaseModel):
    L1_domain: PathLevel
    L2_cat1: PathLevel
    L3_cat2: PathLevel
    L4_cat3: PathLevel


class ClassificationOutput(BaseModel):
    best_path: BestPath = Field(description="Тасалбарт хамгийн тохирох ганц зам.")
    rationale: str = Field(
        description="Яагаад ийм сонголт хийснээ Монгол хэл дээр товч тайлбарла."
    )
    abstain: bool = Field(
        description="Нийт итгэл бага бол True (хүний хяналт шаардлагатай)."
    )
    selected_index: int = Field(
        description="Кандидат жагсаалтаас сонгосон мөрийн 1-based индекс."
    )


# ===================== Simple PII redaction ======================
MSISDN = re.compile(r'\b(?:\+?976)?[ -]?[0-9]{8}\b')
EMAIL = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b')


def redact(text: str) -> str:
    text = MSISDN.sub("[MSISDN]", text)
    text = EMAIL.sub("[EMAIL]", text)
    return text


# ========================= Helpers =========================
def make_id(level: str, parts: list[str]) -> str:
    """
    Оpaque, stable ID using a SHA1 hash over the exact Mongolian path parts.
    Prevents collisions even with Cyrillic text.
    """
    key = "||".join([p for p in parts if p])
    h = hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]
    return f"{level}.{h}"


def blurb_from_names(domain: str, c1: str, c2: str, c3: str) -> str:
    parts = [p for p in [domain, c1, c2, c3] if p]
    return f"Дараах ангилалтай холбоотой асуудал: {' > '.join(parts)}."


# ============== SmartClassifier (Domain → Cat1 → Cat2 → Cat3) ==============
@observe
class SmartClassifier:
    """
    - CSV: Mongolian labels in columns Domain, Cat1, [Cat2], [Cat3], [description]
    - Normalizes each row to a terminal leaf at the deepest available level.
    - FAISS over short blurbs; full MN names & exact IDs in metadata.
    - Single LLM pass (MN prompt/output) returns selected_index; we bind to that row.
    """

    REQUIRED_MIN_COLS = ["Domain", "Cat1"]

    @observe
    def __init__(self, csv_path: str, index_path: str = "taxonomy.index"):
        print("Initializing Smart Classifier (Domain→Cat1→Cat2→Cat3)...")
        self.taxonomy_df = self._load_and_normalize_taxonomy(csv_path)

        # Embeddings
        self.embedding_model = GoogleGenerativeAIEmbeddings(model=EMBED_MODEL_NAME)

        # Vector index
        if os.path.exists(index_path):
            print(f"Loading existing FAISS index from {index_path}...")
            self.faiss_index = FAISS.load_local(
                index_path, self.embedding_model, allow_dangerous_deserialization=True
            )
        else:
            print(f"No index found. Building new FAISS index at {index_path}...")
            self.faiss_index = self._build_and_save_index(index_path)

        # LLM (with fallback)
        try:
            self.llm = ChatGoogleGenerativeAI(model=LLM_MODEL_NAME, temperature=0.0)
        except Exception:
            self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.0)

        # Parser & MN prompt (selected_index required and rationale must echo chosen path)
        self.parser = JsonOutputParser(pydantic_object=ClassificationOutput)
        self.prompt = ChatPromptTemplate.from_template(
            """
            Та бол телеком үйлчилгээний гомдол ангилагч систем. Доорх **боломжит ангиллын замуудаас** (Domain → Cat1 → Cat2 → Cat3) тухайн гомдолд хамгийн тохирох **ганцхан** замыг сонгоно уу.

            **Дагаж мөрдөх заавар:**
            - Зөвхөн **өгөгдсөн ангиллын ID-г** ашиглана уу. **Шинэ ангилал үүсгэж болохгүй**.
            - Зарим ангилал нь **ТӨГСГӨЛИЙН** (цааш дэд ангилалгүй) байж болно. Энэ тохиолдолд дараагийн түвшин рүү **шилжих шаардлагагүй**.
            - Илүү гүнзгий түвшний ангилал тодорхойгүй бол тухайн түвшний `id`-г `"Other/Unknown"` гэж сонгоод, дараагийн түвшин рүү **бүү үргэлжлүүл**.
            - `selected_index` талбарт сонгосон замын **1-ээс эхэлсэн эгнээний дугаарыг** оруулна уу.
            - `rationale` талбарын тайлбарыг **"Сонгосон зам: <ангиллын зам>."** гэж эхлүүлнэ үү.
            - **Гаралт** нь **зөвхөн JSON** форматтай байна. `rationale` талбарын утга **Монгол хэлээр** байна.

            **Гомдлын мэдээлэл (засварласан):**
            {ticket}

            **Боломжит ангиллын замууд (зам, ID, товч тайлбар):**
            {candidate_paths}

            **Дүрэм:**
            1) Эцэг→хүүхэд ангиллын уялдааг чанд мөрдөх. ТӨГСГӨЛИЙН гэж тэмдэглэсэн ангиллыг цааш задлахгүй.
            2) Гомдлын агуулгад хамгийн сайн тохирох ангиллыг сонгох.
            3) Дараагийн түвшний ангилал эргэлзээтэй бол `"Other/Unknown"` гэж сонгоод зогсох.
            4) `selected_index`-г заавал буцаах ба `rationale`-ыг сонгосон замаар эхлүүлэх.
            5) Доорх схемд нийцсэн, **хүчинтэй JSON** буцаах.

            {format_instructions}
            """,  # noqa: RUF001
            partial_variables={
                "format_instructions": self.parser.get_format_instructions()
            },
        )

    # ---------------- Load & Normalize CSV ----------------
    @observe
    def _load_and_normalize_taxonomy(self, csv_path: str) -> pd.DataFrame:
        df = pd.read_csv(csv_path).fillna("")
        for col in self.REQUIRED_MIN_COLS:
            if col not in df.columns:
                raise ValueError(f"CSV missing required column: {col}")

        # Ensure optional columns exist
        for optional in ["Cat2", "Cat3", "description"]:
            if optional not in df.columns:
                df[optional] = ""

        rows = []
        for _, r in df.iterrows():
            d = str(r["Domain"]).strip()
            c1 = str(r["Cat1"]).strip()
            c2 = str(r["Cat2"]).strip()
            c3 = str(r["Cat3"]).strip()
            if not d or not c1:
                continue

            # deepest available level (terminal leaf)
            if c3:
                level = 4
            elif c2:
                level = 3
            else:
                level = 2

            # Opaque, stable IDs (hash)
            L1_id = make_id("dom", [d])
            L2_id = make_id("cat1", [d, c1])
            L3_id = make_id("cat2", [d, c1, c2]) if level >= 3 else TERMINAL_ID
            L4_id = make_id("cat3", [d, c1, c2, c3]) if level == 4 else TERMINAL_ID
            leaf_id = {2: L2_id, 3: L3_id, 4: L4_id}[level]

            desc = str(r.get("description", "")).strip() or blurb_from_names(
                d, c1, c2 if level >= 3 else "", c3 if level == 4 else ""
            )
            path_str = " > ".join(
                [
                    p
                    for p in [d, c1, c2 if level >= 3 else "", c3 if level == 4 else ""]
                    if p
                ]
            )

            rows.append(
                {
                    "id": leaf_id,
                    "L1_id": L1_id,
                    "L1_domain": d,
                    "L2_id": L2_id,
                    "L2_cat1": c1,
                    "L3_id": L3_id,
                    "L3_cat2": c2 if level >= 3 else "",
                    "L4_id": L4_id,
                    "L4_cat3": c3 if level == 4 else "",
                    "level": level,
                    "path_str": path_str,
                    "description": desc,
                }
            )

        norm = pd.DataFrame(rows)
        if norm.empty:
            raise ValueError(
                "No valid rows after normalization. Check your CSV values."
            )
        return norm

    # ---------------- Build FAISS index ----------------
    @observe
    def _build_and_save_index(self, index_path: str) -> FAISS:
        rows = self.taxonomy_df.to_dict(orient="records")
        texts = [r["description"] for r in rows]
        metadatas = []
        for r in rows:
            metadatas.append(
                {
                    "id": r["id"],
                    "level": r["level"],
                    "L1_id": r["L1_id"],
                    "L1_name": r["L1_domain"],
                    "L2_id": r["L2_id"],
                    "L2_name": r["L2_cat1"],
                    "L3_id": r["L3_id"],
                    "L3_name": r["L3_cat2"],
                    "L4_id": r["L4_id"],
                    "L4_name": r["L4_cat3"],
                    "path_str": r["path_str"],
                }
            )
        idx = FAISS.from_texts(
            texts=texts, embedding=self.embedding_model, metadatas=metadatas
        )
        idx.save_local(index_path)
        print(f"Index built and saved to {index_path}.")
        return idx

    # ---------------- Candidate generation ----------------
    @observe
    def _diversify_and_cap(self, docs) -> list[dict[str, Any]]:
        seen = set()
        per_l3: dict[str, int] = {}
        per_l2: dict[str, int] = {}
        out: list[dict[str, Any]] = []

        for d in docs:
            m = d.metadata
            leaf_id = m["id"]
            if leaf_id in seen:
                continue
            seen.add(leaf_id)

            L3 = m["L3_id"]
            L2 = m["L2_id"]
            if per_l3.get(L3, 0) >= MAX_PER_L3:
                continue
            if per_l2.get(L2, 0) >= MAX_PER_L2:
                continue

            out.append(
                {
                    "id": leaf_id,
                    "path": m["path_str"],
                    "L1_id": m["L1_id"],
                    "L1_name": m["L1_name"],
                    "L2_id": m["L2_id"],
                    "L2_name": m["L2_name"],
                    "L3_id": m["L3_id"],
                    "L3_name": m["L3_name"],
                    "L4_id": m["L4_id"],
                    "L4_name": m["L4_name"],
                    "blurb": d.page_content,
                    "level": m["level"],
                }
            )

            per_l3[L3] = per_l3.get(L3, 0) + 1
            per_l2[L2] = per_l2.get(L2, 0) + 1
            if len(out) >= TOP_K_TOTAL:
                break
        return out

    @observe
    def _format_candidates_for_prompt(self, cands: list[dict[str, Any]]) -> str:
        lines = []
        for i, c in enumerate(cands[:TOP_K_TOTAL], 1):
            blurb = c["blurb"]
            if len(blurb) > 220:
                blurb = blurb[:217] + "..."
            lines.append(
                f'{i}) {c["path"]}  '
                f'[IDs: {c["L1_id"]} > {c["L2_id"]} > {c["L3_id"]} > {c["L4_id"]}]  '
                f'Тайлбар: {blurb}'
            )
        return "\n".join(lines)

    @observe
    def _generate_candidates(self, ticket: str) -> list[dict[str, Any]]:
        ticket_red = redact(ticket)
        docs = self.faiss_index.similarity_search(ticket_red, k=TOP_K_RAW)
        cands = self._diversify_and_cap(docs)
        if not cands:
            docs = self.faiss_index.similarity_search(ticket_red, k=60)
            cands = self._diversify_and_cap(docs)
        return cands

    # ---------------- LLM rerank ----------------
    @observe
    async def _rank_candidates(
        self, ticket: str, candidate_paths: list[dict[str, Any]]
    ) -> dict[str, Any]:
        chain = self.prompt | self.llm | self.parser
        formatted_candidates = self._format_candidates_for_prompt(candidate_paths)
        ticket_red = redact(ticket)
        return await chain.ainvoke(
            {"ticket": ticket_red, "candidate_paths": formatted_candidates}
        )

    # ---------------- Early stop & abstain (logic) ----------------
    @staticmethod
    def _early_stop_and_abstain(output: dict[str, Any]) -> dict[str, Any]:
        try:
            bp = output["best_path"]
            c2 = float(bp["L2_cat1"]["confidence"])
            c3 = float(bp["L3_cat2"]["confidence"])
            c4 = float(bp["L4_cat3"]["confidence"])

            low2 = c2 < TAU_L2
            low3 = c3 < TAU_L3
            low4 = c4 < TAU_L4

            def set_unknown(key: str):
                bp[key]["id"] = UNKNOWN_ID
                bp[key]["confidence"] = 0.0

            if low2:
                set_unknown("L2_cat1")
                set_unknown("L3_cat2")
                set_unknown("L4_cat3")
            elif low3:
                set_unknown("L3_cat2")
                set_unknown("L4_cat3")
            elif low4:
                set_unknown("L4_cat3")

            output["abstain"] = bool((low2 and low3) or (low3 and low4) or (c2 < 0.5))
            return output
        except Exception:
            return output

    # ---------------- Bind to chosen candidate & build path_mn ----------------
    @observe
    def _reconcile_with_selected(
        self, result: dict[str, Any], cands: list[dict[str, Any]]
    ) -> dict[str, Any]:
        sel = int(result.get("selected_index", 0))
        if not (1 <= sel <= len(cands)):
            # If model didn't return a valid index, fall back to name enrichment from candidates
            return self._enrich_with_names(result, cands)

        chosen = cands[sel - 1]
        bp = result.get("best_path", {})

        # Overwrite IDs with authoritative ones from the chosen candidate
        bp["L1_domain"]["id"] = chosen["L1_id"]
        bp["L2_cat1"]["id"] = chosen["L2_id"]
        if chosen["L3_id"] != TERMINAL_ID:
            bp["L3_cat2"]["id"] = chosen["L3_id"]
        if chosen["L4_id"] != TERMINAL_ID:
            bp["L4_cat3"]["id"] = chosen["L4_id"]

        # Set Mongolian names directly from the candidate line
        bp["L1_domain"]["name"] = chosen["L1_name"] or ""
        bp["L2_cat1"]["name"] = chosen["L2_name"] or ""
        bp["L3_cat2"]["name"] = (
            "" if chosen["L3_id"] == TERMINAL_ID else (chosen["L3_name"] or "")
        )
        bp["L4_cat3"]["name"] = (
            "" if chosen["L4_id"] == TERMINAL_ID else (chosen["L4_name"] or "")
        )

        # Build pretty path (hide TERMINAL and Unknown)
        parts = [bp["L1_domain"]["name"], bp["L2_cat1"]["name"]]
        if chosen["L3_id"] != TERMINAL_ID and bp["L3_cat2"]["name"]:
            parts.append(bp["L3_cat2"]["name"])
        if chosen["L4_id"] != TERMINAL_ID and bp["L4_cat3"]["name"]:
            parts.append(bp["L4_cat3"]["name"])
        parts = [p for p in parts if p and p != UNKNOWN_LABEL_MN]
        result["path_mn"] = " > ".join(parts)

        # Ensure rationale starts with chosen path (belt & suspenders)
        prefix = f"Сонгосон зам: {result['path_mn']}."
        rat = (result.get("rationale") or "").strip()
        if not rat.startswith(prefix):
            result["rationale"] = prefix + (" " + rat if rat else "")
        return result

    @observe
    def _enrich_with_names(
        self, result: dict[str, Any], cands: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """
        Fallback enrichment if selected_index was invalid. Maps IDs->names from the whole candidate set,
        then builds path_mn. Kept for robustness.
        """
        id2name: dict[str, str] = {}
        for c in cands:
            id2name[c["L1_id"]] = c["L1_name"] or ""
            id2name[c["L2_id"]] = c["L2_name"] or ""
            if c["L3_id"] not in ("", TERMINAL_ID):
                id2name[c["L3_id"]] = c["L3_name"] or ""
            if c["L4_id"] not in ("", TERMINAL_ID):
                id2name[c["L4_id"]] = c["L4_name"] or ""

        bp = result.get("best_path", {})
        for key in ("L1_domain", "L2_cat1", "L3_cat2", "L4_cat3"):
            node = bp.get(key, {})
            pid = node.get("id")
            if pid and pid in id2name:
                node["name"] = id2name[pid]
            if node.get("id") == UNKNOWN_ID:
                node["name"] = UNKNOWN_LABEL_MN

        # Build user-friendly MN path (hide TERMINAL/Unknown)
        def showable(k: str) -> str:
            node = bp.get(k, {})
            if node.get("id") in (TERMINAL_ID, None, ""):
                return ""
            name = node.get("name", "")
            return "" if name == UNKNOWN_LABEL_MN else name

        parts = [
            showable("L1_domain"),
            showable("L2_cat1"),
            showable("L3_cat2"),
            showable("L4_cat3"),
        ]
        result["path_mn"] = " > ".join([p for p in parts if p])
        return result

    # ---------------- Public API ----------------
    @observe
    async def classify(self, ticket: str) -> dict[str, Any]:
        cands = self._generate_candidates(ticket)
        try:
            initial_result = await self._rank_candidates(ticket, cands)
        except ValidationError as e:
            return {
                "best_path": {
                    "L1_domain": {"id": UNKNOWN_ID, "confidence": 0.0},
                    "L2_cat1": {"id": UNKNOWN_ID, "confidence": 0.0},
                    "L3_cat2": {"id": UNKNOWN_ID, "confidence": 0.0},
                    "L4_cat3": {"id": UNKNOWN_ID, "confidence": 0.0},
                },
                "rationale": f"JSON задлалын алдаа: {e}",
                "abstain": True,
                "path_mn": "",
                "selected_index": 0,
            }

        final = self._early_stop_and_abstain(initial_result)
        final = self._reconcile_with_selected(
            final, cands
        )  # authoritative bind to chosen candidate
        langfuse.flush()
        return final


# ============================ MAIN (example) ============================
async def main():
    ticket_to_process = """Get started
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
    """  # noqa: RUF001

    # IMPORTANT: If you changed ID generation, delete old taxonomy.index before first run
    # CSV must include: Domain, Cat1; optional: Cat2, Cat3, description
    classifier = SmartClassifier(csv_path="data/cleaned.csv")
    result = await classifier.classify(ticket_to_process)
    print("\n--- 🏁 ЭЦСИЙН ГАРАЛТ 🏁 ---")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
