import asyncio
import json

import pandas as pd
from dotenv import load_dotenv
from langfuse import Langfuse, get_client, observe

# Import the agent runner functions
from src.agent_1_domain import run_agent_1_domain
from src.agent_2_cat1 import run_agent_2_cat1
from src.agent_3_cat2 import run_agent_3_cat2
from src.agent_4_cat3 import run_agent_4_cat3

load_dotenv()
langfuse = get_client()


class SequentialAgentPipeline:
    def __init__(self, csv_path: str):
        print(f"Loading and preparing taxonomy from {csv_path}...")
        self.df = pd.read_csv(csv_path).fillna("")
        if self.df.empty:
            raise ValueError("Taxonomy CSV is empty or failed to load.")
        self.final_result = {}

    @observe
    async def run(self, ticket: str):
        self.final_result = {"ticket": ticket}

        # --- Stage 1: Domain ---
        domain_options = self.df['Domain'].unique().tolist()
        domain = await run_agent_1_domain(ticket, domain_options)
        self.final_result["domain"] = domain

        # --- Stage 2: Category 1 ---
        cat1_options = self.df[self.df['Domain'] == domain]['Cat1'].unique().tolist()
        if not cat1_options:
            print(f"Error: No Cat1 options found for Domain '{domain}'. Aborting.")
            return self.final_result

        cat1 = await run_agent_2_cat1(ticket, domain, cat1_options)
        self.final_result["category_1"] = cat1

        # --- Stage 3: Category 2 ---
        cat2_df = self.df[(self.df['Domain'] == domain) & (self.df['Cat1'] == cat1)]
        cat2_options = cat2_df['Cat2'].unique().tolist()
        if not cat2_options or cat2_options == ['']:
            print(f"Path terminates at Cat1 for '{domain} -> {cat1}'.")
            return self.final_result

        cat2 = await run_agent_3_cat2(ticket, domain, cat1, cat2_options)
        self.final_result["category_2"] = cat2

        # --- Stage 4: Category 3 ---
        cat3_df = cat2_df[cat2_df['Cat2'] == cat2]
        cat3_options = cat3_df['Cat3'].unique().tolist()
        if not cat3_options or cat3_options == ['']:
            print(f"Path terminates at Cat2 for '{domain} -> {cat1} -> {cat2}'.")
            return self.final_result

        cat3 = await run_agent_4_cat3(ticket, domain, cat1, cat2, cat3_options)
        self.final_result["category_3"] = cat3
        print(f"{domain}>{cat1}>{cat2}>{cat3}")
        langfuse.flush()
        return self.final_result


# --- Main Execution Block ---
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
Та 10-20 минутын дараагаар и-мэйл хаягаа шалгаарай. Таны и-мэйл хаягт тодорхойлолтыг илгээнэ 😊"""

    pipeline = SequentialAgentPipeline("data/cleaned.csv")
    final_categorization = await pipeline.run(ticket_to_process)

    print("\n--- 🏁 FINAL PIPELINE OUTPUT 🏁 ---")
    print(json.dumps(final_categorization, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    # Ensure you have the necessary libraries:
    # pip install python-dotenv langchain-google-genai pandas
    asyncio.run(main())
