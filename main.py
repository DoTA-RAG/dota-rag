from dotenv import load_dotenv
import pandas as pd
import asyncio

load_dotenv()  # Load environment variables globally

from rag.rag_pipeline import run_rag_pipeline


async def main():
    ### Example 1 ###
    question = "How did Nat King Cole's musical career evolve in the 1940s?"
    answer = await run_rag_pipeline(question, mode='routing_v2')
    answer['id'] = 1    
    # print("Final Answer:\n", answer)
    pd.DataFrame([answer]).to_json("sample_answers.jsonl", orient='records', lines=True, force_ascii=False)
    print("saved to sample_answers.jsonl")

if __name__ == "__main__":
    asyncio.run(main())
