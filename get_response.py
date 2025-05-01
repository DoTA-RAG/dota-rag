# from dotenv import load_dotenv

# # Load environment variables globally
# load_dotenv()

# from rag.rag_pipeline import (
#     run_rag_pipeline,
#     # run_rag_pipeline_batch,
# )
# import pandas as pd
# from tqdm import tqdm


# def main():
#     # Read the CSV file
#     df = pd.read_csv("data/data_morgana_examples_live-rag.csv")

#     # Ensure 'response_result' column exists
#     df["response_result"] = None

#     # Iterate over each row with a progress bar
#     for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing questions"):
#         question = row.get("Question")
#         # Skip if question is missing or not a string
#         if pd.isna(question) or not isinstance(question, str) or question.strip() == "":
#             print(f"Skipping row {idx} with invalid question: {question}")
#             continue

#         answer = run_rag_pipeline(question)
#         # answer = run_rag_pipeline_batch(question)
#         df.at[idx, "response_result"] = answer

#     # Save the updated DataFrame to a new CSV file
#     output_file = "data/data_morgana_examples_live-rag_results.csv"
#     df.to_csv(output_file, index=False)
#     print(f"Results saved to {output_file}")


# if __name__ == "__main__":
#     main()


from dotenv import load_dotenv
import time

# Load environment variables globally
load_dotenv()

from ai71.exceptions import APIError
from rag.rag_pipeline import run_rag_pipeline
import pandas as pd
from tqdm import tqdm
import json

def main():
    # Read the CSV file
    df = pd.read_csv("data/data_morgana_examples_live-rag.csv")
    
    all_answers = []

    # Ensure 'response_result' column exists
    df["response_result"] = None

    # Iterate over each row with a progress bar
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing questions"):
        question = row.get("Question")
        # Skip if question is missing or not a string
        if pd.isna(question) or not isinstance(question, str) or not question.strip():
            print(f"Skipping row {idx} with invalid question: {question!r}")
            continue

        # Retry logic
        answer = ""
        max_retries = 5
        for attempt in range(1, max_retries + 1):
            try:
                answer = run_rag_pipeline(question)
                final_answer = answer['answer']
                break  # success!
            except APIError as e:
                msg = str(e)
                # detect rate limit (or you can catch all APIError)
                if "Rate limit exceeded" in msg or "HTTP 429" in msg:
                    wait = 2**attempt
                    print(
                        f"[{idx}] Rate limit hit (attempt {attempt}/{max_retries}), retrying in {wait}s..."
                    )
                    time.sleep(wait)
                    continue
                else:
                    # other API errors: bail out
                    print(f"[{idx}] APIError: {msg}, skipping question.")
                    break
            except Exception as e:
                # unexpected error: log and skip
                print(f"[{idx}] Unexpected error: {e!r}, skipping question.")
                break
        
        all_answers.append(answer)
        
        with open("data/data_morgana_examples_live-rag_results_full.json", 'w') as f:
            json.dump(all_answers, f, indent=2, ensure_ascii=False)

        df.at[idx, "response_result"] = final_answer

    # Save the updated DataFrame to a new CSV file
    output_file = "data/data_morgana_examples_live-rag_results.csv"
    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")
    
if __name__ == "__main__":
    main()
