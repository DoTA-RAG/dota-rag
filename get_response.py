from dotenv import load_dotenv

# Load environment variables globally
load_dotenv()

from rag.rag_pipeline import run_rag_pipeline
import pandas as pd
from tqdm import tqdm


def main():
    # Read the CSV file
    df = pd.read_csv("data/data_morgana_examples_live-rag.csv")

    # Ensure 'response_result' column exists
    df["response_result"] = None

    # Iterate over each row with a progress bar
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing questions"):
        question = row.get("Question")
        # Skip if question is missing or not a string
        if pd.isna(question) or not isinstance(question, str) or question.strip() == "":
            print(f"Skipping row {idx} with invalid question: {question}")
            continue

        answer = run_rag_pipeline(question)
        df.at[idx, "response_result"] = answer

    # Save the updated DataFrame to a new CSV file
    output_file = "data/data_morgana_examples_live-rag_results.csv"
    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()
