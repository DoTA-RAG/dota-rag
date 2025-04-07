import argparse
import os
import re
import json
import time
import pandas as pd
import boto3
from dotenv import load_dotenv
from tqdm.auto import tqdm
from botocore.exceptions import ClientError
import prompt_faithfulness
import prompt_relevance

# Load environment variables
load_dotenv()
REGION_NAME = os.getenv("REGION_NAME")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

# Create a Bedrock Runtime client
bedrock_runtime = boto3.client(
    "bedrock-runtime",
    region_name=REGION_NAME,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
)

# Model ID (e.g., Claude 3 Haiku)
model_id = "anthropic.claude-3-5-sonnet-20240620-v1:0"


def evaluate_example(metric_module, prompt_arg, reference_answer, generated_answer):
    """
    Evaluates a single example using the provided metric module.

    For faithfulness, prompt_arg is the context.
    For relevance, prompt_arg is the user query.
    """
    if metric_module == prompt_faithfulness:
        user_content = metric_module.DEFAULT_USER_TEMPLATE.format(
            context=prompt_arg,
            reference_answer=reference_answer,
            generated_answer=generated_answer,
        )
    elif metric_module == prompt_relevance:
        user_content = metric_module.DEFAULT_USER_TEMPLATE.format(
            query=prompt_arg,
            reference_answer=reference_answer,
            generated_answer=generated_answer,
        )
    else:
        raise ValueError("Unsupported metric module provided.")

    request_body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 16384,
        "temperature": 0.6,
        "system": metric_module.DEFAULT_SYSTEM_TEMPLATE,
        "messages": [{"role": "user", "content": user_content}],
    }

    max_retries = 5
    delay = 5  # initial delay in seconds
    for attempt in range(max_retries):
        try:
            response = bedrock_runtime.invoke_model(
                modelId=model_id, body=json.dumps(request_body)
            )
            break
        except ClientError as e:
            if e.response["Error"]["Code"] == "ThrottlingException":
                print(
                    f"ThrottlingException on attempt {attempt+1}/{max_retries}. Retrying in {delay} seconds..."
                )
                time.sleep(delay)
                delay *= 2  # exponential backoff
            else:
                raise e
    else:
        raise Exception(
            "Max retries exceeded for invoke_model due to ThrottlingException."
        )

    response_body = json.loads(response["body"].read())
    return response_body["content"][0]["text"]


def extract_score(output_text):
    """
    Extracts the numerical score enclosed in \\boxed{...} using regex.
    Returns the score as a string (e.g., "1.000") or None if not found.
    """
    match = re.search(r"\\boxed\{([-\d\.]+)", output_text)
    return match.group(1) if match else None


def main():
    parser = argparse.ArgumentParser(
        description="Process and export evaluation results."
    )
    parser.add_argument(
        "--eval_name",
        type=str,
        default="both",
        help="Metric to evaluate: 'faithfulness', 'relevance', or 'both' (default: both).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation_results.csv",
        help="Output CSV file name for evaluation results.",
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default="data/data_morgana_examples_live-rag_results.csv",
        help="Input CSV file path for evaluation data.",
    )
    args = parser.parse_args()

    # Load the evaluation data
    df = pd.read_csv(args.input_file)
    # df = df[:10]  # Limit to the first 10 rows for testing

    results = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating examples"):
        query = row["Question"]
        context = row["Text"]
        reference_answer = row["Answer"]
        generated_answer = row["response_result"]
        topic = row["Topic"] if "Topic" in row and not pd.isna(row["Topic"]) else "N/A"

        # Initialize scores to None
        faithfulness_score = None
        relevance_score = None

        # Evaluate based on the chosen metric(s)
        if args.eval_name.lower() == "faithfulness":
            output_text = evaluate_example(
                prompt_faithfulness, context, reference_answer, generated_answer
            )
            faithfulness_score = extract_score(output_text)
        elif args.eval_name.lower() == "relevance":
            output_text = evaluate_example(
                prompt_relevance, query, reference_answer, generated_answer
            )
            relevance_score = extract_score(output_text)
        else:
            # Evaluate both metrics
            faithfulness_output = evaluate_example(
                prompt_faithfulness, context, reference_answer, generated_answer
            )
            faithfulness_score = extract_score(faithfulness_output)

            relevance_output = evaluate_example(
                prompt_relevance, query, reference_answer, generated_answer
            )
            relevance_score = extract_score(relevance_output)

        # Compute correctness based on defined thresholds
        # For faithfulness: correct if score equals 1.000
        if faithfulness_score is not None:
            try:
                f_val = float(faithfulness_score)
            except ValueError:
                f_val = 0.0
            f_correct = 1 if f_val == 1.0 else 0
            f_pct = f_correct * 100.0
        else:
            f_correct, f_pct = 0, 0.0

        # For relevance: correct if score equals 2
        if relevance_score is not None:
            try:
                r_val = float(relevance_score)
            except ValueError:
                r_val = 0.0
            r_correct = 1 if r_val == 2.0 else 0
            r_pct = r_correct * 100.0
        else:
            r_correct, r_pct = 0, 0.0

        # Print detailed summary per item
        if args.eval_name.lower() == "faithfulness":
            print(
                f"At {idx} problems | faithfulness_score | {faithfulness_score} : Correct {f_correct}/1 ({f_pct:.2f}%), is_correct: {f_correct==1}"
            )
        elif args.eval_name.lower() == "relevance":
            print(
                f"At {idx} problems |  relevance_score | {relevance_score} : Correct {r_correct}/1 ({r_pct:.2f}%), is_correct: {r_correct==1}"
            )
        else:
            # Print both summaries
            print(
                f"At {idx} problems | faithfulness_score | {faithfulness_score} : Correct {f_correct}/1 ({f_pct:.2f}%), is_correct: {f_correct==1}"
            )
            print(
                f"At {idx} problems |  relevance_score | {relevance_score} : Correct {r_correct}/1 ({r_pct:.2f}%), is_correct: {r_correct==1}"
            )

        results.append(
            {
                "faithfulness": faithfulness_score,
                "relevance": relevance_score,
                "faithfulness_correct": f_correct,
                "relevance_correct": r_correct,
            }
        )

    results_df = pd.DataFrame(results)
    results_df.to_csv(args.output, index=False)
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
