from dotenv import load_dotenv

load_dotenv()  # Load environment variables globally

from rag.rag_pipeline import run_rag_pipeline


def main():
    ### Example 1 ###
    question = "How did Nat King Cole's musical career evolve in the 1940s?"
    answer = run_rag_pipeline(question)
    print("Final Answer:\n", answer)


if __name__ == "__main__":
    main()
