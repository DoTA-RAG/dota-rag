from dotenv import load_dotenv

load_dotenv()  # Load environment variables globally

from rag.rag_pipeline import run_rag_pipeline
import json

def main():
    ### Example 1 ###
    question = "How did Nat King Cole's musical career evolve in the 1940s?"
    answer = run_rag_pipeline(question)
    print("Final Answer:\n", answer)
    with open("r.json", 'w') as f:
        json.dump(answer, f, indent=2, ensure_ascii=False)
    
    with open("reasoning_trace.txt", 'w') as f:
        f.write(answer['reasoning_trace'])

if __name__ == "__main__":
    main()
