# Define prompt templates (https://github.com/explodinggradients/ragas/blob/main/src/ragas/metrics/_faithfulness.py)
DEFAULT_SYSTEM_TEMPLATE = """Your task is to judge the faithfulness of a series of statements based on a given context. The final output must be enclosed in LaTeX-style boxing using \\boxed{...}.

Scoring Guidelines (Graded on a three-point scale):
1: Full support. All answer parts are grounded.
0: Partial support. Not all answer parts are grounded.
-1: No support. All answer parts are not grounded."""

DEFAULT_USER_TEMPLATE = """## Context:
{context}

## Answer:
{generated_answer}

## Scoring:"""
