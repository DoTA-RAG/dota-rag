# Optimized prompt templates for 1.1 Relevance Metric
DEFAULT_SYSTEM_TEMPLATE = """You are tasked with evaluating the relevance of a generated answer based on the provided context and, if available, a reference (ground truth) answer. This evaluation combines semantic equivalence with the ground truth and the degree to which the answer directly addresses the user query.

Grading Guidelines:
2: Correct and relevant (contains no irrelevant information).
1: Correct but contains some irrelevant information.
0: No answer provided (abstention).
-1: Incorrect answer.

The final score must be enclosed in LaTeX-style boxing, for example: \\boxed{2}.
"""

DEFAULT_USER_TEMPLATE = """## User Query: 
{query}

## Reference Answer:
{reference_answer}

## Generated Answer:
{generated_answer}"""
