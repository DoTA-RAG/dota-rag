# Dota RAG Pipeline

This repository implements a **Retrieval-Augmented Generation (RAG) pipeline** for answering support questions. It leverages **AWS SSM** for secure parameter management, a **transformer-based embedding model**, **Pinecone** for vector search, and **AI71's Falcon-180B-chat model** for generating responses.

## 🚀 Features

- **🔐 Secure Parameter Management:** Uses AWS SSM to securely retrieve parameters and secrets.
- **🤖 Transformer Embeddings:** Uses a pre-trained transformer model to generate embeddings for queries.
- **🔍 Pinecone Integration:** Retrieves relevant context via vector search.
- **📚 RAG Pipeline:** Combines retrieved context with a chat model to generate informative answers.
- **📊 Reranking Mechanism:** Uses `bge-reranker-v2-m3` model to enhance search relevance.
- **✅ Evaluation Script:** Processes a CSV file of questions and saves responses for evaluation.

---

## 🛠️ Requirements

- Python **3.8 or later**
- **pip** (Python package manager)
- **Pinecone API key**
- **AWS credentials (for SSM)**
- **AI71 API key (for Falcon-180B-chat)**

---

## 📦 Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/dota-rag.git
   cd dota-rag
   ```

2. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables:**
   - Copy the sample environment file:
     ```bash
     cp env_sample .env
     ```
   - Open `.env` and insert your required keys (AWS credentials, Pinecone API key, AI71 API key, etc.).

---

## 🚀 Running the RAG Pipeline

To run the pipeline for a **single query**, execute:

```bash
python main.py
```

This script runs the retrieval and generation pipeline, using Pinecone to find relevant context and Falcon-180B-chat to generate a response.

---

## 📊 Evaluation (Batch Processing)

The repository includes an **evaluation script** that processes multiple questions from a CSV file.

1. **Prepare your CSV file:**  
   Ensure your file is located at:
   ```
   data/eval/data_morgana_examples_live-rag.csv
   ```
   The CSV should contain a column named **`Question`**.

2. **Run the evaluation script:**
   ```bash
   python evaluate.py
   ```

This script will:
✅ Process each question  
✅ Store the generated response in a new column (`response_result`)  
✅ Save the results in a new file:  
```
data/eval/data_morgana_examples_live-rag_results.csv
```

---

## 📂 Repository Structure

```
dota-rag/
├── data/
│   └── eval/
│       └── data_morgana_examples_live-rag.csv  # CSV file with evaluation questions
├── rag/
│   ├── __init__.py         # Package initialization and exports
│   ├── aws_ssm.py          # AWS SSM utilities
│   ├── embedding.py        # Transformer embedding functions
│   ├── pinecone_utils.py   # Pinecone vector search and reranking functions
│   ├── rag_pipeline.py     # RAG pipeline implementation
│   └── rerank.py           # Reranking functions (bge-reranker-v2-m3)
├── env_sample              # Sample environment file (copy to .env)
├── evaluate.py             # Evaluation script to process CSV questions
├── main.py                 # Main script for running the pipeline
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

---

## 📌 Reranking Mechanism

The pipeline **enhances retrieval accuracy** by reranking results with the `bge-reranker-v2-m3` model.

1. Pinecone retrieves top-k documents.
2. The **reranker model** reorders them based on **query relevance**.
3. The **most relevant** documents are fed into the **chat model**.

### Example Usage:

```python
from rag.rerank import rerank_documents

query = "Tell me about the tech company Apple"
documents = [
    "Apple is a fruit with a crisp texture.",
    "Apple Inc. is a technology company that makes the iPhone.",
    "Many people eat apples for their health benefits.",
    "Apple revolutionized the tech industry with its sleek designs."
]

reranked_results = rerank_documents(query, documents, top_n=3)
print(reranked_results)
```

**Output:**
```
[
    {"score": 0.98, "document": "Apple Inc. is a technology company that makes the iPhone."},
    {"score": 0.91, "document": "Apple revolutionized the tech industry with its sleek designs."},
    {"score": 0.75, "document": "Apple is a fruit with a crisp texture."}
]
```

---

## 🔥 Notes & Limitations

- **Token Limits:** Falcon-180B-chat has a **2048-token limit** (input + output). The pipeline **truncates** context to stay within this.
- **Retries & Skipped Questions:** If a question is missing (`NaN`) or invalid, it is **skipped** in evaluation to avoid errors.
- **AI71 API Authentication:** Ensure **API keys** are set in `.env` before running.

---

## 📜 License

This project is licensed under the **MIT License**.

