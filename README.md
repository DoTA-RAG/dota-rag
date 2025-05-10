# DoTA-RAG Pipeline

## 🚀 Quick Start

### 1. Clone & install

```bash
git clone https://github.com/<your‑gh‑user>/dota‑rag.git
cd dota‑rag
pip install -r requirements.txt
```

### 2. Configure secrets

Copy the sample env file and fill in your keys:

```bash
cp env_sample .env
# edit .env → add AWS, Pinecone, and AI71 keys
```

### 3. Run a single query

```bash
python main.py
```

You will be prompted for a question and receive a model‑generated answer with supporting passages.

### 4. Batch evaluation

```bash
# inference
python get_response.py \
  --input-file data/testset/testset-50q.jsonl \
  --mode routing_v6_final \
  --concurrency 8

# verify
python verify_answer.py data/out/testset-50q-result.jsonl

# llm judge with falcon
python utils/eval/evaluate-falcon.py \
  --input_file data/out/testset-50q-result.jsonl \
  --eval_name both

# llm judge with claude
python utils/eval/evaluate.py \
  --input_file data/out/testset-50q-result.jsonl \
  --eval_name both
```