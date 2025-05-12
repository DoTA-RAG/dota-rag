python get_response.py \
  --input-file data/testset/LiveRAG_LCD_Session1_Question_file.jsonl \
  --mode routing_v6_final \
  --concurrency 8

python get_response.py \
  --input-file data/testset/rerun.jsonl \
  --mode routing_v6_final \
  --concurrency 8

# verify
python verify_answer.py data/out/LiveRAG_LCD_Session1_Question_file-result.jsonl

# llm judge with falcon
python utils/eval/evaluate-falcon.py \
  --input_file data/out/LiveRAG_LCD_Session1_Question_file-result_v1_submit_final.jsonl \
  --eval_name faithfulness