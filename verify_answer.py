import pandas as pd
import jsonschema
from jsonschema import validate
import json
import argparse

# LiveRAG Answer JSON schema: 
json_schema = """
{ 
"$schema": "http://json-schema.org/draft-07/schema#", 

  "title": "Answer file schema", 
  "type": "object", 
  "properties": { 
    "id": { 
      "type": "integer", 
      "description": "Question ID" 
    }, 
    "question": { 
      "type": "string", 
      "description": "The question" 
    }, 
    "passages": { 
      "type": "array", 
      "description": "Passages used and related FineWeb doc IDs, ordered by decreasing importance", 
      "items": { 
        "type": "object", 
        "properties": {
          "passage": { 
            "type": "string", 
            "description": "Passage text" 
          }, 
          "doc_IDs": {
            "type": "array", 
            "description": "Passage related FineWeb doc IDs, ordered by decreasing importance", 
            "items": { 
              "type": "string", 
              "description": "FineWeb doc ID, e.g., <urn:uuid:d69cbebc-133a-4ebe-9378-68235ec9f091>"
            } 
          } 
        },
        "required": ["passage", "doc_IDs"]
      }
    }, 
    "final_prompt": {
      "type": "string",
      "description": "Final prompt, as submitted to Falcon LLM"
    },
    "answer": {
      "type": "string",
      "description": "Your answer"
    }
  },
  "required": ["id", "question", "passages", "final_prompt", "answer"]
}
"""

# Load file
parser = argparse.ArgumentParser(description="Process a JSONL file.")
parser.add_argument(
    "input_file",
    type=str,
    help="Path to the input JSONL file",
)

args = parser.parse_args()

# Load the file to make sure it is ok
loaded_answers = pd.read_json(args.input_file, lines=True)

# Load the JSON schema
schema = json.loads(json_schema)

# Validate each Answer JSON object against the schema
for answer in loaded_answers.to_dict(orient='records'):
    try:
        validate(instance=answer, schema=schema)
        print(f"Answer {answer['id']} is valid.")
    except jsonschema.exceptions.ValidationError as e:
        print(f"Answer {answer['id']} is invalid: {e.message}")