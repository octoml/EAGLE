import os
import json
import openai
import argparse

PROD="https://text.octoai.run"

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="meta-llama-3.1-8b-instruct")
parser.add_argument("--endpoint", type=str, default=PROD)
parser.add_argument("--max-tokens", type=int, default=1024)
parser.add_argument("--temperature", type=float, default=0.0)
args = parser.parse_args()

key = os.environ["OCTOAI_TOKEN"]
endpoint = PROD
client = openai.OpenAI(base_url=f"{endpoint}/v1", api_key=key)

with open("eagle/data/mt_bench/question.jsonl", 'r') as file:
    for line in file:
        data = json.loads(line.strip())
        print(f"Question ID: {data['question_id']}")
        for turn in data['turns']:
            chat_completion = client.chat.completions.create(
                model=args.model,
                messages=[
                    {
                        "role": "user",
                        "content": turn,
                    },
                ],
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
