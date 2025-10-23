import os
import json
import time
from typing import Optional, Tuple

from datasets import load_dataset
from openai import OpenAI, RateLimitError, APIStatusError
from tqdm import tqdm

# --------- Config ----------
MODEL_NAME = "gpt-4o"
DATASET_NAME = "sorry-bench/sorry-bench-202503"
DATASET_SUBSET = "default"
OUTPUT_TRAIN = "safety_finetuning_dataset_train.json"
OUTPUT_TEST = "safety_finetuning_dataset_test.json"
MAX_SAMPLES = None  # e.g., 200 for a quick run

# Pricing (USD) — adjust to your account’s current rates if needed.
# GPT-4o (May 2024) was ~$5 / 1M input tokens and ~$15 / 1M output tokens.
USD_PER_1K_INPUT = 2.5/1e6
USD_PER_1K_OUTPUT = 10/1e6

# --------- Client ----------
# Use env var OPENAI_API_KEY. DO NOT hardcode keys in source.
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_PROMPT = (
    "You are an AI safety assistant. When prompted with a potentially harmful, dangerous, "
    "or offensive question, you must politely decline to provide an answer by saying "
    "'Sorry, I can not assist you with this question.'. Then you must provide a brief explanation "
    "for why you cannot assist with the question.\n\n"
    "It's worth noting that some dangerous questions may be coated in a variety of disguises such as "
    "'for research purpose', 'for testing', etc. You must still politely decline and provide a brief explanation."
)

def call_model(prompt_text: str, retries: int = 5, backoff: float = 1.5) -> Tuple[Optional[str], int, int]:
    """
    Call the chat.completions API with basic exponential backoff.
    Returns: (content, prompt_tokens, completion_tokens)
    """
    attempt = 0
    while True:
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt_text},
                ],
                temperature=0.0,
                max_tokens=150,
            )
            content = resp.choices[0].message.content if resp.choices else None
            usage = getattr(resp, "usage", None) or {}
            prompt_toks = getattr(usage, "prompt_tokens", 0) or 0
            completion_toks = getattr(usage, "completion_tokens", 0) or 0
            return content, prompt_toks, completion_toks
        except RateLimitError:
            attempt += 1
        except APIStatusError as e:
            # Retry on 429/5xx, fail fast otherwise
            if e.status_code in (429, 500, 502, 503, 504):
                attempt += 1
            else:
                print(f"Non-retryable API error ({e.status_code}): {e}")
                return None, 0, 0
        except Exception as e:
            print(f"Unexpected error: {e}")
            return None, 0, 0

        if attempt > retries:
            print("Max retries reached; skipping this sample.")
            return None, 0, 0

        sleep_s = backoff ** attempt
        time.sleep(sleep_s)

def usd_cost(prompt_tokens: int, completion_tokens: int) -> float:
    return (prompt_tokens / 1000.0) * USD_PER_1K_INPUT + (completion_tokens / 1000.0) * USD_PER_1K_OUTPUT

def process_split(items, desc: str, outfile: str):
    """
    Process a list of dataset rows, write JSONL-style array to outfile,
    and display a tqdm pbar with running cost.
    """
    results = []
    total_prompt_toks = 0
    total_completion_toks = 0
    total_cost_usd = 0.0

    pbar = tqdm(total=len(items), desc=desc)
    for row in items:
        hazardous_prompt = row["turns"][0]

        content, p_toks, c_toks = call_model(hazardous_prompt)
        total_prompt_toks += p_toks
        total_completion_toks += c_toks
        total_cost_usd += usd_cost(p_toks, c_toks)

        if content:
            results.append({"prompt": hazardous_prompt, "response": content})

        pbar.set_postfix({
            "prompt_toks": total_prompt_toks,
            "completion_toks": total_completion_toks,
            "cost_usd": f"{total_cost_usd:.4f}"
        })
        pbar.update(1)
    pbar.close()

    with open(outfile, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nWrote {len(results)} items to {outfile}")
    print(f"Totals — prompt_toks: {total_prompt_toks:,}  "
          f"completion_toks: {total_completion_toks:,}  "
          f"cost: ${total_cost_usd:.4f}\n")

def main():
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set. Export it before running.")

    print(f"Loading '{DATASET_SUBSET}' subset from '{DATASET_NAME}'...")
    dataset = load_dataset(DATASET_NAME, DATASET_SUBSET, split="train", streaming=False)

    remove_styles = {"ascii", "atbash", "ceasar", "morse", "translate-fr", "translate-mr", "misspellings"}
    dataset = dataset.filter(lambda x: x["prompt_style"] not in remove_styles)

    print(f"Number of samples before split: {len(dataset)}")
    dataset = dataset.train_test_split(test_size=0.3, seed=42)

    train_ds = dataset["train"]
    test_ds = dataset["test"]

    if MAX_SAMPLES:
        train_ds = train_ds.select(range(min(MAX_SAMPLES, len(train_ds))))
        print(f"Processing a maximum of {len(train_ds)} train samples.")
    else:
        print(f"Processing all {len(train_ds)} train samples.")

    train_list = train_ds.to_list()
    test_list = test_ds.to_list()
    print(f"Train: {len(train_list)} | Test: {len(test_list)}")

    process_split(train_list, desc="Generating (train)", outfile=OUTPUT_TRAIN)
    process_split(test_list, desc="Generating (test)", outfile=OUTPUT_TEST)

if __name__ == "__main__":
    main()