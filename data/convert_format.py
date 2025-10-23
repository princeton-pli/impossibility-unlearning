import os
import json
from datasets import load_dataset
from tqdm import tqdm
import random

def create_chat_message(question, answer):
    """Creates a conversation in the OpenAI chat format."""
    res = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": question},
    {"role": "assistant", "content": answer}
    ]
    return res

def main(args):
    dataset = json.load(open(args.input_file,"r"))
    for entry in dataset:
        question = entry[args.question_key]
        answer = entry[args.answer_key]
        convo = create_chat_message(question, answer)
        entry["conversations"] = convo

    with open(args.output_file,"w") as f:
        json.dump(dataset, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--question_key", type=str, default="input")
    parser.add_argument("--answer_key", type=str, default="output")
    args = parser.parse_args()
    main(args)

