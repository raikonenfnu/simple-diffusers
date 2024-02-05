import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
from simple_llm import SimpleLLM

DEFAULT_HF_MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<s>", "</s>"
DEFAULT_CHAT_SYS_PROMPT = """<s>[INST] <<SYS>>
Be concise. You are a helpful, respectful and honest assistant. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n <</SYS>>\n\n
"""
MAX_NUM_TOKENS=1028

parser = argparse.ArgumentParser()
parser.add_argument(
    "--hf_auth_token",
    type=str,
    default="",
    help="The Hugging face auth token, required for some models",
)
parser.add_argument(
    "--hf_model_name",
    type=str,
    help="HF model name",
    default="meta-llama/Llama-2-7b-chat-hf",
)
parser.add_argument(
    "--max_num_tokens",
    type=int,
    default=1024,
    help="Max number of tokens generated per prompt",
)
parser.add_argument(
    "--device",
    type=str,
    help="Device to run models on.",
    default="cpu",
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=1,
    help="number of prompts to be decoded parallely.",
)


PROMPT_REQUESTS = \
[
    "How do I open a can of beans?",
    "How do I open a can of soup?",
    "How do I open a can of strawberry jam?",
    "How do I open a can of raspberry jam?",
    "What's the tallest building in Paris?",
    "What's the most populous nation on Earth?",
    "What's the most populous nation on Mars?",
    "What do the Mole People actually want and how can we best appease them?",
    "Why is the sky blue?",
    "Where is Waldo?",
    "Who is Waldo?",
    "Why is Waldo?",
    "Is it legal to base jump off the Eiffel Tower?",
    "Is it legal to base jump into a volcano?",
    "Why are cats better than dogs?",
    "Why is the Hulk so angry all the time?",
    "How do I build a time machine?",
    "Is it legal to grow your own catnip?"
]

def append_user_prompt(history, input_prompt):
    user_prompt = f"{B_INST} {input_prompt} {E_INST}"
    history += user_prompt
    return history

def append_bot_prompt(history, input_prompt):
    user_prompt = f"{B_SYS} {input_prompt}{E_SYS} {E_SYS}"
    history += user_prompt
    return history

def parallel_decode(hf_model_name,
                    hf_auth_token,
                    max_num_tokens,
                    device,
                    batch_size):
    tokenizer = AutoTokenizer.from_pretrained(
        hf_model_name,
        use_fast=False,
        token=hf_auth_token,
    )
    llm = SimpleLLM(hf_model_name,
                    hf_auth_token,
                    tokenizer.eos_token_id,
                    device=device,
                    max_num_tokens=max_num_tokens)
    # Sort S.T the padding within a batch will be minimal.
    init_prompt = DEFAULT_CHAT_SYS_PROMPT
    sorted_prompt_requests = sorted(PROMPT_REQUESTS, key = len)
    formatted_prompt_requests = [append_user_prompt(init_prompt, prompt_req) for prompt_req in sorted_prompt_requests]
    batched_prompt_requests = [formatted_prompt_requests[i:i + batch_size] for i in range(0, len(formatted_prompt_requests), batch_size)]
    for batch_prompt_req_iter in batched_prompt_requests:
        initial_input = tokenizer(batch_prompt_req_iter, return_tensors="pt")
        example_input_id = initial_input.input_ids
        result = llm.generate(example_input_id)
        bot_response = tokenizer.decode(result, skip_special_tokens=True)
        print(f"\nBOT: {bot_response}\n")
        prompt = append_bot_prompt(prompt, bot_response)

if __name__ == "__main__":
    args = parser.parse_args()
    parallel_decode(args.hf_model_name, args.hf_auth_token, args.max_num_tokens, args.device, args.batch_size)
