import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse

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

def append_user_prompt(history, input_prompt):
    user_prompt = f"{B_INST} {input_prompt} {E_INST}"
    history += user_prompt
    return history

def append_bot_prompt(history, input_prompt):
    user_prompt = f"{B_SYS} {input_prompt}{E_SYS} {E_SYS}"
    history += user_prompt
    return history

class SimpleLLM(object):
    def __init__(self, hf_model_name, hf_auth_token, stop_token, device="cpu", max_num_tokens=1028):
        self.stop_token = stop_token
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(
            hf_model_name,
            torch_dtype=torch.float16,
            token=hf_auth_token,
            device_map=device
        )
        self.max_num_tokens = max_num_tokens

    def format_out(self, input_id):
        return int(input_id[0].detach().cpu())

    def generate(self, input_ids):
        output_tokens = []
        if self.device != "cpu":
            input_ids = input_ids.to(self.device)
        result = self.model(input_ids)
        next_token = torch.argmax(result.logits[:, -1, :], dim=1)
        output_tokens.append(self.format_out(next_token))
        pkv = result.past_key_values
        next_token = next_token.unsqueeze(-1)
        for _ in range(self.max_num_tokens):
            result = self.model(next_token, past_key_values=pkv)
            next_token = torch.argmax(result.logits[:, -1, :], dim=1)
            output_tokens.append(self.format_out(next_token))
            next_token = next_token.unsqueeze(-1)
            if self.format_out(next_token) == self.stop_token:
                break
            pkv = result.past_key_values
        return output_tokens

    def to(self, device):
        self.device = device
        self.model.to(self.device)

def chat(hf_model_name,
         hf_auth_token,
         max_num_tokens,
         device):
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
    prompt = DEFAULT_CHAT_SYS_PROMPT
    while True:
        user_prompt = input("User prompt: ")
        prompt = append_user_prompt(prompt, user_prompt)
        initial_input = tokenizer(prompt, return_tensors="pt")
        example_input_id = initial_input.input_ids
        result = llm.generate(example_input_id)
        bot_response = tokenizer.decode(result, skip_special_tokens=True)
        print(f"\nBOT: {bot_response}\n")
        prompt = append_bot_prompt(prompt, bot_response)

if __name__ == "__main__":
    args = parser.parse_args()
    chat(args.hf_model_name, args.hf_auth_token, args.max_num_tokens, args.device)
