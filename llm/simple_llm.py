import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

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

