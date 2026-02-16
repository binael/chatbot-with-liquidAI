"""
Inference script for Prime Robotics fine-tuned LLM using LoRA.

This module loads:
- A base LiquidAI causal language model
- A LoRA adapter (fine-tuned weights)
- Corresponding tokenizer

It exposes a `chatbot` function that generates responses
from the fine-tuned model using a chat template format.

Model
-----
Base Model:
    LiquidAI/LFM2.5-1.2B-Instruct

Adapter:
    Local LoRA adapter stored at ADAPTER_PATH

Device
------
Automatically selects CUDA if available, otherwise CPU.
"""

from typing import Dict, List

import torch
from torch import Tensor
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


BASE_MODEL: str = "LiquidAI/LFM2.5-1.2B-Instruct"
ADAPTER_PATH: str = "./prime_robotics_lora"

device: str = "cuda" if torch.cuda.is_available() else "cpu"


# Load tokenizer from adapter directory
tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
)

# Load LoRA adapter into base model
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.to(device)
model.eval()


def chatbot(prompt: str) -> str:
    """
    Generate a response from the fine-tuned chatbot model.

    Parameters
    ----------
    prompt : str
        User input message.

    Returns
    -------
    str
        Generated assistant response text.

    Notes
    -----
    - Uses sampling (`do_sample=True`) for more natural responses.
    - Temperature is set to 0.5 for controlled randomness.
    - Limits output to 50 new tokens.
    - Automatically removes special tokens.
    """
    messages: List[Dict[str, str]] = [
        {"role": "user", "content": prompt}
    ]

    inputs: Tensor = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        output: Tensor = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.5,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    response: str = tokenizer.decode(
        output[0],
        skip_special_tokens=True,
    )

    # Extract assistant portion from chat-formatted output
    return response.split("\nassistant\n")[-1].strip()
