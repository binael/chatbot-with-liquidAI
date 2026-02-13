import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "LiquidAI/LFM2.5-1.2B-Instruct"
ADAPTER_PATH = "./prime_robotics_lora"

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)

model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.to(device)
model.eval()


def chatbot(prompt: str) -> str:
    """
    Generate response from fine-tuned model.
    """
    messages = [{"role": "user", "content": prompt}]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=60,
            temperature=0.7
        )

    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response.split('\nassistant\n')[0]
