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

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            do_sample=True,
            top_p=0.9
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response
