import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


def load_model_config():
    base = "LiquidAI/LFM2.5-1.2B-Instruct"
    adapter = "./prime_robotics_lora"

    tokenizer = AutoTokenizer.from_pretrained(adapter)

    model = AutoModelForCausalLM.from_pretrained(
        base,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        # Removed device_map="auto"
    )

    model = PeftModel.from_pretrained(model, adapter)

    # Explicitly move model to GPU if available after loading the adapter
    if torch.cuda.is_available():
        model = model.to("cuda")

    print(" Model loaded successfully\n")

    model.eval()

    return (tokenizer, model)
