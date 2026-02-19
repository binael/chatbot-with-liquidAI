# Prime Robotics Domain Assistant

LoRA Fine-Tuning of a 1B+ Instruction LLM (LiquidAI Instruct)

---

## 1️ Project Overview

This project demonstrates **efficient domain adaptation** of an instruction-tuned Large Language Model using **LoRA (Low-Rank Adaptation)**.

The goal was to build a lightweight domain-specific assistant capable of answering questions about Prime Robotics programs and offerings.

Instead of full fine-tuning, this project uses parameter-efficient training via PEFT to reduce GPU cost and training time.

---

## 2️⃣ Architecture Overview

**Pipeline:**

1. Website Scraping
2. Dataset Cleaning & Structuring
3. Data Augmentation (Q&A paraphrasing via Google GenAI API)
4. LoRA Fine-Tuning
5. Adapter Integration
6. Flask API Deployment
7. Lightweight Frontend Interface


```
Website Data → Q&A Dataset → LoRA Fine-Tuning → Adapter → Flask API → UI
```

---

## 3️⃣ Model Details

* Base Model: (Insert exact model name, e.g. `LiquidAI/LFM2.5-1.2B-Instruct`)
* Parameter Size: ~1B+
* Training Strategy: LoRA (via HuggingFace PEFT)
* Fine-Tuning Type: Supervised Instruction Tuning

### LoRA Configuration

* r: 16 (or your value)
* alpha: 32
* target_modules: ["q_proj", "v_proj"]
* dropout: 0.05
* bias: none

---

## 4️⃣ Dataset

### Data Sources

* Scraped structured content from official Prime Robotics website
* Augmented with paraphrased Q&A pairs generated using Google GenAI API

### Dataset Characteristics

* Domain-specific
* Clean Q&A format
* Small but focused
* Instruction-style conversational format

Example training sample:

```json
{
  "question": "What programs does Prime Robotics offer?",
  "answer": "Prime Robotics offers robotics training programs for..."
}
```

---

## 5️⃣ Training Setup

* Framework: HuggingFace Transformers
* PEFT: LoRA adapters
* Hardware: CPU
* Epochs: 5
* Learning Rate: 9e-4
* Batch Size: 16

> LoRA freezes the base model weights and trains only lightweight adapter layers, reducing memory consumption and enabling efficient experimentation.

---

## 6️⃣ Inference & Deployment

After training:

* The LoRA adapter is merged or loaded on top of the base model
* A simple Flask API serves responses
* A minimal frontend handles user interaction

---

## 7️⃣ Results & Observations

* Improved domain-specific accuracy
* Reduced hallucinations within domain scope
* Lightweight deployment
* Fast inference

Limitations:

* Small dataset
* No large-scale benchmarking
* No RLHF or safety tuning
* Not production-hardened
* Prone to Overfitting to ensure output

---

## 8️⃣ How to Run

### Clone Repo

```bash
git clone https://github.com/binael/chatbot-with-liquidAI
cd chatbot-with-liquidAI
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run API

```bash
python app.py
```

---

## 9️⃣ Key Takeaways

* Efficient fine-tuning does not require massive infrastructure.
* Focused data + LoRA can meaningfully adapt LLM behavior.
* Small domain assistants are practical and cost-effective.

---

## 🔟 Future Improvements

* Evaluation metrics (BLEU / ROUGE / Human eval)
* Quantization for faster inference
* Better prompt formatting
* Guardrails & safety layer
* Retrieval augmentation

---

## Lessons Learned

* Data quality mattered more than dataset size.
* LoRA dramatically reduced GPU requirements.
* Clear instruction formatting improved response quality.
