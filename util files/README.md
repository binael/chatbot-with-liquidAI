# 🛠 Utils – Dataset Preparation & Fine-Tuning

This directory contains **auxiliary utilities** used to:

- 📊 Collect raw data
- 🧠 Expand the dataset using Generative AI
- 🔥 Fine-tune a Large Language Model using LoRA

> ⚠️ These files are **not part of the main application runtime**.
> They were used only during dataset creation and model training.

---

## 📂 Directory Structure

```
utils/
├── webscraper.py
├── genai_augmentation.py
├── fine_tuning.ipynb
├── generated_primerobotics.json
```

---

# 1️⃣ `webscraper.py`

### Purpose

Scrapes structured Question–Answer data from the Prime Robotics website.

### What It Does

- Uses `aiohttp` for asynchronous requests
- Uses `BeautifulSoup` for HTML parsing
- Extracts relevant content from:
  - Home page
  - About page
  - Courses page

- Formats extracted content as:

```json
{
  "question": "...",
  "answer": "..."
}
```

- Saves results to a JSON file.

### Usage

```bash
python webscraper.py
```

### Output

Creates a base dataset used for model fine-tuning.

---

# 2️⃣ `genai_augmentation.py`

### Purpose

Expands the scraped dataset using Google Gemini API.

Since the initial dataset was small, this script generates multiple variations of each question-answer pair.

### Model Used

```
gemini-3-flash-preview
```

### What It Does

For each Q&A pair:

- Rephrases the question
- Rephrases the answer
- Generates multiple variations (minimum 5 per entry)
- Returns structured JSON output
- Appends new data to the dataset

### Environment Variable Required

```
GENAI_API_KEY=your_api_key_here
```

### Usage

```bash
python genai_augmentation.py
```

### Output

Creates:

```
generated_primerobotics.json
```

This file is the final dataset used for training.

---

# 3️⃣ `fine_tuning.ipynb`

### Purpose

Google Colab notebook used to fine-tune the base model using LoRA.

### Base Model

```
LiquidAI/LFM2.5-1.2B-Instruct
```

### Technologies Used

- PyTorch
- HuggingFace Transformers
- PEFT (LoRA)
- BitsAndBytes (optional for quantization)

### Training Steps

1. Load `generated_primerobotics.json`
2. Convert dataset into chat format
3. Apply tokenizer chat template
4. Configure LoRA
5. Fine-tune model
6. Save adapter weights

### Output

Produces a LoRA adapter directory used during inference.

---

# 4️⃣ `generated_primerobotics.json`

This is the final training dataset containing:

- Original scraped data
- AI-generated augmented variations

Format:

```json
[
  {
    "question": "...",
    "answer": "..."
  }
]
```

---

# 🚀 Full Workflow Summary

```
Website → webscraper.py
        ↓
Base Dataset
        ↓
genai_augmentation.py
        ↓
Expanded Dataset (generated_primerobotics.json)
        ↓
fine_tuning.ipynb (Colab)
        ↓
LoRA Adapter Weights
        ↓
Used in Main Application
```

---

# 🧠 Why This Approach?

- Small dataset → AI-based augmentation
- LoRA → Efficient fine-tuning
- Async scraping → Faster data collection
- Structured JSON → Clean supervised training format

---

# 📌 Notes

- These utilities are not required for production inference.
- They are intended for reproducibility and experimentation.
- If retraining is needed, rerun the workflow in order.
