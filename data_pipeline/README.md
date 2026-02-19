# Prime Robotics — LLM Fine-Tuning Dataset Pipeline

This project builds a domain-specific fine-tuning dataset for
Prime Robotics
using a lightweight two-stage pipeline:

1. **Asynchronous Web Scraping**
2. **Structured Generative Data Augmentation**

The resulting dataset is designed for efficient LoRA-based LLM fine-tuning.

---

# Objective

Create a clean, domain-focused Q&A dataset suitable for:

* Instruction tuning
* LoRA adaptation
* Small-scale assistant deployment
* Controlled domain grounding

---

# Architecture Overview

```
Website → Async Scraper → Structured Q&A
              ↓
      Generative Expansion (Gemini)
              ↓
     Fine-Tuning Ready Dataset
```

---

# Module 1 — Asynchronous Scraper

Extracts structured content from:

* Home page
* About page
* Courses page

### Key Features

* `aiohttp` + `asyncio.gather` for concurrent scraping
* BeautifulSoup parsing
* Structured Q&A formatting
* Clean JSON output

### Output

`primerobotics.json`

```json
[
  {
    "question": "What courses are available?",
    "answer": "..."
  }
]
```

Engineering signal:

* Non-blocking I/O
* Separation of concerns
* Type annotations
* Clean dataset formatting

---

# 📦 Module 2 — Generative Data Expansion

Expands dataset using
Google Generative AI (Gemini).

For each Q&A pair:

* Generates ≥5 paraphrased variations
* Enforces structured JSON schema
* Validates using Pydantic models
* Handles API failures with retry logic

### Output

`generated_primerobotics.json`

Engineering signal:

* Structured AI generation
* Schema-constrained outputs
* Controlled rate limiting
* Defensive error handling

---

# Technical Highlights

* Async scraping for performance
* Structured data transformation
* Schema-validated AI generation
* Modular pipeline design
* Clean JSON artifacts for fine-tuning

---

# Dataset Flow

| Stage    | File                           | Purpose                    |
| -------- | ------------------------------ | -------------------------- |
| Scraped  | `primerobotics.json`           | Raw structured data        |
| Expanded | `generated_primerobotics.json` | Augmented training dataset |

---

# Why This Matters

Instead of collecting massive data, this pipeline:

* Focuses on domain clarity
* Uses generative augmentation strategically
* Produces clean instruction-style data
* Enables efficient LoRA fine-tuning

This demonstrates practical AI engineering — not just model training.

---

# Next Steps

* Deduplication layer
* Automated evaluation metrics
* Retrieval augmentation
* Production API integration