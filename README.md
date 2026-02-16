Here is a **professional main-project README** for your repository, specifically covering:

- `app.py`
- `chatbot.py`
- Static files (HTML, CSS, JS)
- Installation guide
- `requirements.txt`
- Project structure
- Run instructions

You can paste this into your root `README.md`.

---

# 🤖 Prime Robotics AI Chatbot

A fine-tuned Large Language Model (LLM) chatbot built to answer questions about **Prime Robotics**.

The system uses:

- 🧠 LoRA fine-tuned model (`LiquidAI/LFM2.5-1.2B-Instruct`)
- 🔥 PyTorch + HuggingFace Transformers
- ⚡ Flask backend API
- 🎨 HTML + CSS + JavaScript frontend

---

# 📂 Project Structure

```
├── app.py                  # Flask backend server
├── chatbot.py              # Model loading & inference logic
├── templates/
│   └── index.html          # Frontend UI
├── static/
│   ├── styles.css          # Styling
│   └── scripts.js          # Frontend logic (AJAX requests)
├── prime_robotics_lora/    # LoRA adapter weights
├── requirements.txt
└── README.md
```

---

# 🧠 Core Components

---

## 1️⃣ `chatbot.py`

This file handles:

- Loading the base model
- Loading LoRA adapter weights
- Loading tokenizer
- Running inference
- Generating responses

### Base Model

```
LiquidAI/LFM2.5-1.2B-Instruct
```

### Key Responsibilities

- Detects GPU automatically
- Applies chat template
- Generates response using:
  - `temperature=0.5`
  - `do_sample=True`
  - `max_new_tokens=50`

- Cleans assistant output
- Returns formatted response string

This file contains the **main AI logic** of the project.

---

## 2️⃣ `app.py`

This file serves as the backend API using Flask.

### Routes

### `GET /`

Renders the chatbot interface.

### `POST /ask`

Accepts:

```json
{
  "message": "User question here"
}
```

Returns:

```json
{
  "response": "Model generated answer"
}
```

### Responsibilities

- Receives user message
- Calls `chatbot()` function
- Returns JSON response
- Runs Flask server

---

# 🎨 Frontend (Static Files)

Located inside:

```
templates/
static/
```

### `index.html`

- Chat interface UI
- Input box
- Send button
- Displays chat messages

### `styles.css`

- Layout styling
- Chat bubbles
- Responsive design

### `scripts.js`

- Sends POST request to `/ask`
- Displays loading state
- Updates chat window dynamically

The frontend communicates with Flask backend using AJAX (`fetch`).

---

# ⚙️ Installation Guide

---

## 1️⃣ Clone Repository

```bash
git clone https://github.com/yourusername/prime-robotics-chatbot.git
cd prime-robotics-chatbot
```

---

## 2️⃣ Create Virtual Environment (Recommended)

```bash
python -m venv venv
```

Activate:

### Windows

```bash
venv\Scripts\activate
```

### Mac/Linux

```bash
source venv/bin/activate
```

---

## 3️⃣ Install Dependencies

Create a file called:

```
requirements.txt
```

With the following contents:

```
torch
transformers
peft
flask
accelerate
bitsandbytes
```

Then install:

```bash
pip install -r requirements.txt
```

---

# 🚀 Running the Application

```bash
python app.py
```

Server runs at:

```
http://localhost:5000
```

Open it in your browser.

---

# 💻 GPU Support

If CUDA is available, the model automatically runs on GPU.

Otherwise, it falls back to CPU.

---

# 🧠 Model Details

- Base Model: `LiquidAI/LFM2.5-1.2B-Instruct`
- Fine-tuning method: LoRA (PEFT)
- Only adapter weights stored locally
- Efficient inference
- Low memory usage

---

# 🔄 Request Flow

```
User (Browser)
      ↓
Frontend (HTML/JS)
      ↓
POST /ask
      ↓
Flask (app.py)
      ↓
chatbot.py
      ↓
Model.generate()
      ↓
Response JSON
      ↓
Frontend display
```

---

# 🧪 Example API Call

Using curl:

```bash
curl -X POST http://localhost:5000/ask \
     -H "Content-Type: application/json" \
     -d '{"message":"What courses does Prime Robotics offer?"}'
```

---

# 📌 Notes

- Model may require significant RAM if running on CPU.
- For production deployment, use:
  - Gunicorn
  - Docker
  - Nginx reverse proxy

---

# 🔥 Future Improvements

- Streaming responses
- Conversation memory
- Rate limiting
- Logging
- Docker containerization
- HuggingFace deployment

---

# 👨‍💻 Author

Built as an end-to-end fine-tuned LLM application:

- Dataset scraping
- Dataset augmentation
- LoRA fine-tuning
- Flask deployment
- Custom frontend
