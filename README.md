# 🚀 RLHF PoC — Reinforcement Learning from Human Feedback (Azure OpenAI + Streamlit)

A lightweight proof-of-concept demonstrating the core ideas behind **Reinforcement Learning from Human Feedback (RLHF)** using **Azure OpenAI** for text generation and a **tiny CPU-friendly reward model** to learn human preferences.

Built for **Python 3.11** — fully **CPU compatible** (no GPU required).

---

## ✨ Features

- Generate multiple AI responses using Azure OpenAI  
-  Collect human preference labels  
-  Train a lightweight reward model on CPU  
-  Rerank AI outputs using predicted human preference  
-  Visualize and download feedback datasets  
-  Secure credential handling via `.env`  
-  Extremely lightweight — runs on any laptop  

---

## 🧠 Architecture Overview
```
User Prompt ─► Azure GPT (generates responses)
             │
             ▼
   Human chooses best (Label tab)
             │
             ▼
 Reward Model learns from feedback
             │
             ▼
 New outputs reranked by reward score
```

This demonstrates the **core RLHF loop** in a simplified, developer-friendly format.

---

## 📦 Project Structure
```
├── ui.py                   # Streamlit app (main UI)
├── reward.py               # Reward model logic
├── data/
│   ├── preferences.jsonl   # Human feedback data
│   └── reward_model/       # Trained reward model files
├── .env                    # Azure credentials
├── requirements.txt         # Dependencies
├── .gitignore              # Ignore unnecessary files
└── README.md               # Documentation
```

---

## ⚙️ Setup Instructions

### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/<your-username>/rlhf-azure-poc.git
cd rlhf-azure-poc
```
2️⃣ Create a Virtual Environment
```
python -m venv .venv
source .venv/bin/activate    # macOS/Linux
.venv\Scripts\activate       # Windows
```

3️⃣ Install Dependencies
```
pip install -r requirements.txt
```
4️⃣ Configure Azure Environment Variables
Create a .env file in your project root:
```
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
AZURE_OPENAI_API_KEY=your-azure-api-key
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o-mini
AZURE_OPENAI_API_VERSION=2024-08-01-preview
```
▶️ Run the App
Launch the Streamlit web interface:
```
streamlit run ui.py
```

Then open your browser:
```
http://localhost:8501
```
---

## 🧩 Usage Guide
### 💬 Generate
Enter any prompt and generate multiple responses.
Example:
“Explain Newton’s First Law to a 10-year-old.”

### ✅ Label
Select your preferred response and add a reason.
Feedback is stored in:
```
data/preferences.jsonl.
```

### 📈 Train Reward Model

- A minimal RLHF reward model using:
- SentenceTransformers (all-MiniLM-L6-v2) for embeddings
- Logistic Regression for classification
- Runs fully on CPU

### 🏅 Rerank
Score new AI outputs using the trained reward model and reorder by predicted human preference.

### 🗂 Dataset
Browse and download all labeled preference data.

### 🧠 Reward Model Details
Implemented in reward.py :

- Embeddings: all-MiniLM-L6-v2 (SentenceTransformers)
- Classifier: Logistic Regression (Scikit-learn)
- Device: CPU (safe for low-resource systems)
- Input: (Prompt + Response) text pairs
- Output: Probability of human preference

Model is saved at:
```
data/reward_model/reward_model.joblib
```
Load example:
```
from reward import TinyRewardModel
rm = TinyRewardModel.load("data/reward_model")
```
---

Example Prompts for Testing:

- Category---Example Prompt
- Education---Explain gravity like I’m 5 years old.
- Professional---Write a thank-you email after a job interview.
- Creative---Describe the ocean as if it could talk.
- Technical---Write a Python function to check if a number is prime.
- Empathy---How would you comfort a friend who failed an exam?

---

📜 License
MIT License.
