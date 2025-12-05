RLHF PoC — Reinforcement Learning from Human Feedback (Azure OpenAI + Streamlit)

A lightweight proof-of-concept demonstrating the principles of Reinforcement Learning from Human Feedback (RLHF) using Azure OpenAI for text generation and a tiny CPU-compatible reward model for learning human preferences.

Built for Python 3.11, works fully on CPU (no GPU required).

🚀 Features

✅ Generate multiple AI responses for a given prompt using Azure OpenAI
✅ Label your preferred responses (human feedback collection)
✅ Train a lightweight reward model (TinyRewardModel) on CPU
✅ Score and rerank new outputs by predicted human preference
✅ Visualize and download your preference dataset
✅ Secure .env-based credential management
✅ Minimal resource usage — ideal for laptops and dev environments

🧠 Architecture Overview

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


This project replicates the core RLHF feedback-training loop in a simplified form.

📦 Project Structure
├── ui.py                   # Streamlit app (main UI)
├── reward.py               # Reward model logic
├── data/
│   ├── preferences.jsonl   # Human feedback data
│   └── reward_model/       # Trained reward model files
├── .env                    # Azure credentials
├── requirements.txt         # Dependencies
├── .gitignore              # Ignore unnecessary files
└── README.md               # Documentation

⚙️ Setup Instructions

1️⃣ Clone the Repository
git clone https://github.com/<your-username>/rlhf-azure-poc.git
cd rlhf-azure-poc

2️⃣ Create a Virtual Environment
python -m venv .venv
source .venv/bin/activate    # macOS/Linux
.venv\Scripts\activate       # Windows

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Configure Azure Environment Variables

Create a .env file in your project root:

AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
AZURE_OPENAI_API_KEY=your-azure-api-key
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o-mini
AZURE_OPENAI_API_VERSION=2024-08-01-preview

▶️ Run the App

Launch the Streamlit web interface:
streamlit run ui.py

Then open your browser at:

http://localhost:8501

🧩 Usage Guide

💬 Generate

Enter any prompt and generate multiple responses from your Azure OpenAI model.
Example:

“Explain Newton’s First Law to a 10-year-old.”

✅ Label

Select your preferred response and optionally add a reason.
Feedback is stored in data/preferences.jsonl.

📈 Train Reward Model

Train a lightweight reward model that learns your preferences using:

SentenceTransformers (all-MiniLM-L6-v2) for embeddings

Logistic Regression for classification

🏅 Rerank

Use your trained reward model to rank new generations by predicted “human preference.”

🗂 Dataset

Browse and download all labeled preference data.

🧠 Reward Model Details

Implemented in reward.py :

Embeddings: all-MiniLM-L6-v2 (SentenceTransformers)
Classifier: Logistic Regression (Scikit-learn)
Device: CPU (safe for low-resource systems)

Input: (Prompt + Response) text pairs
Output: Probability of human preference

Model is saved at:
data/reward_model/reward_model.joblib


You can reload it anytime:

from reward import TinyRewardModel
rm = TinyRewardModel.load("data/reward_model")

Example Prompts for Testing:

Category---Example Prompt
Education---Explain gravity like I’m 5 years old.
Professional---Write a thank-you email after a job interview.
Creative---Describe the ocean as if it could talk.
Technical---Write a Python function to check if a number is prime.
Empathy---How would you comfort a friend who failed an exam?