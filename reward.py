"""
reward.py — TinyRewardModel for RLHF PoC
----------------------------------------
A lightweight CPU-compatible reward model that learns
from human preference data.

Uses SentenceTransformer for text embeddings and
Logistic Regression for binary classification.

Works perfectly with ui.py (Streamlit RLHF PoC)
and Azure OpenAI completions.
"""

import os
import numpy as np
import joblib
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class TinyRewardModel:
    """
    A simple reward model that learns human preferences between candidate responses.
    Trains on (prompt + response) pairs using embeddings and logistic regression.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedding model and classifier.
        Forces CPU execution to avoid meta tensor errors.
        """
        print("🔧 Initializing TinyRewardModel... (CPU mode)")
        self.device = "cpu"  # Force CPU mode to avoid 'meta tensor' issue
        self.model = SentenceTransformer(model_name, device=self.device)
        self.classifier = LogisticRegression(max_iter=300)

    # ------------------------------------------------------------
    # Embedding utilities
    # ------------------------------------------------------------
    def _embed(self, prompt: str, response: str) -> np.ndarray:
        """
        Generate embedding for a (prompt, response) pair.
        The model encodes combined semantic meaning.
        """
        text = f"Prompt: {prompt}\nResponse: {response}"
        emb = self.model.encode([text], convert_to_numpy=True, show_progress_bar=False)
        return emb[0]

    # ------------------------------------------------------------
    # Training pipeline
    # ------------------------------------------------------------
    def fit_from_preferences(self, prefs: List[Dict]) -> Dict[str, float]:
        """
        Train the reward model from a list of human preference entries.

        Each entry format:
        {
            "prompt": "...",
            "responses": [{"text": "..."}],
            "chosen_index": 0
        }
        """
        print(f"🧠 Training on {len(prefs)} labeled examples...")

        X, y = [], []
        for pref in prefs:
            prompt = pref.get("prompt", "")
            responses = pref.get("responses", [])
            chosen_idx = pref.get("chosen_index")

            if chosen_idx is None or len(responses) < 2:
                continue

            # Embed chosen response
            chosen_resp = responses[chosen_idx]["text"]
            chosen_emb = self._embed(prompt, chosen_resp)
            X.append(chosen_emb)
            y.append(1)  # Preferred

            # Embed one non-chosen response (negative example)
            for i, r in enumerate(responses):
                if i != chosen_idx:
                    neg_emb = self._embed(prompt, r["text"])
                    X.append(neg_emb)
                    y.append(0)  # Not preferred

        X = np.array(X)
        y = np.array(y)

        if len(X) == 0:
            raise ValueError("❌ No valid labeled data found to train reward model.")

        # Train/test split for evaluation
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.classifier.fit(X_train, y_train)
        preds = self.classifier.predict(X_test)
        acc = accuracy_score(y_test, preds)

        print(f"✅ Reward model trained. Test accuracy: {acc:.3f}")
        return {"test_accuracy": acc, "train_size": len(X_train), "test_size": len(X_test)}

    # ------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------
    def score(self, prompt: str, response: str) -> float:
        """
        Compute a reward score (probability that response is preferred).
        """
        emb = self._embed(prompt, response)
        score = float(self.classifier.predict_proba([emb])[0][1])
        return score

    # ------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------
    def save(self, folder: str | os.PathLike) -> None:
        """
        Save model weights and embeddings to a directory.
        """
        os.makedirs(folder, exist_ok=True)
        joblib.dump(self.classifier, os.path.join(folder, "reward_model.joblib"))
        print(f"💾 Reward model saved to: {folder}")

    @classmethod
    def load(cls, folder: str | os.PathLike) -> "TinyRewardModel":
        """
        Load reward model from a directory.
        """
        print(f"📦 Loading reward model from {folder}...")
        model = cls()
        model.classifier = joblib.load(os.path.join(folder, "reward_model.joblib"))
        print("✅ Reward model loaded successfully.")
        return model
