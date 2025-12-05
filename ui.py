"""
ui.py — Streamlit RLHF PoC using Azure OpenAI (Python 3.11)

Key features:
- Credentials (endpoint, API key, deployment, version) loaded from .env
- Sidebar shows only generation settings
- Supports Azure API version
- Streamlined production-style RLHF demo
- ✨ NEW: Complete Journey tab tracks full workflow
"""

from __future__ import annotations

import os
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv

from reward import TinyRewardModel

# Load environment variables
load_dotenv()

API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")

DATA_DIR = Path(__file__).parent / "data"
PREFS_PATH = DATA_DIR / "preferences.jsonl"
REWARD_DIR = DATA_DIR / "reward_model"
JOURNEY_PATH = DATA_DIR / "complete_journey.jsonl"  # NEW: Track complete workflow

DATA_DIR.mkdir(parents=True, exist_ok=True)
REWARD_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("rlhf-ui")

# Helper Functions
def ensure_azure_config() -> None:
    """Ensure Azure credentials exist."""
    if not (ENDPOINT and API_KEY and DEPLOYMENT_NAME and API_VERSION):
        raise EnvironmentError(
            "❌ Missing Azure OpenAI credentials. Please set them in your .env file."
        )


def append_jsonl(path: Path, obj: Dict) -> None:
    """Append one record to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> List[Dict]:
    """Read a JSONL file into a list."""
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def get_azure_client() -> OpenAI:
    """
    Create an Azure OpenAI client using API version.
    """
    ensure_azure_config()
    base_url = ENDPOINT.rstrip("/") + f"/openai/deployments/{DEPLOYMENT_NAME}"
    return OpenAI(
        api_key=API_KEY,
        base_url=base_url,
        default_query={"api-version": API_VERSION},
        default_headers={"api-key": API_KEY},
    )


def azure_generate(prompt: str, n: int = 2, temperature: float = 0.9, max_tokens: int = 256) -> List[str]:
    """Generate `n` completions using Azure OpenAI API."""
    client = get_azure_client()
    responses = []
    for _ in range(n):
        try:
            resp = client.chat.completions.create(
                model=DEPLOYMENT_NAME,
                messages=[
                    {"role": "system", "content": "You are a concise, helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=0.95,
            )
            text = resp.choices[0].message.content.strip()
            responses.append(text)
        except Exception as e:
            logger.error(f"Azure generation failed: {e}")
            responses.append(f"[Error generating response: {e}]")
    return responses


# Streamlit UI
st.set_page_config(page_title="RLHF PoC", page_icon="🤖", layout="wide")
st.title("🚀 RLHF POC")
st.caption("Lightweight RLHF workflow: Generate → Label → Train → Rerank.")

with st.sidebar:
    st.header("⚙️ Generation Settings")
    temperature = st.slider("Temperature", 0.1, 1.5, 0.9, 0.1)
    n_candidates = st.slider("Candidates per prompt", 2, 6, 3, 1)
    max_tokens = st.number_input("Max tokens per output", 64, 1024, 256, 64)

    if st.button("🔍 Test Azure Connection"):
        try:
            client = get_azure_client()
            resp = client.chat.completions.create(
                model=DEPLOYMENT_NAME,
                messages=[{"role": "user", "content": "Ping"}],
                max_tokens=5,
            )
            msg = resp.choices[0].message.content
            st.success(f"✅ Azure connection OK — Model responded: {msg}")
        except Exception as e:
            st.error(f"❌ Azure connection failed: {e}")

tabs = st.tabs(["💬 Generate", "✅ Label", "📈 Train Reward", "🏅 Rerank", "🗂 Dataset", "🎯 Complete Journey"])

# Generate Tab
with tabs[0]:
    st.subheader("Generate candidate responses")
    prompt = st.text_area("Prompt", "Explain Newton's First Law to a 10-year-old.", height=120)

    if st.button("Generate candidates"):
        try:
            with st.spinner("Generating responses..."):
                candidates = azure_generate(prompt, n=n_candidates, temperature=temperature, max_tokens=max_tokens)
            st.session_state["prompt"] = prompt
            st.session_state["candidates"] = candidates
            st.success(f"Generated {len(candidates)} candidates.")
        except Exception as e:
            st.error(str(e))

    if "candidates" in st.session_state:
        for i, text in enumerate(st.session_state["candidates"]):
            st.markdown(f"**Candidate {i}**")
            st.write(text)
            st.markdown("---")

# Label Tab
with tabs[1]:
    st.subheader("Human preference labeling")
    prompt = st.session_state.get("prompt")
    candidates = st.session_state.get("candidates")

    if not (prompt and candidates):
        st.warning("Generate candidates first.")
    else:
        pick = st.radio("Pick your preferred candidate:", [f"Candidate {i}" for i in range(len(candidates))])
        reason = st.text_input("Optional reason for choice:")

        if st.button("Save preference"):
            chosen_index = int(pick.split()[-1])
            entry = {
                "timestamp": time.time(),
                "prompt": prompt,
                "responses": [{"text": c} for c in candidates],
                "chosen_index": chosen_index,
                "reason": reason,
                "annotator": "user",
            }
            append_jsonl(PREFS_PATH, entry)
            st.success("✅ Preference saved!")

# Train Reward Model
with tabs[2]:
    st.subheader("Train TinyRewardModel")
    prefs = read_jsonl(PREFS_PATH)
    st.write(f"Loaded {len(prefs)} labeled examples.")

    if st.button("Train model"):
        if not prefs:
            st.error("No labeled data found. Label examples first.")
        else:
            with st.spinner("Training reward model..."):
                try:
                    rm = TinyRewardModel()
                    metrics = rm.fit_from_preferences(prefs)
                    rm.save(REWARD_DIR)
                    st.session_state["reward_model"] = rm
                    st.success(f"✅ Model trained (Accuracy: {metrics['test_accuracy']:.3f})")
                except Exception as e:
                    st.error(str(e))

    if REWARD_DIR.joinpath("reward_model.joblib").exists():
        if st.button("Load saved model"):
            try:
                rm = TinyRewardModel.load(REWARD_DIR)
                st.session_state["reward_model"] = rm
                st.success("✅ Reward model loaded.")
            except Exception as e:
                st.error(str(e))

# ----------------------------
# 🏅 Rerank Tab (ENHANCED: Saves complete journey)
# ----------------------------
with tabs[3]:
    st.subheader("Rerank new generations using reward model")
    new_prompt = st.text_area("Prompt to test", "Write a thank-you email after an interview.", height=120)
    k = st.slider("Candidates (k)", 2, 8, 4, 1)

    if st.button("Generate & Rerank"):
        rm = st.session_state.get("reward_model")
        if not rm:
            st.error("Train or load a reward model first.")
        else:
            try:
                with st.spinner("Generating and scoring..."):
                    cands = azure_generate(new_prompt, n=k, temperature=temperature, max_tokens=max_tokens)
                    scored: List[Tuple[str, float]] = [(c, rm.score(new_prompt, c)) for c in cands]
                    scored.sort(key=lambda x: x[1], reverse=True)
                    
                    best, best_score = scored[0]
                    
                    # NEW: Save complete journey entry
                    # Find if this prompt was labeled before
                    prefs = read_jsonl(PREFS_PATH)
                    selected_candidate = None
                    for pref in prefs:
                        if pref["prompt"] == new_prompt:
                            chosen_idx = pref.get("chosen_index")
                            if chosen_idx is not None and chosen_idx < len(pref["responses"]):
                                selected_candidate = pref["responses"][chosen_idx]["text"]
                            break
                    
                    journey_entry = {
                        "timestamp": time.time(),
                        "prompt": new_prompt,
                        "all_candidates": cands,
                        "selected_candidate": selected_candidate,  # From labeling phase
                        "top_ranked_output": best,
                        "top_ranked_score": best_score,
                        "all_scores": [{"text": txt, "score": sc} for txt, sc in scored]
                    }
                    append_jsonl(JOURNEY_PATH, journey_entry)
                    
                    st.success(f"🏅 Top candidate (score={best_score:.3f}):")
                    st.write(best)

                    with st.expander("View all candidates and scores"):
                        for i, (txt, sc) in enumerate(scored):
                            st.markdown(f"**#{i+1} — score={sc:.3f}**")
                            st.write(txt)
                            st.markdown("---")
            except Exception as e:
                st.error(str(e))

# ----------------------------
# 🗂 Dataset Tab
# ----------------------------
with tabs[4]:
    st.subheader("Preference dataset browser")
    prefs = read_jsonl(PREFS_PATH)

    if prefs:
        rows = [
            {
                "when": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(r.get("timestamp", 0))),
                "prompt": r["prompt"][:100],
                "num_responses": len(r["responses"]),
                "chosen_index": r.get("chosen_index"),
                "reason": r.get("reason", ""),
            }
            for r in prefs
        ]

        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, height=400)
        st.download_button("⬇️ Download JSONL", Path(PREFS_PATH).read_bytes(), "preferences.jsonl")
    else:
        st.info("No data yet. Label some examples first.")

# ----------------------------
# 🎯 NEW TAB: Complete Journey
# ----------------------------
with tabs[5]:
    st.subheader("🎯 Complete Journey: Before & After Training")
    st.caption(
        "Track the full RLHF workflow for each prompt: "
        "generated candidates → your selection → model's top choice after training."
    )
    
    journey_data = read_jsonl(JOURNEY_PATH)
    
    if not journey_data:
        st.info(
            "No journey data yet. Use the 🏅 Rerank tab to generate and score candidates. "
            "The complete workflow will be tracked here."
        )
    else:
        st.write(f"**Total journeys tracked:** {len(journey_data)}")
        
        # Display each journey as an expandable card
        for idx, journey in enumerate(reversed(journey_data), 1):
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(journey["timestamp"]))
            
            with st.expander(f"Journey #{len(journey_data) - idx + 1} — {timestamp} — {journey['prompt'][:60]}..."):
                # Prompt
                st.markdown("### 📝 Prompt")
                st.info(journey["prompt"])
                
                # All generated candidates
                st.markdown("### 🤖 All Generated Candidates")
                for i, cand in enumerate(journey["all_candidates"]):
                    st.markdown(f"**Candidate {i}:**")
                    st.write(cand)
                    st.markdown("---")
                
                # Selected candidate (human preference)
                st.markdown("### ✅ Your Selected Candidate (Human Preference)")
                if journey["selected_candidate"]:
                    st.success(journey["selected_candidate"])
                else:
                    st.warning("Not labeled yet. Go to ✅ Label tab to select your preference.")
                
                # Top ranked output after training
                st.markdown("### 🏆 Top Ranked Output (After Training)")
                st.markdown(f"**Score:** `{journey['top_ranked_score']:.3f}`")
                st.success(journey["top_ranked_output"])
                
                # Comparison insight
                if journey["selected_candidate"]:
                    if journey["selected_candidate"] == journey["top_ranked_output"]:
                        st.balloons()
                        st.success("🎉 **Perfect Match!** Model's top choice matches your preference!")
                    else:
                        st.info("💡 **Different Choice:** Model ranked a different candidate higher. This helps identify areas for improvement.")
                
                # All scores table
                st.markdown("### 📊 Complete Scoring Breakdown")
                scores_df = pd.DataFrame([
                    {
                        "Rank": i + 1,
                        "Score": round(item["score"], 3),
                        "Response": item["text"][:100] + "..." if len(item["text"]) > 100 else item["text"]
                    }
                    for i, item in enumerate(journey["all_scores"])
                ])
                st.dataframe(scores_df, use_container_width=True, hide_index=True)
        
        # Download option
        st.markdown("---")
        st.download_button(
            "⬇️ Download Complete Journey Data",
            Path(JOURNEY_PATH).read_bytes() if JOURNEY_PATH.exists() else b"",
            "complete_journey.jsonl",
        )