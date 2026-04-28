

"""
Clip insight generator - explains why a moment scored high and gives editing tips.
"""
import logging
from pathlib import Path
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

def generate_clip_insights(
        clip: dict,
        all_words: list[dict],
        chat_df: "pd.DataFrame | None" = None,
        config: dict | None = None,
) -> dict:
    """
    Build a human-readable summary for a detected clip.
    Returns a dict with keys like 'dominant_signal', 'viral_category',
    'keywords', 'chat_peak', 'visual_mood', 'editing_tips'.
    """
    start, end = clip["start"], clip["end"]
    window_words = [w for w in all_words if start <= w["start"] <= end or start <= w["end"] <= end]
    hype_words = config.get("detector", {}).get("hype_words", []) if config else []
    laugh_patterns = config.get("detector", {}).get("laughter_patterns", []) if config else []

    found_hype = set()
    found_laugh = set()
    full_text = " ".join(w["word"] for w in window_words)

    for w in window_words:
        word_lower = w["word"].lower()
        for hw in hype_words:
            if hw in word_lower:
                found_hype.add(hw)
        for lp in laugh_patterns:
            if lp in word_lower:
                found_laugh.add(lp)

    keywords = list(found_hype | found_laugh)

    # --- Chat insight ---
    chat_peak_msg = "No chat data"
    if chat_df is not None and not chat_df.empty:
        mask = (chat_df["time"] >= start) & (chat_df["time"] <= end)
        clip_chat = chat_df[mask]
        if not clip_chat.empty:
            peak_count = len(clip_chat)
            peak_msg = clip_chat["message"].value_counts().idxmax() if len(clip_chat) > 0 else ""
            chat_peak_msg = f"{peak_count} messages, top: \"{peak_msg}\""
        else:
            chat_peak_msg = "0 messages in this window"

    # --- Visual insight ---
    visual_score = clip.get("visual_score", 0)
    if visual_score > 0.7:
        visual_mood = "😆 Strong reaction (big smile/surprise)"
    elif visual_score > 0.4:
        visual_mood = "🙂 Mild reaction"
    elif visual_score > 0.01:
        visual_mood = " Neutral expression or no facecam"
    else:
        visual_mood = "Visual Indicator not active"

    # --- Dominant signal ---
    scores = {
        "loudness": clip.get("loudness_score", 0),
        "transcript": clip.get("transcript_score", 0),
        "chat": clip.get("chat_score", 0),
        "visual": visual_score,
    }
    try:
        dominant = max(scores, key=lambda k: scores[k])
    except (ValueError, KeyError):
        dominant = "unknown"

    # --- Viral category guess 
    category = "general highlight"
    if "lol" in full_text.lower() or found_laugh or visual_score > 0.6:
        category = "😂 funny reaction"
    elif "clutch" in full_text.lower() or "insane" in full_text.lower():
        category = "🎯 clutch play"
    elif "rage" in full_text.lower() or "wtf" in full_text.lower():
        category = "😡 rage / WTF moment"
    elif "pog" in full_text.lower() or scores["loudness"] > 2.0:
        category = "🔥 hype / pog moment"
    elif len(window_words) > 20 and any("?" in w["word"] for w in window_words):
        category = "💬 interesting discussion / Q&A"

    # --- Editing tips ---
    tips = []
    if scores["loudness"] > 1.5:
        tips.append("Keep game audio loud - energy is in the sound.")
    if scores["transcript"] > 1.0:
        tips.append("Add captions - highlight key phrases.")
    if visual_score > 0.5:
        tips.append("Facecam is key - consider zooming in on your/streamer's reaction.")
    if scores["chat"] > 1.0:
        tips.append("Overlay some funny chat messages on screen.")
    if (end - start) > 15:
        tips.append("Trim tighter - viral clips perform best at 5-15s.")
    if not tips:
        tips.append("This clip is ready to post as-is!")

    return {
        "dominant_signal": dominant,
        "viral_category": category,
        "keywords": keywords,
        "chat_peak": chat_peak_msg,
        "visual_mood": visual_mood,
        "editing_tips": tips,
        "full_text_sample": full_text[:150] + "..." if len(full_text) > 150 else full_text,
    }