import streamlit as st
from model_generator import generate_for_user, mock_engagement_score
from textblob import TextBlob
import textstat

# Page Setup
st.set_page_config(page_title="AI Content Marketing Optimizer", page_icon="🤖", layout="wide")
st.title("🎯 AI Content Marketing Optimizer")
st.markdown("Generate, analyze, and optimize social media content using your AI model.")

# Inputs
platform = st.selectbox("Platform", ["Instagram", "YouTube", "TikTok", "Blog", "LinkedIn"])
topic = st.text_input("Topic / Niche", "AI in Healthcare")
tone = st.selectbox("Tone of the Content", ["Professional", "Friendly", "Witty"])
size = st.selectbox("Content Size", ["Short", "Medium", "Long"])
n_variations = st.slider("Number of Variations", 1, 5, 3)
generate_btn = st.button("Generate Content")

# Metrics
def sentiment_score(text):
    blob = TextBlob(text)
    return round((blob.sentiment.polarity + 1) / 2, 3)

def readability_score(text):
    try:
        return round(textstat.flesch_reading_ease(text), 2)
    except:
        return 0

if generate_btn:
    with st.spinner("⏳ Generating content..."):
        top_raw, variations_raw = generate_for_user(platform, topic, tone, size, n_variations)

        scored = []
        for v in variations_raw:
            score = {
                "content": v,
                "readability": readability_score(v),
                "sentiment_strength": sentiment_score(v),
                "engagement_score": mock_engagement_score(v)
            }
            scored.append(score)

        top = max(scored, key=lambda x: x["engagement_score"])

    st.markdown("### 🔹 Generated Content Variations")
    for i, s in enumerate(scored, 1):
        st.subheader(f"Variation {i}")
        st.write(s["content"])
        st.json({
            "readability": s["readability"],
            "sentiment_strength": s["sentiment_strength"],
            "engagement_score": s["engagement_score"]
        })

    st.markdown("### 🏆 Top Content")
    st.write(top["content"])
    st.json({
        "readability": top["readability"],
        "sentiment_strength": top["sentiment_strength"],
        "engagement_score": top["engagement_score"]
    })
