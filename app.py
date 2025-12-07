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

def calculate_sentiment(text):
    blob = TextBlob(text)
    return round((blob.sentiment.polarity + 1) / 2, 3)

def calculate_readability(text):
    try: 
        return round(textstat.flesch_reading_ease(text), 2)
    except: 
        return 0

def clean_generated_text(text):
    lines = text.split("\n")
    numbered = [line for line in lines if line.strip() and line.strip()[0].isdigit()]
    return "\n".join(numbered) if numbered else text.strip()

if generate_btn:
    with st.spinner("⏳ Generating dynamic content..."):
        top_content, variations = generate_for_user(platform, topic, tone, size, n_variations)

        scored_contents = []
        for v in variations:
            clean_text = clean_generated_text(v)
            readability = calculate_readability(clean_text)
            sentiment = calculate_sentiment(clean_text)
            engagement = mock_engagement_score(clean_text)

            scored_contents.append({
                "content": clean_text,
                "readability": readability,
                "sentiment_strength": sentiment,
                "engagement_score": engagement
            })

        top = max(scored_contents, key=lambda x: x["engagement_score"])

    st.markdown("### 🔹 Generated Content Variations")
    for idx, c in enumerate(scored_contents, 1):
        st.subheader(f"Variation {idx}")
        st.write(c["content"])
        st.json({
            "readability": c["readability"],
            "sentiment_strength": c["sentiment_strength"],
            "engagement_score": c["engagement_score"]
        })

    st.markdown("### 🏆 Top Content")
    st.write(top["content"])
    st.json({
        "readability": top["readability"],
        "sentiment_strength": top["sentiment_strength"],
        "engagement_score": top["engagement_score"]
    })
