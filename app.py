import streamlit as st
from model_generator import generate_for_user, mock_engagement_score
from textblob import TextBlob
import textstat

# -----------------------------
# Page Setup
# -----------------------------
st.set_page_config(
    page_title="AI Content Marketing Optimizer",
    page_icon="🤖",
    layout="wide"
)

st.title("🎯 AI Content Marketing Optimizer")
st.markdown(
    "Generate, analyze, and optimize social media content using your AI model."
)

# -----------------------------
# User Inputs
# -----------------------------
platform = st.selectbox("Platform", ["Instagram", "YouTube", "TikTok", "Blog", "LinkedIn"])
topic = st.text_input("Topic / Niche", "AI in Healthcare")
tone = st.selectbox("Tone of the Content", ["Professional", "Friendly", "Witty"])
size = st.selectbox("Content Size", ["Short", "Medium", "Long"])
n_variations = st.slider("Number of Variations", 1, 5, 3)

generate_btn = st.button("Generate Content")

# -----------------------------
# Helper Functions
# -----------------------------
def calculate_sentiment(text):
    """Returns sentiment polarity 0-1 (positive vs negative)."""
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity  # -1 to 1
    return round((polarity + 1) / 2, 3)  # convert to 0-1

def calculate_readability(text):
    """Returns Flesch Reading Ease score (0-100)."""
    try:
        return round(textstat.flesch_reading_ease(text), 2)
    except:
        return 0

def clean_generated_text(text):
    """
    Cleans the model output by removing instruction lines and keeping numbered content.
    """
    lines = text.split("\n")
    numbered_lines = [line for line in lines if line.strip() and line.strip()[0].isdigit()]
    if numbered_lines:
        return "\n".join(numbered_lines)
    else:
        return text.strip()

# -----------------------------
# Generate Content
# -----------------------------
if generate_btn:
    with st.spinner("⏳ Generating content variations..."):
        # Generate content
        top_content, variations = generate_for_user(platform, topic, tone, size, n_variations)

        # Clean and score each variation
        scored_contents = []
        for v in variations:
            clean_text = clean_generated_text(v['content'])
            readability = calculate_readability(clean_text)
            sentiment = calculate_sentiment(clean_text)
            engagement = mock_engagement_score(clean_text)  # replace with real metric if available
            scored_contents.append({
                "content": clean_text,
                "readability": readability,
                "sentiment_strength": sentiment,
                "engagement_score": engagement
            })

        # Identify top content
        top = max(scored_contents, key=lambda x: x["engagement_score"])

    # -----------------------------
    # Display Results
    # -----------------------------
    st.markdown("### 🔹 Generated Content Variations")
    for i, c in enumerate(scored_contents, 1):
        st.subheader(f"Variation {i}")
        st.write(c["content"])
        st.json({
            "readability": c["readability"],
            "sentiment_strength": c["sentiment_strength"],
            "engagement_score": c["engagement_score"]
        })

    st.markdown("### 🏆 Top Content Recommendation")
    st.write(top["content"])
    st.json({
        "readability": top["readability"],
        "sentiment_strength": top["sentiment_strength"],
        "engagement_score": top["engagement_score"]
    })
