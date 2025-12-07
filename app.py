import streamlit as st
from model_generator import generate_content, build_prompt
from sentiment_analyzer import analyze_sentiment
from performance_metrics import performance_metrics, pick_best_version

st.set_page_config(page_title="AI Content Marketing Optimizer", layout="wide")

st.title("🎯 AI Content Marketing Optimizer")
st.write("Generate, analyze, and optimize content using your fine-tuned AI model.")

st.markdown("---")

# -----------------------------
# Inputs
# -----------------------------
platform = st.text_input("📝 Platform (Instagram, YouTube, Blog, LinkedIn, etc.)")
topic = st.text_input("🎯 Topic / Niche")
tone = st.selectbox("🎭 Tone of the Content", ["friendly", "professional", "witty", "emotional"])
size = st.selectbox("📏 Content Size", ["short", "medium", "long"])

generate_btn = st.button("🚀 Generate Optimized Content")

# -----------------------------
# Run the Model
# -----------------------------
if generate_btn:

    if not platform or not topic:
        st.warning("⚠️ Please enter both Platform and Topic before generating.")
        st.stop()

    prompt = build_prompt(platform, topic, tone, size)

    st.info("⏳ Generating content variations... please wait.")
    variations = generate_content(prompt)

    st.markdown("---")
    st.subheader("✨ Generated Content Variations")

    all_metrics = []

    for i, text in enumerate(variations):
        st.write(f"### 🔹 Variation {i+1}")
        st.write(text)

        sentiment, sentiment_score = analyze_sentiment(text)

        metrics = performance_metrics(text, sentiment_score)
        all_metrics.append(metrics)

        st.json(metrics)
        st.markdown("---")

    # -----------------------------
    # Best Output Selection
    # -----------------------------
    best_text, best_score = pick_best_version(variations)

    st.success("🏆 **Top Content Recommendation (Best Engagement Score)**")
    st.write(best_text)
    st.write(f"📊 **Engagement Score:** {best_score}")
