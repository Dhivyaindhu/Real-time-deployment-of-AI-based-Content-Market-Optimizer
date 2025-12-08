import gradio as gr
from model_generator import get_variations
from sentiment_analyzer import analyze_sentiment
from performance_metrics import performance_metrics, pick_best_version

# Function to generate content and metrics
def generate(platform, topic, tone, size):
    # Generate variations
    variations = get_variations(platform, topic, tone, size)

    results = []
    for i, text in enumerate(variations):
        # Sentiment & performance metrics
        sentiment, sentiment_score = analyze_sentiment(text)
        metrics = performance_metrics(text, sentiment_score)

        results.append({
            "Variation": f"Variation {i+1}",
            "Content": text,
            "Metrics": metrics
        })

    # Pick best content
    best_text, best_score = pick_best_version(variations)

    return results, best_text, best_score

# Define Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("## 🎯 AI Content Marketing Optimizer")
    with gr.Row():
        with gr.Column():
            platform_input = gr.Textbox(label="Platform (Instagram, YouTube, Blog, LinkedIn, etc.)")
            topic_input = gr.Textbox(label="Topic / Niche")
            tone_input = gr.Dropdown(["friendly", "professional", "witty", "emotional"], label="Tone")
            size_input = gr.Dropdown(["short", "medium", "long"], label="Content Size")
            generate_btn = gr.Button("🚀 Generate Optimized Content")
        with gr.Column():
            output_variations = gr.Dataframe(headers=["Variation", "Content", "Metrics"])
            best_content = gr.Textbox(label="🏆 Top Content Recommendation")
            best_score = gr.Textbox(label="📊 Engagement Score")

    generate_btn.click(
        fn=generate,
        inputs=[platform_input, topic_input, tone_input, size_input],
        outputs=[output_variations, best_content, best_score]
    )

# Launch Gradio app
if __name__ == "__main__":
    demo.launch()
