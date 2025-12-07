import torch
from transformers import pipeline
import random

# -----------------------------
# Setup model
# -----------------------------
# Use CPU on Streamlit Cloud
device = -1

# Use your fine-tuned model if small enough; otherwise fallback to distilgpt2
try:
    model_path = "./AI_Content_Optimizer_Trained"
    generator = pipeline(
        "text-generation",
        model=model_path,
        tokenizer=model_path,
        device=device,
    )
except Exception as e:
    print("Failed to load local model. Using distilgpt2 fallback.")
    generator = pipeline(
        "text-generation",
        model="distilgpt2",
        tokenizer="distilgpt2",
        device=device,
    )

# -----------------------------
# Prompt Builder
# -----------------------------
def build_prompt(platform, topic, tone="professional", size="medium"):
    """
    Builds a dynamic prompt for social media content generation.
    """
    size_map = {
        "short": "1-2 sentences per idea",
        "medium": "3-4 sentences per idea",
        "long": "5-6 sentences with examples"
    }
    size_instruction = size_map.get(size.lower(), size_map["medium"])
    
    prompt = (
        f"Generate {size_instruction} social media content for {platform} "
        f"about '{topic}' in a {tone} tone. Format as a numbered list."
    )
    return prompt

# -----------------------------
# Content Generation
# -----------------------------
def generate_content(prompt, max_tokens=400, n_variations=3, temperature=0.7):
    """
    Generates multiple content variations using the model.
    Returns a list of generated texts.
    """
    outputs = generator(
        prompt,
        max_new_tokens=max_tokens,
        do_sample=True,
        temperature=temperature,
        num_return_sequences=n_variations,
        no_repeat_ngram_size=3,
        top_p=0.9,
    )
    
    return [out['generated_text'].strip() for out in outputs]

# -----------------------------
# Engagement Scoring (optional)
# -----------------------------
def mock_engagement_score(text):
    """
    Placeholder function to calculate a mock engagement score.
    Replace with real metrics: readability, sentiment, engagement, etc.
    """
    return round(random.uniform(60, 100), 2)

def select_top_content(contents):
    """
    Returns the content with the highest engagement score.
    """
    scored_contents = []
    for c in contents:
        score = mock_engagement_score(c)
        scored_contents.append({"content": c, "engagement_score": score})
    scored_contents.sort(key=lambda x: x["engagement_score"], reverse=True)
    return scored_contents[0], scored_contents

# -----------------------------
# Dynamic Function for Common Use
# -----------------------------
def generate_for_user(platform, topic, tone="professional", size="medium", n_variations=3):
    """
    Generates content variations and selects top recommendation for any user input.
    """
    prompt = build_prompt(platform, topic, tone, size)
    variations = generate_content(prompt, n_variations=n_variations)
    top_content, scored_contents = select_top_content(variations)
    return top_content, scored_contents

# -----------------------------
# Quick Test (dynamic example)
# -----------------------------
if __name__ == "__main__":
    # Example user input (can be replaced with Streamlit inputs)
    user_platform = input("Enter platform (Instagram, LinkedIn, Blog, etc.): ")
    user_topic = input("Enter topic/niche: ")
    user_tone = input("Enter tone (professional, friendly, etc.): ")
    user_size = input("Enter content size (short, medium, long): ")

    top_content, scored_contents = generate_for_user(
        user_platform, user_topic, user_tone, user_size, n_variations=3
    )

    print("\n🔹 Generated Variations:")
    for i, c in enumerate(scored_contents, 1):
        print(f"Variation {i}:\n{c['content']}\nEngagement Score: {c['engagement_score']}\n")

    print("🏆 Top Content Recommendation:")
    print(f"{top_content['content']}\nEngagement Score: {top_content['engagement_score']}")
