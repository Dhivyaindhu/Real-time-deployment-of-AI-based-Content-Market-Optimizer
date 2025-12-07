import torch
from transformers import pipeline
import random

device = -1  # CPU

try:
    model_path = "./AI_Content_Optimizer_Trained"
    generator = pipeline("text-generation", model=model_path, tokenizer=model_path, device=device)
except:
    generator = pipeline("text-generation", model="distilgpt2", tokenizer="distilgpt2", device=device)

def build_prompt(platform, topic, tone="professional", size="medium"):
    size_map = {"short": "1-2 sentences", "medium": "3-4 sentences", "long": "5-6 sentences with examples"}
    size_instruction = size_map.get(size.lower(), size_map["medium"])
    prompt = f"""
You are a social media content expert. Generate {size_instruction} social media content for {platform} about "{topic}" in a {tone} tone. Format as a numbered list.

Example:
1. AI-powered imaging tools can help doctors diagnose diseases faster and more accurately.
2. Machine learning algorithms analyze patient data to suggest personalized treatment plans.
3. Chatbots provide patients with instant support, improving engagement and satisfaction.

Now generate 3 variations for the topic: {topic}.
"""
    return prompt

def generate_content(prompt, max_tokens=400, n_variations=3, temperature=0.8):
    outputs = generator(prompt, max_new_tokens=max_tokens, do_sample=True, temperature=temperature,
                        num_return_sequences=n_variations, no_repeat_ngram_size=3, top_p=0.9)
    return [out['generated_text'].strip() for out in outputs]

def mock_engagement_score(text):
    return round(random.uniform(60, 100), 2)

def select_top_content(contents):
    scored = [{"content": c, "engagement_score": mock_engagement_score(c)} for c in contents]
    scored.sort(key=lambda x: x["engagement_score"], reverse=True)
    return scored[0], scored

def generate_for_user(platform, topic, tone="professional", size="medium", n_variations=3):
    prompt = build_prompt(platform, topic, tone, size)
    variations = generate_content(prompt, n_variations=n_variations)
    top, _ = select_top_content(variations)
    return top, variations
