import torch
from transformers import pipeline
import random

device = -1  # CPU

# Load custom model if available, else fallback
try:
    model_path = "./AI_Content_Optimizer_Trained"
    generator = pipeline(
        "text-generation",
        model=model_path,
        tokenizer=model_path,
        device=device
    )
except:
    print("⚠️ Custom model not found. Using distilgpt2 fallback.")
    generator = pipeline(
        "text-generation",
        model="distilgpt2",
        tokenizer="distilgpt2",
        device=device
    )

def build_prompt(platform, topic, tone="professional", size="medium", n_variations=3):
    size_map = {
        "short": "1–2 sentences",
        "medium": "3–4 sentences",
        "long": "5–8 sentences"
    }
    size_instruction = size_map.get(size.lower(), size_map["medium"])

    prompt = f"""
Generate {n_variations} unique content variations for {platform}.
Topic: "{topic}"
Tone: {tone}
Length: {size_instruction}

Each variation must:
- Follow the tone style
- Match the platform audience
- Be formatted as a numbered list (1., 2., 3., ...)

Do NOT include examples. 
Do NOT repeat the instructions. 
Start generating now.
"""
    return prompt.strip()

def generate_content(prompt, max_tokens=350, n_variations=3, temperature=0.85):
    outputs = generator(
        prompt,
        max_new_tokens=max_tokens,
        temperature=temperature,
        do_sample=True,
        top_p=0.92,
        num_return_sequences=n_variations
    )
    return [out["generated_text"].strip() for out in outputs]

def mock_engagement_score(text):
    return round(random.uniform(60, 100), 2)

def select_top_content(contents):
    scored = [{"content": c, "engagement_score": mock_engagement_score(c)} for c in contents]
    return max(scored, key=lambda x: x["engagement_score"]), scored

def generate_for_user(platform, topic, tone="professional", size="medium", n_variations=3):
    prompt = build_prompt(platform, topic, tone, size, n_variations)
    variations = generate_content(prompt, n_variations=n_variations)
    top, _ = select_top_content(variations)
    return top, variations
