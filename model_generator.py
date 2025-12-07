import torch
from transformers import pipeline
import re
import random

device = -1  # CPU

# Try loading custom LoRA model
try:
    model_path = "./AI_Content_Optimizer_Trained"
    generator = pipeline(
        "text-generation",
        model=model_path,
        tokenizer=model_path,
        device=device
    )
    print("✅ Loaded trained model")
except:
    print("⚠️ Trained model not found. Using distilgpt2.")
    generator = pipeline(
        "text-generation",
        model="distilgpt2",
        tokenizer="distilgpt2",
        device=device
    )

# -------------------------------------------------------------------
# STRICT FORMATTER (MOST IMPORTANT PART)
# -------------------------------------------------------------------

def clean_output(text):
    """
    Removes repeated prompt, repeated instructions, junk, and extracts only numbered content.
    """
    text = text.replace("\n\n", "\n")
    lines = text.split("\n")

    cleaned = []
    for line in lines:
        line = line.strip()

        # Extract only lines starting with "1." "2." "3." etc.
        if re.match(r"^\d+\.", line):
            cleaned.append(line)

    return "\n".join(cleaned).strip()


# -------------------------------------------------------------------
# STRONG PROMPT BUILDER
# -------------------------------------------------------------------

def build_prompt(platform, topic, tone, size, n):
    size_map = {
        "Short": "1–2 sentences",
        "Medium": "3–4 sentences",
        "Long": "5–8 sentences"
    }

    return f"""
You are an AI social media content generator.

Generate EXACTLY {n} short content variations for {platform} about: "{topic}".

Tone: {tone}
Length: {size_map[size]}

Rules:
1. You MUST output only a numbered list.
2. Format strictly like:
   1. sentence
   2. sentence
   3. sentence
3. NO paragraphs, NO hashtags, NO emojis.
4. NO repeating instructions.
5. Only clean, final content.

Now generate the content:
""".strip()


# -------------------------------------------------------------------
# GENERATOR FUNCTION
# -------------------------------------------------------------------

def generate_content(prompt, n_sequences):
    raw_outputs = generator(
        prompt,
        max_new_tokens=180,
        temperature=0.8,
        top_p=0.9,
        num_return_sequences=n_sequences,
        do_sample=True
    )

    return [clean_output(o["generated_text"]) for o in raw_outputs]


# -------------------------------------------------------------------
# SCORING (FAKE ENGAGEMENT SCORE)
# -------------------------------------------------------------------

def mock_engagement_score(text):
    return round(random.uniform(65, 100), 2)


# -------------------------------------------------------------------
# MAIN FUNCTION CALLED FROM app.py
# -------------------------------------------------------------------

def generate_for_user(platform, topic, tone, size, n):
    prompt = build_prompt(platform, topic, tone, size, n)
    variations = generate_content(prompt, n)

    # If any variation fails to follow structure → try to fix
    fixed_variations = []
    for v in variations:
        if v.strip() == "":
            # fallback: create blank numbered structure
            v = "\n".join([f"{i+1}. ..." for i in range(n)])
        fixed_variations.append(v)

    return fixed_variations[0], fixed_variations
