import torch
from transformers import pipeline

# Load fine-tuned QLoRA model
model_path = "./AI_Content_Optimizer_Trained"
device = 0 if torch.cuda.is_available() else -1

generator = pipeline(
    "text-generation",
    model=model_path,
    tokenizer=model_path,
    device=device,
)

# Build prompt
def build_prompt(platform, topic, tone="friendly", size="medium"):
    size_map = {
        "short": "Keep it concise, 1–2 sentences.",
        "medium": "Provide 3–4 sentences with moderate detail.",
        "long": "Provide 5–6 sentences with deeper insights."
    }
    size_instruction = size_map.get(size.lower(), size_map["medium"])
    return (f"You are a social media content expert. "
            f"Generate content for {platform} about '{topic}' with a {tone} tone. "
            f"{size_instruction} Format as a clean numbered list without repetition.")

# Generate variations
def generate_content(prompt, max_tokens=180, n_variations=3, temperature=0.7):
    outputs = generator(
        prompt,
        max_new_tokens=max_tokens,
        do_sample=True,
        temperature=temperature,
        num_return_sequences=n_variations,
        no_repeat_ngram_size=3
    )
    return [output["generated_text"] for output in outputs]

# Clean duplicates
def optimize_output(text_list):
    cleaned = []
    for text in text_list:
        lines = text.split("\n")
        unique = []
        seen = set()
        for line in lines:
            line = line.strip()
            if line and line not in seen:
                seen.add(line)
                unique.append(line)
        cleaned.append("\n".join(unique))
    return cleaned

def get_variations(platform, topic, tone="friendly", size="medium"):
    prompt = build_prompt(platform, topic, tone, size)
    raw_outputs = generate_content(prompt)
    return optimize_output(raw_outputs)
