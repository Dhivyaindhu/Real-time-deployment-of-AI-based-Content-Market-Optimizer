import torch
from transformers import pipeline

model_path = "./AI_Content_Optimizer_Trained"
device = 0 if torch.cuda.is_available() else -1

generator = pipeline(
    "text-generation",
    model=model_path,
    tokenizer=model_path,
    device=device,
)

def build_prompt(platform, topic, tone="friendly", size="medium"):
    size_map = {
        "short": "Keep it concise, 1-2 sentences per idea.",
        "medium": "Provide moderate detail, 3-4 sentences per idea.",
        "long": "Provide detailed content with examples, 5-6 sentences per idea."
    }
    size_instruction = size_map.get(size.lower(), size_map["medium"])

    prompt = (
        f"You are a social media content expert. "
        f"Generate content for {platform} about '{topic}' with a {tone} tone. "
        f"{size_instruction} "
        f"Format the output as a numbered list without repetition."
    )
    return prompt

def generate_content(prompt, max_tokens=150, n_variations=3, temperature=0.7):
    outputs = generator(
        prompt,
        max_new_tokens=max_tokens,
        do_sample=True,
        temperature=temperature,
        num_return_sequences=n_variations,
        no_repeat_ngram_size=3
    )
    return [out['generated_text'] for out in outputs]
