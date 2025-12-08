import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# -------------------------------------------------------------------
# Model paths
# -------------------------------------------------------------------
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
LORA_MODEL_PATH = "./AI_Content_Optimizer_Trained"

# -------------------------------------------------------------------
# Load LoRA-adapted model (CPU compatible)
# -------------------------------------------------------------------
def load_model():
    try:
        # Load tokenizer from base model
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            dtype=torch.float32,  # CPU-friendly
            device_map="auto",
            low_cpu_mem_usage=True
        )

        # Apply LoRA adapter
        model = PeftModel.from_pretrained(base_model, LORA_MODEL_PATH)
        model.eval()

        print(f"ℹ️ LoRA model loaded successfully from {LORA_MODEL_PATH}")
        return model, tokenizer

    except Exception as e:
        print(f"Error loading LoRA model: {str(e)}")
        return None, None

model, tokenizer = load_model()

# -------------------------------------------------------------------
# Build prompt
# -------------------------------------------------------------------
def build_prompt(platform, topic, tone="friendly", size="medium"):
    size_map = {
        "short": "Keep it concise, 1–2 sentences.",
        "medium": "Provide 3–4 sentences with moderate detail.",
        "long": "Provide 5–6 sentences with deeper insights."
    }
    size_instruction = size_map.get(size.lower(), size_map["medium"])
    return (
        f"Generate social media content for {platform} about '{topic}' with a {tone} tone. "
        f"{size_instruction} Format as a clean numbered list with no repetition."
    )

# -------------------------------------------------------------------
# Generate content
# -------------------------------------------------------------------
def generate_content(prompt, max_tokens=200, n_variations=3, temperature=0.7):
    if model is None or tokenizer is None:
        return ["Error: Model not loaded."]

    results = []
    try:
        for _ in range(n_variations):
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=0.9,
                    top_k=50,
                    repetition_penalty=1.2,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
                )
            text = tokenizer.decode(output[0], skip_special_tokens=True)
            if text.startswith(prompt):
                text = text[len(prompt):].strip()
            results.append(text)

            # Free up memory after each generation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    except Exception as e:
        results.append(f"Error: {str(e)}")

    return results

# -------------------------------------------------------------------
# Optimize / clean output
# -------------------------------------------------------------------
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

# -------------------------------------------------------------------
# Public function to get variations
# -------------------------------------------------------------------
def get_variations(platform, topic, tone="friendly", size="medium"):
    if model is None or tokenizer is None:
        return ["Model not available."]

    prompt = build_prompt(platform, topic, tone, size)
    raw_outputs = generate_content(prompt)
    return optimize_output(raw_outputs)
