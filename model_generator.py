import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# -------------------------------------------------------------------
# Load Base Model + LoRA Adapter Correctly
# -------------------------------------------------------------------

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
LORA_PATH = "./AI_Content_Optimizer_Trained"

# Load tokenizer from the base model
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Load LoRA adapter
model = PeftModel.from_pretrained(base_model, LORA_PATH)

model.eval()

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
        f"You are a social media content expert. "
        f"Generate content for {platform} about '{topic}' with a {tone} tone. "
        f"{size_instruction} Format the response as a clean numbered list with no repetition."
    )

# -------------------------------------------------------------------
# Generate content
# -------------------------------------------------------------------

def generate_content(prompt, max_tokens=180, n_variations=3, temperature=0.7):
    results = []

    for _ in range(n_variations):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        output = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
            repetition_penalty=1.2,
        )

        text = tokenizer.decode(output[0], skip_special_tokens=True)
        results.append(text)

    return results

# -------------------------------------------------------------------
# Clean duplicates
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
# Main function called from Streamlit
# -------------------------------------------------------------------

def get_variations(platform, topic, tone="friendly", size="medium"):
    prompt = build_prompt(platform, topic, tone, size)
    raw_outputs = generate_content(prompt)
    return optimize_output(raw_outputs)
