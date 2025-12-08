import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import streamlit as st

# -------------------------------------------------------------------
# Configuration - Using base model only
# -------------------------------------------------------------------
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

# -------------------------------------------------------------------
# Load Base Model Only (No LoRA)
# -------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading AI model...")
def load_model():
    """Load base model without LoRA adapter"""
    try:
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        
        model.eval()
        
        st.info("ℹ️ Using base Qwen model (fine-tuned adapter not loaded)")
        return model, tokenizer
    
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
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
        f"You are a social media content expert. "
        f"Generate content for {platform} about '{topic}' with a {tone} tone. "
        f"{size_instruction} Format the response as a clean numbered list with no repetition."
    )

# -------------------------------------------------------------------
# Generate content
# -------------------------------------------------------------------
def generate_content(prompt, max_tokens=180, n_variations=3, temperature=0.7):
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
                    repetition_penalty=1.2,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
                )
            
            text = tokenizer.decode(output[0], skip_special_tokens=True)
            results.append(text)
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    except Exception as e:
        st.error(f"Generation error: {str(e)}")
        results.append(f"Error: {str(e)}")
    
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
# Main function
# -------------------------------------------------------------------
def get_variations(platform, topic, tone="friendly", size="medium"):
    if model is None or tokenizer is None:
        return ["Model not available."]
    
    prompt = build_prompt(platform, topic, tone, size)
    raw_outputs = generate_content(prompt)
    return optimize_output(raw_outputs)
