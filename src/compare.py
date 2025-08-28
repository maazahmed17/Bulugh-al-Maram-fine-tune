# ==============================================================================
# COMPARE ORIGINAL VS FINE-TUNED MODEL - FIXED VERSION
# ==============================================================================

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_original_model():
    """Load the original Phi-2 without any fine-tuning"""
    print("Loading original base model...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/phi-2",
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    print("✅ Original model loaded!")
    return model, tokenizer

def load_fine_tuned_model(base_model, tokenizer):
    """Load your fine-tuned model with LoRA adapters"""
    print("Loading fine-tuned adapters...")
    model = PeftModel.from_pretrained(
        base_model,
        "../models/bulugh-al-maram-phi2-final",
        torch_dtype=torch.float16,
        is_trainable=False,
    )
    
    print("✅ Fine-tuned model loaded!")
    return model, tokenizer

def generate_response(model, tokenizer, prompt, max_length=150, temperature=0.7):
    """Generate text with proper tokenization handling"""
    try:
        # FIXED: Add truncation and max_length to prevent malformed tokenization
        inputs = tokenizer(
            prompt, 
            return_tensors="pt", 
            padding=True,
            truncation=True,        # KEY FIX: Truncate long prompts
            max_length=512,         # KEY FIX: Limit input length
        )
        
        # Debug: Check input shapes
        print(f"Input IDs shape: {inputs['input_ids'].shape}")
        print(f"Input IDs device: {inputs['input_ids'].device}")
        
        # Move inputs to model device
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Generate with proper settings
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,     # CHANGED: Use max_new_tokens instead of max_length
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1,  # Prevent repetition
            )
        
        # Decode response
        response = tokenizer.decode(outputs, skip_special_tokens=True)
        return response[len(prompt):].strip()
        
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        return f"Error: {str(e)}"

def compare_responses():
    """Compare original vs fine-tuned model responses"""
    print("Loading models...")
    
    # Load models
    original_model, tokenizer = load_original_model()
    fine_tuned_model, tokenizer = load_fine_tuned_model(original_model, tokenizer)
    
    # FIXED: Use shorter, clearer prompt to avoid tokenization issues
    prompt = "What is the Islamic ruling on shaving the beard?"
    
    print(f"\n{'='*60}")
    print(f"PROMPT: {prompt}")
    print('='*60)
    
    # Generate responses
    print("\n🔸 ORIGINAL MODEL:")
    original_response = generate_response(original_model, tokenizer, prompt)
    print(original_response)
    
    print("\n🔹 FINE-TUNED MODEL:")
    fine_tuned_response = generate_response(fine_tuned_model, tokenizer, prompt)
    print(fine_tuned_response)
    
    # Compare results
    if original_response.strip() == fine_tuned_response.strip():
        print("\n⚠️ WARNING: Responses are identical - fine-tuning may not be influencing output.")
    else:
        print("\n✅ SUCCESS: Responses differ - fine-tuning is working!")

if __name__ == "__main__":
    compare_responses()
