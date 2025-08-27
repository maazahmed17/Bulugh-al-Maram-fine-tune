# ==============================================================================
# COMPARE ORIGINAL VS FINE-TUNED MODEL
# ==============================================================================
# This script helps you see the difference between the original Phi-2
# and your fine-tuned version on the same prompts
# ==============================================================================

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

def load_original_model():
    """Load the original Phi-2 without any fine-tuning"""
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
    
    return model, tokenizer

def load_fine_tuned_model():
    """Load your fine-tuned model with LoRA adapters"""
    # Same as previous script
    base_model, tokenizer = load_original_model()
    
    fine_tuned_model = PeftModel.from_pretrained(
        base_model,
        "../models/bulugh-al-maram-phi2-final",
        torch_dtype=torch.float16,
    )
    
    return fine_tuned_model, tokenizer

def compare_responses():
    """Compare original vs fine-tuned model responses"""
    print("Loading models...")
    original_model, tokenizer = load_original_model()
    fine_tuned_model, _ = load_fine_tuned_model()
    
    test_prompts = [
        "What is the Islamic ruling on",
        "In matters of prayer",
        "The Prophet (PBUH) said",
        # Add your specific domain prompts
    ]
    
    for prompt in test_prompts:
        print(f"\n{'='*60}")
        print(f"PROMPT: {prompt}")
        print('='*60)
        
        inputs = tokenizer(prompt, return_tensors="pt", padding=True)
        
        # Original model response
        with torch.no_grad():
            original_output = original_model.generate(
                **inputs, max_length=150, temperature=0.7, do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        original_response = tokenizer.decode(original_output[0], skip_special_tokens=True)
        
        # Fine-tuned model response  
        with torch.no_grad():
            fine_tuned_output = fine_tuned_model.generate(
                **inputs, max_length=150, temperature=0.7, do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        fine_tuned_response = tokenizer.decode(fine_tuned_output[0], skip_special_tokens=True)
        
        print(f"\n🔸 ORIGINAL MODEL:")
        print(original_response[len(prompt):].strip())
        
        print(f"\n🔹 FINE-TUNED MODEL:")
        print(fine_tuned_response[len(prompt):].strip())
        print()

if __name__ == "__main__":
    compare_responses()
