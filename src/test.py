# ==============================================================================
# TEST YOUR FINE-TUNED MODEL
# ==============================================================================
# This script loads your trained LoRA adapters and tests the model's performance
# to see if fine-tuning improved responses on your specific domain
# ==============================================================================

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import PeftModel
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_fine_tuned_model():
    """
    Loads the base Phi-2 model and applies your trained LoRA adapters
    """
    print("Loading base model...")
    
    # Same quantization config as training
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        "microsoft/phi-2",
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # Load your fine-tuned LoRA adapters
    print("Loading your trained LoRA adapters...")
    model = PeftModel.from_pretrained(
        base_model,
        "../models/bulugh-al-maram-phi2-final",  # Your saved adapters
        torch_dtype=torch.float16,
        is_trainable=False,  # Inference mode - reduces warnings
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    print("✅ Fine-tuned model loaded successfully!")
    return model, tokenizer

def generate_response(model, tokenizer, prompt, max_length=200):
    """
    Generate text using your fine-tuned model
    """
    try:
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt", padding=True)
        
        # Move inputs to the same device as the model (fixes device mismatch)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.7,          # Controls creativity (0.1 = conservative, 1.0 = creative)
                do_sample=True,           # Enable sampling for variety
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        # Decode and return
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response
        
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        return f"Error: {str(e)}"

def test_model():
    """
    Test your fine-tuned model with sample prompts
    """
    # Load your fine-tuned model
    model, tokenizer = load_fine_tuned_model()
    
    # Test prompts (customized for your Islamic jurisprudence domain)
    test_prompts = [
        "What is the ruling on shaving the beard,What is the punishment for the one who shaves the beard, in what circumstances is it allowed to shave the beard?", 
        
        # Add more relevant to your training data
    ]
    
    print("\n" + "="*50)
    print("TESTING YOUR FINE-TUNED MODEL")
    print("="*50)
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n--- Test {i} ---")
        print(f"Prompt: {prompt}")
        print("Response:")
        
        response = generate_response(model, tokenizer, prompt)
        # Remove the prompt from response to show only generated text
        generated_text = response[len(prompt):].strip()
        print(f"Generated: {generated_text}")
        print("-" * 30)

if __name__ == "__main__":
    test_model()
