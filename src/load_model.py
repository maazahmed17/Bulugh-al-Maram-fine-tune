# In source/load_model.py

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

def main():
    """
    This script loads a pre-trained language model and its tokenizer
    with 4-bit quantization.
    """
    # 1. Define the model ID we want to use from Hugging Face
    model_id = "microsoft/phi-2"
    print(f"Loading model: {model_id}")

    # 2. Set up the Quantization Configuration
    # This is where we tell the model to load in 4-bit.
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4", # Use "nf4" for NormalFloat 4-bit
        bnb_4bit_compute_dtype=torch.bfloat16 # The compute dtype for better performance
    )
    
    # 3. Load the Model
    # We pass the quantization_config to the from_pretrained method.
    # device_map="auto" tells accelerate to figure out how to load the model layers
    # onto the GPU and CPU memory in the most optimal way.
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        quantization_config=bnb_config, 
        device_map="auto",
        trust_remote_code=True # Phi-2 requires this
    )
    print("Model loaded successfully in 4-bit!")

    # 4. Load the Tokenizer
    # The tokenizer translates text into numbers the model can understand.
    # It must match the model we are using.
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    # Phi-2 doesn't have a default padding token, so we set it to the end-of-sequence token.
    tokenizer.pad_token = tokenizer.eos_token
    print("Tokenizer loaded successfully!")


    # ---- NEW SECTION: LoRA CONFIGURATION ----
    print("\nApplying LoRA configuration...")
    lora_config = LoraConfig(
        r=16,  # Rank of the update matrices. Lower numbers = fewer parameters to train.
        lora_alpha=32, # A scaling factor for the LoRA weights.
        target_modules=["q_proj", "k_proj", "v_proj"], # Specific layers to apply LoRA to.
        lora_dropout=0.05, # Dropout probability to prevent overfitting.
        bias="none", # We are not training the bias terms.
        task_type="CAUSAL_LM" # Specifying the task type.
    )

    # Apply the LoRA config to our quantized model
    model = get_peft_model(model, lora_config)
    print("LoRA configuration applied successfully!")

    # Print the number of trainable parameters
    model.print_trainable_parameters()
    # ----------------------------------------
    
    # Print the memory footprint of the model
    print("\nModel Memory Footprint:")
    print(f"{model.get_memory_footprint() / 1e9:.2f} GB")


if __name__ == "__main__":
    main()