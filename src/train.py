# ==============================================================================
# FINE-TUNING MICROSOFT PHI-2 WITH LORA ADAPTERS
# ==============================================================================
# This script fine-tunes the Microsoft Phi-2 model using:
# - 4-bit quantization (to fit in 8GB VRAM)
# - LoRA adapters (Parameter-Efficient Fine-Tuning)
# - TRL's SFTTrainer (Supervised Fine-Tuning Trainer)
#
# Hardware Requirements: RTX 4060 (8GB VRAM) or similar
# ==============================================================================

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,    # For loading language models
    AutoTokenizer,           # For loading tokenizers
    BitsAndBytesConfig,      # For 4-bit quantization configuration
)
from peft import LoraConfig          # For LoRA adapter configuration
from trl import SFTTrainer, SFTConfig # For supervised fine-tuning


# ==============================================================================
# DATA FORMATTING FUNCTION
# ==============================================================================
def format_instruction(example):
    """
    Extracts the 'text' field from each dataset example.
    
    This function tells the trainer which field in your JSONL data
    contains the training text. Your data/pairs.jsonl should have:
    {"text": "Your training example here"}
    {"text": "Another training example"}
    """
    return example['text']


# ==============================================================================
# MODEL CONFIGURATION
# ==============================================================================

# Model identifier from Hugging Face Hub
model_id = "microsoft/phi-2"  # 2.7B parameter model, perfect for RTX 4060

# 4-bit quantization configuration
# This reduces memory usage by ~75% (from ~11GB to ~3GB)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                    # Enable 4-bit quantization
    bnb_4bit_quant_type="nf4",            # Use NF4 (optimal for neural networks)
    bnb_4bit_compute_dtype=torch.bfloat16, # Use bfloat16 for computations
)


# ==============================================================================
# MODEL LOADING
# ==============================================================================

# Load the base model with quantization
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,    # Apply 4-bit quantization
    device_map="auto",                 # Automatically map to available GPU
    trust_remote_code=True,            # Required for Phi-2 model
)

# Disable caching during training (saves memory and is recommended for fine-tuning)
model.config.use_cache = False


# ==============================================================================
# TOKENIZER SETUP
# ==============================================================================

# Load the tokenizer that matches our model
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

# Set padding token to end-of-sequence token (required for batch processing)
tokenizer.pad_token = tokenizer.eos_token

# Pad sequences on the right side (standard for causal language modeling)
tokenizer.padding_side = "right"


# ==============================================================================
# LORA ADAPTER CONFIGURATION
# ==============================================================================

# LoRA (Low-Rank Adaptation) configuration
# Instead of training all 2.7B parameters, we only train small adapters (~0.28%)
lora_config = LoraConfig(
    r=64,                                    # Rank: controls adapter size (16 is good balance)
    lora_alpha=128,                           # Scaling factor (typically 2x the rank)
    target_modules=["q_proj", "k_proj", "v_proj","dense"], # Apply to attention layers only
    lora_dropout=0.05,                       # Dropout rate for adapters (prevents overfitting)
    bias="none",                             # Don't adapt bias parameters
    task_type="CAUSAL_LM",                   # Task type: Causal Language Modeling
)


# ==============================================================================
# DATASET LOADING
# ==============================================================================

# Load your custom dataset from JSONL file
# Make sure your data/pairs.jsonl exists and has {"text": "..."} format
dataset = load_dataset("json", data_files="data/pairs.jsonl", split="train")


# ==============================================================================
# TRAINING CONFIGURATION
# ==============================================================================

# Training configuration using SFTConfig (modern TRL API)
training_args = SFTConfig(
    # === Output and Logging ===
    output_dir="../models/bulugh-al-maram-phi2",  # Where to save checkpoints
    logging_steps=10,                             # Log training metrics every 10 steps
    
    # === Training Schedule ===
    num_train_epochs=5,                 # Train for 1 full epoch
    max_steps=100,                      # Limit to 100 steps for quick testing
    save_strategy="steps",              # Save checkpoints based on steps
    save_steps=50,                      # Save checkpoint every 50 steps
    
    # === Batch Size and Memory ===
    per_device_train_batch_size=1,      # Process 1 example at a time (fits in 8GB VRAM)
    gradient_accumulation_steps=4,      # Accumulate gradients over 4 steps (effective batch size = 4)
    dataloader_drop_last=True,          # Drop incomplete batches (avoids errors)
    
    # === Learning Rate ===
    learning_rate=3e-4,                 # Conservative learning rate (don't break pre-trained knowledge)
    
    warmup_ratio=0.03,  # Add warmup to avoid early instability
    lr_scheduler_type="cosine",

    # === Performance Optimizations ===
    fp16=True,                          # Use 16-bit floating point (saves memory)
    report_to=None,                     # Don't log to wandb/tensorboard
    
    # === SFT-Specific Parameters ===
    max_length=512,                     # Maximum sequence length (fits in VRAM)
    packing=False,                      # Don't pack multiple examples together
)


# ==============================================================================
# TRAINER SETUP
# ==============================================================================

# Create the SFTTrainer (Supervised Fine-Tuning Trainer)
trainer = SFTTrainer(
    model=model,                        # The quantized Phi-2 model
    train_dataset=dataset,              # Your custom dataset
    peft_config=lora_config,            # LoRA adapter configuration
    formatting_func=format_instruction, # Function to extract text from data
    processing_class=tokenizer,         # Tokenizer (modern parameter name)
    args=training_args,                 # Training configuration
)


# ==============================================================================
# TRAINING EXECUTION
# ==============================================================================

print("Starting training...")
print(f"Trainable parameters: {trainer.model.print_trainable_parameters()}")

# Start the fine-tuning process
# You'll see progress bars and loss values during training
trainer.train()

print("Training finished!")


# ==============================================================================
# MODEL SAVING
# ==============================================================================

# Save the trained LoRA adapters
trainer.save_model("../models/bulugh-al-maram-phi2-final")
print("Model saved successfully!")

# ==============================================================================
# WHAT HAPPENS DURING TRAINING:
# ==============================================================================
"""
1. MODEL LOADING (2-3 minutes):
   - Downloads and quantizes Phi-2 model
   - Applies LoRA adapters to attention layers
   - Shows "trainable params: 7,864,320 || all params: 2,787,548,160 || trainable%: 0.2821"

2. TRAINING LOOP (depends on dataset size):
   - Processes your data in batches of 1
   - Accumulates gradients over 4 steps
   - Updates only the LoRA adapter weights
   - Saves checkpoints every 50 steps
   - Shows progress bars and loss values

3. SAVING:
   - Saves final LoRA adapters to ../models/bulugh-al-maram-phi2-final/
   - You can later load these adapters with the base model for inference

MEMORY USAGE:
- Base model: ~3GB (thanks to 4-bit quantization)
- LoRA adapters: ~30MB
- Training overhead: ~2GB
- Total: ~5GB (well within your 8GB VRAM)

WHAT YOU'RE TRAINING:
- Only 0.28% of the model parameters (LoRA adapters)
- Focused on attention mechanisms (q_proj, k_proj, v_proj)
- Preserves pre-trained knowledge while adapting to your data
"""
