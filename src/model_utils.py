import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

def load_tokenizer(model_id, hf_token=None):
    """Loads the tokenizer for the specified model."""
    print(f"Loading tokenizer for: {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=True, token=hf_token
    )
    tokenizer.padding_side = "right"  # Important for Causal LM
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token # Set pad token if missing
    print("✓ Tokenizer loaded.")
    return tokenizer

def get_bnb_config():
    """Returns the BitsAndBytes configuration for 4-bit quantization."""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=False,
    )

def load_base_model(model_id, hf_token=None):
    """Loads the base model with 4-bit quantization."""
    print(f"Loading base model: {model_id}...")
    bnb_config = get_bnb_config()
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto", # Automatically distribute across GPUs
        trust_remote_code=True,
        token=hf_token,
    )
    model.config.use_cache = False  # Disable cache for training
    model.config.pretraining_tp = 1 # Compatibility setting if needed
    print("✓ Base model loaded.")
    return model

def get_lora_config(
    r=32,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
):
    """Returns the LoRA configuration."""
    print("Configuring PEFT (LoRA)...")
    return LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )

def apply_lora_to_model(model, lora_config):
    """Prepares model for k-bit training and applies LoRA adapter."""
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    print("✓ LoRA adapter applied.")
    model.print_trainable_parameters()
    return model
