import os
import sys
from huggingface_hub import login

# --- Add src directory to Python path ---
# This allows importing modules from the 'src' directory
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, "..", "src"))
if src_dir not in sys.path:
    sys.path.append(src_dir)

# --- Import project modules ---
from data_processing import (
    load_hc3_dataset,
    filter_dataset_by_length,
    prepare_training_dataset,
)
from model_utils import (
    load_tokenizer,
    load_base_model,
    get_lora_config,
    apply_lora_to_model,
)
from training import get_training_args, train_model

def main():
    # --- Configuration ---
    # Model and Dataset
    model_id = "mistralai/Mistral-7B-Instruct-v0.3"
    dataset_name = "Hello-SimpleAI/HC3"
    dataset_subset = "reddit_eli5"
    # Set sample_size=None to use the full dataset, or a number for testing
    sample_size = None # e.g., 500 for a small test run

    # Output and Naming
    project_root = os.path.abspath(os.path.join(current_dir, ".."))
    new_model_base_name = "mistral-7b-humanizer-v1" # Base name for outputs
    output_dir = os.path.join(project_root, "results", "models", f"{new_model_base_name}-training_output")
    final_adapter_path = os.path.join(project_root, "results", "models", f"{new_model_base_name}-final")

    # Hugging Face Token (replace with your actual token or use env variable)
    hf_token = "YOUR TOKEN" # IMPORTANT: Replace or use environment variable

    # Data Filtering Parameters
    min_human_answer_length = 75
    iqr_multiplier = 1.5

    # LoRA Parameters
    lora_r = 32
    lora_alpha = 16
    lora_dropout = 0.1
    # target_modules = [...] # Default in model_utils is usually good for Mistral

    # Training Arguments (adjust as needed based on your hardware)
    num_train_epochs = 1
    per_device_train_batch_size = 1 # Keep low for large models
    gradient_accumulation_steps = 8 # Effective batch size = batch_size * grad_accum
    learning_rate = 2e-4
    save_steps = 200 # Save checkpoints more frequently if needed
    logging_steps = 50
    # bf16 = True # Enabled by default in get_training_args if available
    # gradient_checkpointing = True # Enabled by default

    # --- Hugging Face Login ---
    print("Logging into Hugging Face Hub...")
    try:
        login(token=hf_token)
        print("✓ Login successful.")
    except Exception as e:
        print(f"Could not log in to Hugging Face Hub: {e}")
        print("Please ensure your token is correct.")
        # Decide if you want to exit or continue without login
        # sys.exit(1) # Uncomment to exit if login fails

    # --- 1. Load and Prepare Dataset ---
    raw_dataset = load_hc3_dataset(dataset_name, dataset_subset, sample_size=sample_size)
    filtered_dataset = filter_dataset_by_length(
        raw_dataset,
        min_length=min_human_answer_length,
        iqr_multiplier=iqr_multiplier
    )
    if len(filtered_dataset) == 0:
        print("Dataset is empty after filtering. Cannot proceed with training.")
        sys.exit(1)

    formatted_dataset = prepare_training_dataset(filtered_dataset)
    if len(formatted_dataset) == 0:
        print("Dataset is empty after formatting. Cannot proceed with training.")
        sys.exit(1)

    # --- 2. Load Tokenizer and Model ---
    tokenizer = load_tokenizer(model_id, hf_token=hf_token)
    base_model = load_base_model(model_id, hf_token=hf_token)

    # --- 3. Configure and Apply PEFT (LoRA) ---
    lora_config = get_lora_config(
        r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout
    )
    model = apply_lora_to_model(base_model, lora_config)

    # --- 4. Configure Training Arguments ---
    training_args = get_training_args(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        save_steps=save_steps,
        logging_steps=logging_steps,
        # bf16=bf16, # Handled in get_training_args
        # gradient_checkpointing=gradient_checkpointing # Handled in get_training_args
    )

    # --- 5. Train the Model ---
    success = train_model(
        model=model,
        tokenizer=tokenizer,
        dataset=formatted_dataset,
        peft_config=lora_config,
        training_args=training_args,
        final_adapter_path=final_adapter_path
    )

    if success:
        print("\n🎉 Training and saving completed successfully!")
    else:
        print("\n❌ Training process encountered errors.")

if __name__ == "__main__":
    main()
