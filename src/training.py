import os
import torch
import traceback
from transformers import TrainingArguments
from trl import SFTTrainer

def get_training_args(
    output_dir,
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    optim="adamw_torch",
    save_steps=500,
    logging_steps=100,
    learning_rate=2e-4,
    weight_decay=0.001,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    report_to="tensorboard",
    bf16=True, # Enable bfloat16 training
    gradient_checkpointing=True,
):
    """Configures and returns TrainingArguments."""
    print("Configuring Training Arguments...")
    # Ensure bf16 is available if requested
    use_bf16 = bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    if bf16 and not use_bf16:
        print("Warning: bf16 requested but not available. Disabling.")

    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        optim=optim,
        save_strategy="steps", # Save based on steps
        save_steps=save_steps,
        logging_steps=logging_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        fp16=False, # Disable fp16 if using bf16
        bf16=use_bf16, # Use bf16 if available
        max_grad_norm=max_grad_norm,
        max_steps=-1, # Train for num_train_epochs
        warmup_ratio=warmup_ratio,
        group_by_length=True, # Efficiency improvement
        lr_scheduler_type=lr_scheduler_type,
        report_to=report_to,
        gradient_checkpointing=gradient_checkpointing,
        # evaluation_strategy="steps", # Add if you have an eval set
        # eval_steps=200,             # Add if you have an eval set
        save_total_limit=2, # Keep only the last 2 checkpoints
        load_best_model_at_end=False, # Set to True if using evaluation
    )

def train_model(model, tokenizer, dataset, peft_config, training_args, final_adapter_path):
    """Initializes SFTTrainer and runs the training loop."""
    print("\nInitializing Trainer...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,
        dataset_text_field="text", # Specify the column containing formatted text
        max_seq_length=1024,      # Adjust based on your data/model
        packing=False,            # Set packing=True if desired (can improve efficiency)
    )

    print("Starting training...")
    try:
        trainer.train()
        print("✓ Training finished.")
    except Exception as e:
        print(f"💥 Training error: {e}")
        traceback.print_exc()
        return False # Indicate failure

    # --- Save the final adapter ---
    print(f"\nSaving final LoRA adapter weights to {final_adapter_path}")
    try:
        # Ensure the target directory exists
        os.makedirs(os.path.dirname(final_adapter_path), exist_ok=True)
        trainer.model.save_pretrained(final_adapter_path) # Saves only the adapter
        tokenizer.save_pretrained(final_adapter_path) # Save tokenizer with adapter
        print("✓ Adapter and tokenizer saved.")
        return True # Indicate success
    except Exception as e:
        print(f"💥 Error saving adapter: {e}")
        traceback.print_exc()
        return False # Indicate failure

