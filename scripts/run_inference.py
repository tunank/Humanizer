import os
import sys

# --- Add src directory to Python path ---
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, "..", "src"))
if src_dir not in sys.path:
    sys.path.append(src_dir)

# --- Import project modules ---
from inference import (
    load_model_for_inference,
    generate_humanized_response,
    generate_base_model_response,
)

def main():
    # --- Configuration ---
    base_model_id = "mistralai/Mistral-7B-Instruct-v0.3"
    project_root = os.path.abspath(os.path.join(current_dir, ".."))
    adapter_base_name = "mistral-7b-humanizer-v1" # Should match the name used in training
    adapter_path = os.path.join(project_root, "results", "models", f"{adapter_base_name}-final")

    # Hugging Face Token (replace with your actual token or use env variable)
    hf_token = "YOUR TOKEN" # IMPORTANT: Replace or use environment variable

    # --- Test Example ---
    question_test = (
        'Why is every book I hear about a "NY Times #1 Best Seller"? ELI5: '
        'Why is every book I hear about a "NY Times #1 Best Seller"? '
        'Shouldn\'t there only be one "#1" best seller? Please explain like I\'m five.'
    )
    ai_response_test = (
        "There are many different best seller lists that are published by various "
        "organizations, and the New York Times is just one of them. The New York "
        "Times best seller list is a weekly list that ranks the best-selling books "
        "in the United States based on sales data from a number of different "
        "retailers. The list is published in the New York Times newspaper and is "
        "widely considered to be one of the most influential best seller lists in "
        "the book industry. \nIt's important to note that the New York Times best "
        "seller list is not the only best seller list out there, and there are many "
        "other lists that rank the top-selling books in different categories or in "
        "different countries. So it's possible that a book could be a best seller "
        "on one list but not on another. \nAdditionally, the term \"best seller\" "
        "is often used more broadly to refer to any book that is selling well, "
        "regardless of whether it is on a specific best seller list or not. So it's "
        "possible that you may hear about a book being a \"best seller\" even if it "
        "is not specifically ranked as a number one best seller on the New York "
        "Times list or any other list."
    )

    # --- Check if Adapter Exists ---
    if not os.path.exists(adapter_path):
        print(f"❌ Error: Adapter path not found: {adapter_path}")
        print("Please ensure the training script ran successfully and saved the adapter.")
        sys.exit(1)

    # --- 1. Load Fine-tuned Model ---
    model, tokenizer = load_model_for_inference(
        base_model_id=base_model_id,
        adapter_path=adapter_path,
        hf_token=hf_token
    )

    # --- 2. Generate Response with Fine-tuned Model ---
    humanized_response = generate_humanized_response(
        model=model,
        tokenizer=tokenizer,
        question=question_test,
        ai_response=ai_response_test,
        # Adjust generation parameters if needed
        max_new_tokens=300,
        temperature=0.6,
        top_p=0.85,
    )

    # --- 3. (Optional) Compare with Base Model ---
    run_base_comparison = True # Set to False to skip comparison
    if run_base_comparison:
        # Clear GPU memory before loading another large model if necessary
        del model
        del tokenizer
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("\n--- Cleared fine-tuned model from memory for base model comparison ---")

        base_response = generate_base_model_response(
            base_model_id=base_model_id,
            question=question_test,
            ai_response=ai_response_test,
            hf_token=hf_token,
            max_new_tokens=300,
            temperature=0.6,
            top_p=0.85,
        )

    print("\n✅ Inference testing complete.")

if __name__ == "__main__":
    main()
