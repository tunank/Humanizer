import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import warnings
from .model_utils import get_bnb_config # Import from within the package

warnings.filterwarnings("ignore")

def load_model_for_inference(base_model_id, adapter_path, hf_token=None):
    """Loads the base model and applies the fine-tuned adapter for inference."""
    print(f"\n--- Loading Model for Inference ---")
    print(f"Base model: {base_model_id}")
    print(f"Adapter path: {adapter_path}")

    # 1. Load Tokenizer (from adapter path to ensure consistency)
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(adapter_path, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("✓ Tokenizer loaded.")

    # 2. Load Base Model with Quantization
    print("Loading base model with quantization...")
    bnb_config = get_bnb_config()
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        token=hf_token,
    )
    base_model.config.use_cache = True # Enable cache for inference
    print("✓ Base model loaded.")

    # 3. Apply LoRA Adapter
    print("Applying LoRA adapter...")
    model = PeftModel.from_pretrained(
        base_model,
        adapter_path,
        torch_dtype=torch.bfloat16,
        device_map="auto", # Ensure adapter is on the same device map
    )
    model.eval() # Set to evaluation mode
    print("✓ LoRA adapter applied and model set to eval mode.")

    return model, tokenizer


def generate_humanized_response(
    model,
    tokenizer,
    question,
    ai_response,
    max_new_tokens=256,
    num_beams=4,
    temperature=0.7,
    top_p=0.9,
    top_k=50,
    do_sample=True,
    no_repeat_ngram_size=3,
):
    """Generates a response using the fine-tuned model."""
    # Format the prompt exactly as used in training
    prompt = (
        f"<s>[INST] Rewrite the following AI response to the question so it sounds more "
        f"natural and human-like, while ensuring the key information and "
        f"level of detail are preserved.\n\n"
        f'Question: "{question}"\n\n'
        f'AI Response: "{ai_response}" [/INST] ' # Note the space after [/INST]
    )

    print("\n--- Input Prompt for Fine-tuned Model ---")
    print(prompt)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=False, # No padding needed for single inference
        truncation=True,
        max_length=1024, # Ensure prompt doesn't exceed max length
    ).to(model.device)

    print("\n--- Generating humanized response... ---")
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask, # Pass attention mask
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=do_sample,
            no_repeat_ngram_size=no_repeat_ngram_size,
            early_stopping=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens
    prompt_length = inputs.input_ids.shape[1]
    response_ids = outputs[0][prompt_length:]
    humanized_response = tokenizer.decode(response_ids, skip_special_tokens=True)

    print("\n--- Generated Humanized Response (Fine-tuned Model) ---")
    print(humanized_response)
    print(f"Word count: {len(humanized_response.split())}")

    return humanized_response


def generate_base_model_response(
    base_model_id,
    question,
    ai_response,
    hf_token=None,
    max_new_tokens=256,
    temperature=0.7,
    top_p=0.9,
):
    """Generates a response using the original base model for comparison."""
    print("\n\n--- COMPARING WITH BASE MODEL (NO FINE-TUNING) ---")

    # Load base model and tokenizer (without quantization for direct comparison if VRAM allows,
    # otherwise add quantization here too)
    print(f"Loading base model: {base_model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, token=hf_token)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16, # Use bfloat16
        device_map="auto",
        token=hf_token,
    )
    model.eval()
    print("✓ Base model loaded.")

    # Use the model's chat template if available and appropriate
    system_prompt = (
        "You are a helpful assistant that rewrites text to sound more natural, "
        "conversational, and human, while preserving key information."
    )
    user_prompt = (
        f'Please rewrite the following AI response to make it sound more human and conversational. '
        f'The response is answering this question: "{question}"\n\n'
        f'AI Response: "{ai_response}"'
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    try:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception: # Fallback if chat template fails
         print("Warning: Could not apply chat template. Using basic prompt.")
         prompt = f"{system_prompt}\n\nUSER: {user_prompt}\nASSISTANT:"


    print("\n--- Input Prompt for Base Model ---")
    print(prompt)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    print("\n--- Generating base model response... ---")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens
    prompt_length = inputs.input_ids.shape[1]
    response_ids = outputs[0][prompt_length:]
    base_response = tokenizer.decode(response_ids, skip_special_tokens=True)

    print("\n--- Base Model Response (No Fine-tuning) ---")
    print(base_response)
    print(f"Word count: {len(base_response.split())}")

    return base_response
