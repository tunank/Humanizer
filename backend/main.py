import os
import torch
import warnings
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
# Note: PEFT is NOT used in the new endpoint
from dotenv import load_dotenv
import gc # Garbage collector

warnings.filterwarnings("ignore")
load_dotenv()

# --- Configuration ---
# You might want to pass this via request or use env vars
# Using v0.3 here as an example, adjust if needed
DEFAULT_BASE_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
HF_TOKEN = os.getenv("HF_TOKEN")

# --- Pydantic Models ---
class HumanizeBaseRequest(BaseModel):
    question: str
    ai_response: str
    # Optional: Allow specifying model per request, otherwise use default
    model_id: str = DEFAULT_BASE_MODEL_ID

# --- FastAPI App Initialization ---
app = FastAPI(title="Humanizer AI API - Base Model Test")

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"], # Adjust as needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- API Endpoint for Base Model Humanization (Loads model per request) ---
@app.post("/humanize_base", summary="Humanize using Base Model (Loads per request - INEFFICIENT)")
async def humanize_base_model(request: HumanizeBaseRequest):
    """
    Loads the specified base model, generates a humanized response,
    and unloads the model. **WARNING: Very inefficient.**
    """
    model = None # Ensure model is None initially in this scope
    tokenizer = None # Ensure tokenizer is None initially

    print("\n--- Received Request for /humanize_base ---")
    print(f"Using base model: {request.model_id}")
    print(f"Question: {request.question[:100]}...")
    print(f"AI Response: {request.ai_response[:100]}...")

    # --- Start Base Model Test Logic (derived from your function) ---
    try:
        # 1. Load Tokenizer and Model (inside the request!)
        print(f"Loading base model: {request.model_id}...")
        # Use quantization for feasibility even in this inefficient setup
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=False,
        )
        tokenizer = AutoTokenizer.from_pretrained(request.model_id, token=HF_TOKEN)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token # Set pad token

        model = AutoModelForCausalLM.from_pretrained(
            request.model_id,
            quantization_config=bnb_config, # Use quantization
            torch_dtype=torch.bfloat16,
            device_map="auto", # Use GPU if available
            trust_remote_code=True,
            token=HF_TOKEN
        )
        model.eval() # Set to evaluation mode
        print("✓ Base model loaded for this request")

        # 2. Format Prompt using Chat Template
        # Using a slightly more direct system prompt for rewriting
        system_prompt = "You are an assistant that rewrites AI-generated text to sound more natural and conversational, like a human would write."
        user_prompt = f"Rewrite the following AI response to make it sound more human. The original question was: \"{request.question}\"\n\nAI Response to rewrite:\n\"\"\"\n{request.ai_response}\n\"\"\""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        # Apply chat template
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True # Adds the prompt structure for the assistant's turn
        )
        print("\n--- Input Prompt (Base Model) ---")
        print(prompt)

        # 3. Tokenize and Generate Response
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        print("\n--- Generating base model response... ---")
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask, # Pass attention mask
                max_new_tokens=256,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id, # Use EOS for padding during generation
                eos_token_id=tokenizer.eos_token_id
            )

        # 4. Decode Response
        prompt_length = inputs.input_ids.shape[1]
        response_ids = outputs[0][prompt_length:]
        base_response = tokenizer.decode(response_ids, skip_special_tokens=True)

        print("\n--- Base Model Generated Response ---")
        print(base_response)

        # Prepare the successful response data
        response_data = {"base_model_response": base_response.strip()}

    except Exception as e:
        print(f"!!! Error during base model processing: {e}")
        # Ensure cleanup happens even on error before raising HTTPException
        if model is not None:
            del model
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        print("Cleaned up resources after error.")
        raise HTTPException(status_code=500, detail=f"Error processing base model: {e}")

    finally:
        # 5. Cleanup: Crucial to free GPU memory after request is done
        if model is not None:
            del model
            print("Deleted base model from memory.")
        if tokenizer is not None:
            del tokenizer
            print("Deleted tokenizer from memory.")
        # Force garbage collection and clear CUDA cache
        gc.collect()
        torch.cuda.empty_cache()
        print("Forced garbage collection and cleared CUDA cache.")

    # Return the successful response
    return response_data