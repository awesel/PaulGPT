#!/usr/bin/env python3
"""
Compares outputs from a base GGUF model and the same model with a LoRA adapter.
Loads two Llama instances and runs inference on both for each user prompt.
Streams tokens + final stats, running entirely on CPU.
Allows asking multiple questions in a loop.
"""

import sys
import time
import psutil
import math
try:
    from llama_cpp import Llama  # pip install llama-cpp-python>=0.2.24
except ImportError:
    print("Error: llama-cpp-python not found.")
    print("Please install it using: pip install llama-cpp-python")
    sys.exit(1)

# --- Configuration ---
# 1. Path to the base quantized GGUF model
# CHANGE IF YOUR PATH IS DIFFERENT
BASE_MODEL_GGUF_PATH = "/Users/andrew/PaulGPT/inference/models/gemma-3-quant/gemma-3-4b-it-q4_0.gguf"

# 2. Path to the GGUF LoRA adapter
# <-- CHANGE IF YOUR FILENAME IS DIFFERENT
LORA_ADAPTER_GGUF_PATH = "/Users/andrew/PaulGPT/inference/round_2/Round_2-F16-LoRA.gguf"

# 3. Inference Parameters
# Max new tokens to generate per model (adjust as needed)
MAX_TOK = 256
N_THREADS = None       # Use default (usually all cores) for CPU
N_CTX = 4096           # Context window size (must be same for both)
TEMPERATURE = 0.5       # Greedy (0.0 means deterministic) - Adjust if needed
PROMPT_TEMPLATE = "<start_of_turn>user\n{user_question}<end_of_turn>\n<start_of_turn>model\n"
EXIT_COMMANDS = ["quit", "exit", "q"]  # User can type these to exit

# --- Helper Function for Inference ---


def run_inference(model_instance, model_label, prompt, max_tokens, temp, stop_tokens):
    """Runs inference on a given Llama instance and prints output/stats."""
    print(f"\n─── GENERATION ({model_label}) ───")
    # print(f"Prompt: {prompt}") # Optional: print the full prompt
    print("Response:")

    start_time = None
    token_count = 0
    total_text = ""

    try:
        start_time = time.time()
        for token_data in model_instance(
            prompt,
            max_tokens=max_tokens,
            stop=stop_tokens,
            stream=True,
            temperature=temp,
            top_p=0.9,               # optional, but good for diversity
            top_k=50,                # optional, caps the token choices
            repeat_penalty=1.8,      # push down tokens you’ve already used
        ):
            # Check if generation stopped
            finish_reason = token_data["choices"][0].get("finish_reason")
            if finish_reason is not None:
                # print(f"\n(Finish Reason: {finish_reason})") # Optional debug info
                break  # Stop criteria reached

            token_text = token_data["choices"][0]["text"]
            total_text += token_text
            sys.stdout.write(token_text)
            sys.stdout.flush()
            token_count += 1

    except Exception as e:
        print(f"\nError during {model_label} generation: {e}")

    # Final newline after generation text
    print()

    # --- Statistics ---
    print(f"\n─── STATS ({model_label}) ───")
    if start_time is not None and token_count > 0:
        total_time = time.time() - start_time
        avg_tps = token_count / total_time if total_time > 0 else 0
        # Note: RAM usage reflects the total process RAM, not just this specific model.
        ram_usage_mb = psutil.Process().memory_info().rss / (1024**2)
        print(
            f"Generated {token_count} tokens in {total_time:.1f}s ({avg_tps:.2f} tok/s avg)")
        print(f"Total Process RAM usage: {ram_usage_mb:,.0f} MB")
    elif start_time is not None:
        total_time = time.time() - start_time
        print(f"No new tokens generated (ran for {total_time:.1f}s).")
    else:
        print(f"Generation loop did not start for {model_label}.")
    print("-" * 20)


# --- Model Loading ---
llm_base = None
llm_lora = None

# Load Base Model Instance
print(f"Loading base model: {BASE_MODEL_GGUF_PATH}")
try:
    llm_base = Llama(
        model_path=BASE_MODEL_GGUF_PATH,
        # No lora_path here
        n_threads=N_THREADS,
        n_ctx=N_CTX,
        use_mlock=True,  # Try to keep in RAM
        n_gpu_layers=0,  # CPU only
        logits_all=False,
        verbose=False,
        last_n_tokens_size=64,      # look back at the last 64 tokens
    )
    print("Base model loaded successfully.")
except Exception as e:
    print(f"\nError loading base model: {e}")
    print(
        f"Please check if the file exists and is valid: {BASE_MODEL_GGUF_PATH}")
    sys.exit(1)

# Load LoRA Model Instance
print(f"\nLoading base model WITH LoRA adapter: {LORA_ADAPTER_GGUF_PATH}")
try:
    llm_lora = Llama(
        model_path=BASE_MODEL_GGUF_PATH,  # Same base model
        lora_path=LORA_ADAPTER_GGUF_PATH,  # Apply the LoRA adapter
        n_threads=N_THREADS,
        n_ctx=N_CTX,
        use_mlock=True,  # Try to keep in RAM
        n_gpu_layers=0,  # CPU only
        logits_all=False,
        verbose=False,
        last_n_tokens_size=64,      # look back at the last 64 tokens
    )
    print("LoRA model loaded successfully.")
    print("\nNote: Loading two models will increase RAM usage significantly.")
except Exception as e:
    print(f"\nError loading model with LoRA: {e}")
    print("Please check:")
    print(f"1. Base model file: {BASE_MODEL_GGUF_PATH}")
    print(f"2. LoRA GGUF file: {LORA_ADAPTER_GGUF_PATH}")
    print("3. llama-cpp-python installation.")
    # Clean up base model if LoRA loading fails
    if llm_base:
        del llm_base
    sys.exit(1)


# --- Main Inference Loop ---
stop_tokens = ["<end_of_turn>", "<|endoftext|>",
               "<|im_end|>"]  # Define stop tokens

while True:
    try:
        user_input = input(
            f"\nEnter your question (or type '{'/'.join(EXIT_COMMANDS)}' to exit): ")
    except EOFError:  # Handle Ctrl+D
        print("\nExiting...")
        break

    if user_input.lower().strip() in EXIT_COMMANDS:
        print("Exiting...")
        break

    if not user_input.strip():
        print("Please enter a question.")
        continue

    # Construct the prompt for the current question
    current_prompt = PROMPT_TEMPLATE.format(user_question=user_input)

    print("\n" + "="*40)
    print(f"Processing Question: {user_input}")

    # Run inference on Base Model
    if llm_base:
        run_inference(
            model_instance=llm_base,
            model_label="Base Model",
            prompt=current_prompt,
            max_tokens=MAX_TOK,
            temp=TEMPERATURE,
            stop_tokens=stop_tokens,
        )
    else:
        print("\nBase model instance not available.")

    # Run inference on LoRA Model
    if llm_lora:
        run_inference(
            model_instance=llm_lora,
            model_label="LoRA Model",
            prompt=current_prompt,
            max_tokens=MAX_TOK,
            temp=TEMPERATURE,
            stop_tokens=stop_tokens,
        )
    else:
        print("\nLoRA model instance not available.")

    print("\n" + "="*40)
    print("Ready for next question.")
    print("="*40)


# --- End Main Inference Loop ---

# Cleanup (optional, helps release memory explicitly if needed)
if 'llm_base' in locals() and llm_base:
    del llm_base
if 'llm_lora' in locals() and llm_lora:
    del llm_lora
print("Script finished.")
