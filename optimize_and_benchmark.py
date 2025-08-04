# --- IMPORTS ---
import torch
import os
import time
import pandas as pd
from typing import Dict
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from datasets import load_dataset
from accelerate import Accelerator
from awq import AutoAWQForCausalLM
from torch.nn.utils import prune

# --- CONFIGURATION ---
MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
OUTPUT_DIR = "optimized_models"
os.makedirs(OUTPUT_DIR, exist_ok=True)
DEVICE = "cuda"

# Check for CUDA availability
if not torch.cuda.is_available():
    raise SystemError("CUDA is not available. Please ensure you have a GPU enabled and correctly configured.")

# --- HELPER FUNCTIONS ---

def benchmark_model(model, tokenizer, prompt="What is the capital of France?", max_new_tokens=100):
    """
    Benchmarks a model for inference speed (TPS) and VRAM usage.
    """
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    model.eval()
    model.to(DEVICE)
    
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(DEVICE)

    start_event.record()
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, generation_config=GenerationConfig(do_sample=True, top_p=0.95))
    end_event.record()
    torch.cuda.synchronize(DEVICE)

    ttft = start_event.elapsed_time(end_event) / 1000.0
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated_tokens = outputs[0].size(0) - inputs.input_ids.size(1)
    tps = generated_tokens / ttft
    vram_usage = torch.cuda.max_memory_allocated(DEVICE) / (1024**3) # in GB

    return {
        "TTFT (s)": ttft,
        "TPS": tps,
        "VRAM (GB)": vram_usage,
        "Generated Text": generated_text
    }

def calculate_perplexity(model, tokenizer, dataset, device="cuda"):
    """
    Calculates the perplexity of a model on a given dataset.
    """
    model.eval()
    encodings = tokenizer("\n\n".join(dataset['text']), return_tensors='pt')
    max_length = 512
    stride = 512
    seq_len = encodings.input_ids.size(1)
    
    nlls = []
    
    for i in tqdm(range(0, seq_len, stride)):
        begin_loc = max(i + stride - max_length, i)
        end_loc = i + stride
        if end_loc > seq_len:
            end_loc = seq_len
        trg_len = end_loc - begin_loc

        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100
        
        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss
        
        nlls.append(neg_log_likelihood)
    
    ppl = torch.exp(torch.stack(nlls).mean())
    return ppl.item()

# Load perplexity dataset
print("Loading perplexity dataset...")
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
dataset = dataset.filter(lambda example: len(example['text']) > 10)

# Initialize results table
results = pd.DataFrame()

# --- Phase 1: Benchmarking Baseline FP16 Model ---
print("\n--- Phase 1: Benchmarking Baseline FP16 Model ---")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
# Load with offloading for a memory-safe baseline
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="auto", torch_dtype=torch.float16)

baseline_metrics = {
    "Size (MB)": os.path.getsize(f"{os.path.expanduser('~')}/.cache/huggingface/hub/models--{MODEL_ID.replace('/', '--')}/snapshots/main/model.safetensors") / (1024**2),
    "Perplexity": calculate_perplexity(model, tokenizer, dataset, device=DEVICE)
}
# TTFT, TPS, VRAM will be measured on the final, small models for a fair comparison
baseline_metrics["TTFT (s)"] = "N/A"
baseline_metrics["TPS"] = "N/A"
baseline_metrics["VRAM (GB)"] = "N/A"
baseline_metrics["Generated Text"] = "N/A"

results.loc["Baseline (FP16)"] = baseline_metrics
print("Baseline Metrics:\n", results)
del model
torch.cuda.empty_cache()


# --- Phase 2: The Multi-Step, PyTorch-Based Pipeline (Pruning + AWQ) ---
print("\n--- Phase 2: Optimizing with Pruning + AWQ ---")
# Reload the FP16 model for optimization
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16, device_map="auto")

# 2.1 Pruning Step
print("Applying 20% random unstructured pruning...")
linear_layers = {name: module for name, module in model.named_modules() if isinstance(module, torch.nn.Linear)}

for name, module in linear_layers.items():
    prune.random_unstructured(module, amount=0.2)
    # Make pruning permanent for serialization
    prune.remove(module, "weight")
    
pruned_model_path = os.path.join(OUTPUT_DIR, "pruned_model")
model.save_pretrained(pruned_model_path)
print(f"Pruned model saved to {pruned_model_path}")
del model
torch.cuda.empty_cache()


# 2.2 AWQ Quantization Step
print("Applying 4-bit AWQ quantization...")
model = AutoAWQForCausalLM.from_pretrained(pruned_model_path, **{"low_cpu_mem_usage": True, "torch_dtype": torch.float16})
tokenizer = AutoTokenizer.from_pretrained(pruned_model_path, trust_remote_code=True)
quant_config = { "zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM" }
model.quantize(tokenizer, quant_config=quant_config)
awq_model_path = os.path.join(OUTPUT_DIR, "awq_optimized_model")
model.save_quantized(awq_model_path)
tokenizer.save_pretrained(awq_model_path)
print(f"Quantized model saved to {awq_model_path}")


# --- Phase 3: Final Benchmarking and Results ---
print("\n--- Phase 3: Benchmarking Final Optimized Model ---")
# Load the final optimized model
optimized_model = AutoAWQForCausalLM.from_quantized(
    awq_model_path,
    fuse_layers=True,
    trust_remote_code=False,
    safetensors=True
)
tokenizer = AutoTokenizer.from_pretrained(awq_model_path, trust_remote_code=True)

# Run benchmarks
optimized_metrics = benchmark_model(optimized_model, tokenizer)
optimized_metrics["Size (MB)"] = os.path.getsize(os.path.join(awq_model_path, "awq_model.safetensors")) / (1024**2)
optimized_metrics["Perplexity"] = calculate_perplexity(optimized_model, tokenizer, dataset, device=DEVICE)

results.loc["Optimized (Pruned+AWQ)"] = optimized_metrics

print("\n--- Final Results ---")
print(results)
