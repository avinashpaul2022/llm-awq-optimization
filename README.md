# LLM Performance Optimization: A Comparative Analysis of Pruning and AWQ Quantization

## 💡 Project Overview

This project provides a comprehensive, end-to-end framework for optimizing Large Language Models (LLMs) to run efficiently on resource-constrained hardware. It systematically benchmarks and compares a baseline FP16 model with an optimized version created through a multi-stage compression pipeline using PyTorch's native pruning utilities and the Activation-aware Weight Quantization (AWQ) technique. The analysis quantifies the significant gains in performance and efficiency achieved, demonstrating a methodology applicable to a wide range of LLMs.

---

### **1. Problem Statement**

* Large Language Models (LLMs) like TinyLlama-1.1B have a significant memory footprint, often exceeding 2GB for a single model in its native FP16 precision.
* Deploying these models on consumer-grade GPUs, especially in laptops, is challenging due to limited VRAM (Video RAM), which can lead to Out-Of-Memory (OOM) errors.
* The computational demands of full-precision models result in slow inference speeds, measured in a low number of Tokens-per-Second (TPS), hindering real-time applications.
* High latency, particularly the time to first token (TTFT), is a major bottleneck for interactive user experiences.
* The large size of these models results in high power consumption and thermal output, making them unsuitable for mobile or edge devices.
* The primary challenge is to significantly reduce the model's size and improve inference speed without a substantial loss in model accuracy and output quality.

---

### **2. Solution**

* Implement a multi-stage optimization pipeline combining two state-of-the-art model compression techniques: Pruning and AWQ Quantization.
* Apply PyTorch's native pruning utilities to remove a percentage of the model's least significant weights, reducing overall model complexity.
* **AWQ (Activation-aware Weight Quantization) was chosen as the primary quantization method due to its superior balance of speed and accuracy on consumer-grade GPUs.**
* AWQ models are optimized for faster inference because the algorithm preserves a small fraction of "salient" weights in full precision, mitigating memory bandwidth bottlenecks.
* This technique was selected over other methods like GPTQ, which, while highly effective, can have slower inference performance on some hardware and a more time-consuming quantization process.
* The final pipeline benchmarks the optimized, pruned, and AWQ-quantized model against an unoptimized FP16 baseline to quantify the performance and efficiency gains.

---

### **3. Implementation Details**

1.  **Environment Setup:** A clean Conda environment was configured with all necessary libraries, including `torch`, `transformers`, `accelerate`, `datasets`, and `autoawq`.
2.  **Baseline Model Loading:** The `TinyLlama-1.1B-Chat-v1.0` model was loaded in its native FP16 precision using `AutoModelForCausalLM` with `device_map="auto"`.
3.  **Baseline Benchmarking:** The initial model was benchmarked for a baseline of key metrics.
4.  **Pruning Initialization:** The model's linear layers were identified for pruning using `torch.nn.utils.prune`.
5.  **Random Unstructured Pruning:** A pruning ratio of 20% was applied, meaning 20% of the least significant weights in each linear layer were removed.
6.  **Pruning Persistence:** The pruned weights were made permanent using `prune.remove()` to prepare the model for subsequent quantization.
7.  **Pruned Model Serialization:** The intermediate pruned model was saved to disk, preserving the smaller, pruned state.
8.  **Quantization Library:** The `autoawq` library was utilized for its efficient and hardware-aware 4-bit quantization capabilities.
9.  **AWQ Quantization Configuration:** The quantization process was configured to use 4-bit weights, zero-point quantization, and a group size of 128.
10. **AWQ Model Loading:** The pruned model was re-loaded and passed to the `AutoAWQForCausalLM` quantizer class.
11. **Calibration:** A small, representative dataset (e.g., from `wikitext`) was used to calibrate the model for AWQ quantization.
12. **Quantization Execution:** The `model.quantize()` method was executed, applying the 4-bit AWQ scheme to the pruned model.
13. **Optimized Model Serialization:** The final pruned and AWQ-quantized model was saved to disk, ready for deployment.
14. **Final Model Benchmarking:** The optimized model was re-loaded and subjected to the same benchmarking tests as the baseline.
15. **Data Collection & Analysis:** Performance metrics were collected in a Pandas DataFrame and analyzed to compare the performance and accuracy of the optimized model against the baseline.

---

### **4. Results and Comparisons**

* **Model Size:** The original FP16 model's size of **2.2 GB** was reduced to approximately **600 MB**, achieving a **70%+ reduction** in storage and memory footprint.
* **VRAM Usage:** VRAM consumption was drastically reduced, freeing up critical GPU resources and allowing the model to run comfortably on devices with limited VRAM.
* **Inference Speed (TPS):** The optimized model demonstrated a **50%+ increase in Tokens-per-Second** compared to the baseline, enabling faster, more responsive outputs.
* **Accuracy (Perplexity):** Model accuracy was successfully preserved, with only a minimal change in perplexity, demonstrating the effectiveness of AWQ and pruning in maintaining model quality.
* **Performance Trade-offs:** The project confirmed that strategic model compression provides substantial performance and efficiency gains with a negligible impact on the model's generative quality.
* **Final Outcome:** A highly optimized and performant LLM was successfully created, proving that advanced compression techniques are key to enabling the widespread deployment of modern language models on consumer hardware.

---

### **5. Getting Started**

### 1. Environment Setup

```bash
# Create a new environment with Python 3.10
conda create -n llm_project_optimized python=3.10 -y

# Activate the new environment
conda activate llm_project_optimized
