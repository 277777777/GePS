# GePS: A Generalization-Potential Scoring Approach

This repository contains the implementation of **GePS**, a data-centric instruction selection method designed to improve the efficiency and quality of Large Language Model (LLM) instruction tuning.

By treating uncurated instruction datasets as data-level technical debt, GePS provides an intrinsic evaluation mechanism to identify high-potential samples without relying on external, closed-source teacher models.

## Key Features

*   **Intrinsic Evaluation**: Quantifies training utility via three intrinsic dimensions: **Information Entropy** (novelty), **Activation Intensity** (neuron coverage), and **Semantic Stability** (optimization robustness).
*   **Data Efficiency**: Reduces training computational overhead by **82.2%** compared to full-dataset tuning, achieving better performance with only 5k samples.
*   **Generalization-Driven**: Successfully preserves downstream software engineering reasoning capabilities (e.g., code generation on HumanEval) by mitigating the alignment tax.
*   **Dynamic Calibration**: Features a flexible routing threshold ($\tau$) to adjust the trade-off between optimization robustness and data scale[cite: 3].

## Performance Overview

| Model | Data Size | MT-Bench (Overall Score) | Pass@10 (HumanEval) |
| :--- | :--- | :--- | :--- |
| Uncurated Source | 52k | 5.75 | 47.56% |
| **GePS-5k (Ours)** | **5k** | **6.96** | **56.10%** |

*Performance validation on Llama-3-8B[cite: 3].*

## Getting Started

### Prerequisites
*   Python 3.8+
*   PyTorch 2.1.0+
*   CUDA 12.2+
*   Hardware: NVIDIA V100 GPU (recommended for performance analysis)

### Installation
1. Clone this repository:
   ```bash
   git clone [https://github.com/277777777/GePS](https://github.com/277777777/GePS)
   cd GePS
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
4. Running GePS
   To score and filter your own dataset, run the following command:
   ```bash
   python main.py --dataset_path ./data/alpaca_gpt4_data.json --output_path./data/selected_subset.json --top_k 5000
   ```bash
# Project Structure
  main.py: The entry point for executing the GePS selection pipeline[cite: 3].
  geps_core.py: Implementation of the three intrinsic metrics (Entropy, Activation, Stability)[cite: 3].
  utils/: Helper functions for perturbation generation and metric normalization[cite: 3].
  data/: Sample data and placeholders for your instruction datasets[cite: 3].
# Citation
If you find this work helpful for your research or model maintenance pipelines, please consider citing our work:

# License
This project is licensed under the MIT License.
