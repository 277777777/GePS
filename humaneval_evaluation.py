import argparse
import torch
import os
import json
import math
import concurrent.futures
import multiprocessing as mp
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# 导入官方评估工具
from human_eval.data import read_problems
from human_eval.evaluation import evaluate_functional_correctness

def setup_model(model_path, lora_path=None):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base_model, lora_path) if lora_path else base_model
    model.eval()
    return model, tokenizer

def batch_generate_code(model, tokenizer, problems_chunk, model_type, batch_size, gpu_id):
    results = []
    # 每次处理一个 batch
    for i in range(0, len(problems_chunk), batch_size):
        batch = problems_chunk[i:i+batch_size]
        prompts = []
        
        for p in batch:
            # 针对代码补全任务，Instruct 模型的微调模版需要精准包裹
            if model_type == "llama3":
                prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are an expert Python programmer. Complete the following Python function exactly as requested without explanation.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{p['prompt']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{p['prompt']}"
            else:
                prompt = f"<s>[INST] <<SYS>>\nYou are an expert Python programmer. Complete the following Python function exactly as requested without explanation.\n<</SYS>>\n\n{p['prompt']} [/INST]\n{p['prompt']}"
            prompts.append(prompt)
                
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(model.device)
        
        # 需要生成的样本数 (测 Pass@10 就设为 10，测 Pass@100 就设为 100)
        # 建议先设为 10 跑一版看看效果
        num_samples = 100 
        
        with torch.no_grad():
            # 循环生成多次，积攒多份不同的答案
            for _ in range(num_samples):
                # 开启采样 (do_sample=True) 并设置温度参数
                outputs = model.generate(
                    **inputs, 
                    max_new_tokens=512, 
                    do_sample=True,          # 必须开启采样，否则每次生成的代码都一样
                    temperature=0.8,         # 学术界测 Pass@10 和 100 的标准温度参数
                    top_p=0.95,              # 配合温度控制生成质量
                    pad_token_id=tokenizer.eos_token_id
                )
            
                for j, output in enumerate(outputs):
                    input_len = inputs['input_ids'][j].shape[0]
                    generated_text = tokenizer.decode(output[input_len:], skip_special_tokens=True)
                    # 将模型生成的后半部分代码与前半部分 Prompt 拼接
                    full_code = batch[j]['prompt'] + generated_text
                    
                    results.append({
                        "task_id": batch[j]['task_id'],
                        "completion": full_code
                    })
    return results

def worker_generate(gpu_pair, worker_idx, model_path, lora_path, model_type, problems_chunk, batch_size):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_pair 
    model, tokenizer = setup_model(model_path, lora_path)
    return worker_idx, batch_generate_code(model, tokenizer, problems_chunk, model_type, batch_size, worker_idx)

def parallel_generate(model_path, lora_path, model_type, problems, batch_size=4, num_workers=5):
    problems_list = list(problems.values())
    chunk_size = math.ceil(len(problems_list) / num_workers)
    chunks = [problems_list[i:i+chunk_size] for i in range(0, len(problems_list), chunk_size)]
    actual_workers = len(chunks)
    
    gpu_pairs = ["0,1", "2,3", "4,5", "6,7", "8,9"]
    results = [None] * actual_workers
    ctx = mp.get_context('spawn')
    
    print(f"\n[*] 🚀 成功激活 10 张 V100 显卡，划分为 {actual_workers} 个并行代码生成节点...")
    with concurrent.futures.ProcessPoolExecutor(max_workers=actual_workers, mp_context=ctx) as executor:
        futures = [
            executor.submit(worker_generate, gpu_pairs[i], i, model_path, lora_path, model_type, chunks[i], batch_size) 
            for i in range(actual_workers)
        ]
        for future in tqdm(concurrent.futures.as_completed(futures), total=actual_workers, desc="多卡节点流水线生成中"):
            worker_idx, resp = future.result()
            results[worker_idx] = resp
            
    final_outputs = []
    for r in results:
        final_outputs.extend(r)
    return final_outputs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--lora_path", type=str, required=True)
    parser.add_argument("--model_type", type=str, choices=["llama2", "llama3"], required=True)
    parser.add_argument("--model_name", type=str, required=True)
    args = parser.parse_args()

    # 1. 读取官方题库 (含 164 道标准 Python 编程题)
    problems = read_problems()
    
    print(f"\n{'='*50}\n开始代码能力评估任务: {args.model_name}\n{'='*50}")
    
    # 2. 10卡并行生成代码答案
    samples = parallel_generate(args.model_path, args.lora_path, args.model_type, problems, batch_size=2)
    
    # 3. 将生成的代码临时写入 JSONL 文件供评估器读取
    output_filepath = f"results/evals/{args.model_name}_humaneval_samples.jsonl"
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    with open(output_filepath, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")
            
    print(f"\n[*] 💻 代码生成完毕，结果已安全封存至: {output_filepath}")
    print("[*] ⚖️ 正在启动本地沙箱进行自动化 Functional Testing...")
    
    # 4. 调用官方沙箱执行测试用例并打分
    results = evaluate_functional_correctness(output_filepath)
    
    print("\n" + "="*50)
    print(f"[{args.model_name}] 核心代码能力评测结果:")
    print(f" - Pass@1  得分: {results.get('pass@1', 0)*100:.2f}%")
    if 'pass@10' in results:
        print(f" - Pass@10 得分: {results['pass@10']*100:.2f}%")
    if 'pass@100' in results:
        print(f" - Pass@100得分: {results['pass@100']*100:.2f}%")
    print("="*50)

if __name__ == "__main__":
    try:
        mp.set_start_method('spawn', force=True) 
    except RuntimeError:
        pass
    main()
