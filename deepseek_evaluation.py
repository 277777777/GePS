import argparse
import torch
import requests
import json
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import os
from datasets import load_dataset
import multiprocessing as mp
import concurrent.futures
import math
import unicodedata
import re

class DeepSeekEvaluator:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.deepseek.com/chat/completions"
        self.headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    
    def evaluate_response(self, instruction, model_response, reference_response=None):
        # 1. 完整构建 Prompt 和 Payload
        if reference_response:
            prompt = f"""请扮演严格的AI评判员。评估两个AI助手（A和B）对问题的回答质量。
[问题]\n{instruction}\n[助手A(基准)]\n{reference_response}\n[助手B(你的模型)]\n{model_response}
给出分析，最后必须明确输出裁决：Verdict: A 或 Verdict: B 或 Verdict: TIE"""
            max_tokens = 500
        else:
            prompt = f"""请扮演严格的AI评判员。根据问题和回答给出一个1到10分的评分。
[问题]\n{instruction}\n[回答]\n{model_response}\n给出分析，最后给出一个1-10分的总分（仅需数字）。"""
            max_tokens = 300
        
        payload = {"model": "deepseek-chat", "messages": [{"role": "user", "content": prompt}], "temperature": 0.0, "max_tokens": max_tokens}
        
        # 2. 发起请求与错误处理
        for attempt in range(3):
            try:
                response = requests.post(self.base_url, headers=self.headers, json=payload, timeout=40)
                response.raise_for_status() # 拦截 429 限流或欠费
                
                content = response.json()["choices"][0]["message"]["content"].strip()
                
                # 3. 提取结果
                if reference_response:
                    if "Verdict: A" in content: return "A"
                    if "Verdict: B" in content: return "B"
                    if "Verdict: TIE" in content: return "TIE"
                    if content.endswith("A"): return "A"
                    if content.endswith("B"): return "B"
                    return "TIE"
                else:
                    content = unicodedata.normalize('NFKC', content)
                    nums = re.findall(r'\d+\.?\d*', content)
                    if nums: return nums[-1]
                    return "5.0"
                    
            except requests.exceptions.RequestException as e:
                print(f"\n[⚠️ API 网络/限流错误] 尝试 {attempt+1}/3 失败: {e}")
                if hasattr(e, 'response') and e.response is not None:
                    print(f"返回详情: {e.response.text}")
                time.sleep(5)
            except Exception as e:
                # 这次绝对不偷偷藏起本地代码错误了
                print(f"\n[⚠️ 本地代码执行异常] {type(e).__name__}: {e}")
                time.sleep(2)
                
        return "TIE" if reference_response else "5.0"

    def batch_evaluate(self, instructions, model_responses, reference_responses=None):
        results = [None] * len(instructions)
        def evaluate_single(idx):
            ref = reference_responses[idx] if reference_responses else None
            return idx, self.evaluate_response(instructions[idx], model_responses[idx], ref)

        # 开启 5 线程并发访问 API
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(evaluate_single, i) for i in range(len(instructions))]
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(instructions), desc="DeepSeek 裁判高速打分中"):
                idx, res = future.result()
                results[idx] = res
        return results

def setup_model(model_path, lora_path=None):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left" # 批量推理必须在左侧填充
    
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base_model, lora_path) if lora_path else base_model
    model.eval()
    return model, tokenizer

def batch_generate_responses(model, tokenizer, instructions, model_type, batch_size, gpu_id):
    responses = []
    for i in tqdm(range(0, len(instructions), batch_size), desc=f"GPU {gpu_id} 生成中", position=gpu_id, leave=False):
        batch_insts = instructions[i:i+batch_size]
        prompts = []
        for inst in batch_insts:
            if model_type == "llama3":
                prompts.append(f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful AI assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{inst}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n")
            else:
                prompts.append(f"<s>[INST] <<SYS>>\nYou are a helpful AI assistant.\n<</SYS>>\n\n{inst} [/INST]")
                
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=1024, do_sample=True, temperature=0.7, pad_token_id=tokenizer.eos_token_id)
        
        for j, output in enumerate(outputs):
            input_len = inputs['input_ids'][j].shape[0]
            responses.append(tokenizer.decode(output[input_len:], skip_special_tokens=True).strip())
    return responses

# 【火力全开1】多进程数据分发机制 (升级为双卡一组)
def worker_generate(gpu_pair, worker_idx, model_path, lora_path, model_type, instructions, batch_size):
    # 让当前进程只能看到分配给它的 2 张显卡
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_pair 
    # device_map="auto" 会自动把 Llama-3 均匀切分到这 2 张卡上
    model, tokenizer = setup_model(model_path, lora_path)
    return worker_idx, batch_generate_responses(model, tokenizer, instructions, model_type, batch_size, worker_idx)

def parallel_generate(model_path, lora_path, model_type, instructions, batch_size=4, num_workers=5):
    chunk_size = math.ceil(len(instructions) / num_workers)
    chunks = [instructions[i:i+chunk_size] for i in range(0, len(instructions), chunk_size)]
    actual_workers = len(chunks)
    
    # 提前分配好显卡对 (0和1一组，2和3一组...)
    gpu_pairs = ["0,1", "2,3", "4,5", "6,7", "8,9"]
    
    print(f"\n[*] 🚀 成功唤醒 10 张 V100 显卡，组成 {actual_workers} 个超大显存节点 (每节点 32GB)...")
    
    results = [None] * actual_workers
    ctx = mp.get_context('spawn')
    with concurrent.futures.ProcessPoolExecutor(max_workers=actual_workers, mp_context=ctx) as executor:
        futures = [executor.submit(worker_generate, gpu_pairs[i], i, model_path, lora_path, model_type, chunks[i], batch_size) for i in range(actual_workers)]
        for future in concurrent.futures.as_completed(futures):
            worker_idx, resp = future.result()
            results[worker_idx] = resp
            
    final_responses = []
    for r in results: final_responses.extend(r)
    return final_responses

def run_alpaca_eval_with_deepseek(args, deepseek_evaluator):
    print("\n[*] ======= 阶段 1: Alpaca-Eval 评估 =======")
    dataset = load_dataset("json", data_files="./datasets/offline_data/alpaca_eval.json", split="train")
    instructions, reference_responses = [item["instruction"] for item in dataset], [item["output"] for item in dataset]
    
    model_responses = parallel_generate(args.model_path, args.lora_path, args.model_type, instructions, batch_size=2)
    evaluations = deepseek_evaluator.batch_evaluate(instructions, model_responses, reference_responses)
    
    wins, ties, total = evaluations.count("B"), evaluations.count("TIE"), len(evaluations)
    win_rate = ((wins + 0.5 * ties) / total * 100) if total > 0 else 0
    avg_length = sum(len(r.split()) for r in model_responses) / len(model_responses) if model_responses else 0
    return {"win_rate": win_rate, "length": avg_length}

def run_mt_bench_with_deepseek(args, deepseek_evaluator):
    print("\n[*] ======= 阶段 2: MT-Bench 评估 =======")
    dataset = load_dataset("json", data_files="./datasets/offline_data/questions.jsonl", split="train")
    
    mt_bench_questions, all_questions = {}, []
    for item in dataset:
        cat = item['category']
        mt_bench_questions.setdefault(cat, []).append(item['turns'][0])
        all_questions.append(item['turns'][0])
        
    all_responses = parallel_generate(args.model_path, args.lora_path, args.model_type, all_questions, batch_size=2)
    
    import json
    with open(f"results/evals/{args.model_name}_answers.json", "w", encoding="utf-8") as f:
        json.dump(all_responses, f, ensure_ascii=False, indent=2)

    category_scores = {}
    idx = 0
    for category, qs in mt_bench_questions.items():
        cat_responses = all_responses[idx:idx+len(qs)]
        idx += len(qs)
        evals = deepseek_evaluator.batch_evaluate(qs, cat_responses)
        scores = []
        for s in evals:
            try:
                clean_s = unicodedata.normalize('NFKC', str(s))
                val = float(clean_s)
                scores.append(min(max(val, 1), 10))
            except:
                scores.append(5.0)
        category_scores[category] = sum(scores) / len(scores) if scores else 0
        
    category_scores["overall"] = sum(category_scores.values()) / len(category_scores) if category_scores else 0
    return category_scores

def run_evaluation(args):
    os.makedirs("results/evals", exist_ok=True)
    deepseek_evaluator = DeepSeekEvaluator("sk-XXXXXXXXXXXXX") 
    
    print(f"\n{'='*50}\n开始全功率评估模型: {args.model_name}\n{'='*50}")
    
    alpaca_results = run_alpaca_eval_with_deepseek(args, deepseek_evaluator)
    mt_bench_results = run_mt_bench_with_deepseek(args, deepseek_evaluator)
    
    print("\n" + "="*50)
    print(f"[{args.model_name}] Alpaca-Eval 胜率: {alpaca_results['win_rate']:.2f}% (平均长度: {alpaca_results['length']:.0f})")
    print(f"[{args.model_name}] MT-Bench 总分:   {mt_bench_results['overall']:.2f} / 10.0")
    print("="*50)

    # 简单保存为CSV方便画图
    pd.DataFrame([alpaca_results]).to_csv(f"results/evals/{args.model_name}_alpaca.csv", index=False)
    pd.DataFrame([mt_bench_results]).to_csv(f"results/evals/{args.model_name}_mtbench.csv", index=False)

if __name__ == "__main__":
    # 必须设置 spawn，否则多进程会导致 CUDA 崩溃
    try: mp.set_start_method('spawn', force=True) 
    except RuntimeError: pass
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--lora_path", type=str, required=True)
    parser.add_argument("--model_type", type=str, choices=["llama2", "llama3"], required=True)
    parser.add_argument("--model_name", type=str, required=True)
    args = parser.parse_args()
    
    run_evaluation(args)
