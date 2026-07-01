import hashlib
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import math
import gc
import json
import logging
import os
import textwrap
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM

# 设置日志和路径
logger_root = "./logs"

def ensure_folder(folder_path, parents=False):
    """确保文件夹存在"""
    os.makedirs(folder_path, exist_ok=True)

def setup_logger(log_folder, log_file_name, console_output=True):
    """设置日志记录器"""
    ensure_folder(log_folder)
    
    # 创建日志记录器
    logger = logging.getLogger("geps")
    logger.setLevel(logging.INFO)
    
    # 清除现有的处理器
    logger.handlers.clear()
    
    # 创建文件处理器
    file_handler = logging.FileHandler(os.path.join(log_folder, log_file_name))
    file_handler.setLevel(logging.INFO)
    
    # 创建控制台处理器
    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
    
    # 创建格式化器
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    if console_output:
        console_handler.setFormatter(formatter)
    
    # 添加处理器到记录器
    logger.addHandler(file_handler)
    if console_output:
        logger.addHandler(console_handler)
    
    return logger

class GePSCalculator:
    def __init__(self, model_path, device=None):
        """
        初始化GePS计算器 (已兼容 Llama-2 & Llama-3)
        """
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logging.getLogger("geps.calculator")
        
        # 加载本地模型和tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        # Llama-3 特殊的 Padding token 处理
        if self.tokenizer.pad_token is None:
            # Llama-3 通常使用 <|eot_id|> 或 <|end_of_text|>
            if "<|eot_id|>" in self.tokenizer.vocab:
                self.tokenizer.pad_token = "<|eot_id|>"
            else:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 【关键修复 1】: Llama-3 强烈建议使用 bfloat16，float16 极易导致 loss 溢出变为 NaN
        torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        
        if torch.cuda.is_available():
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                dtype=torch_dtype,
                device_map="auto",
                trust_remote_code=True,
                attn_implementation="eager" # 强制 Eager 模式提取完整的 Attention 矩阵，自动处理 GQA
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                dtype=torch.float32,
                trust_remote_code=True,
                attn_implementation="eager"
            ).to(self.device)
        
        self.model.eval()
        self.config = self.model.config
        self.cache = {}

    def calculate_entropy_normalized(self, instruction):
        """计算归一化的指令熵"""
        try:
            inputs = self.tokenizer(
                instruction, 
                return_tensors="pt", 
                truncation=True, 
                max_length=512,
                padding=True
            ).to(self.device)
            input_ids = inputs["input_ids"]
            
            if input_ids.size(1) <= 1:
                return 0.5
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = input_ids[:, 1:].contiguous()
                
                loss_fct = nn.CrossEntropyLoss(reduction='none', ignore_index=self.tokenizer.pad_token_id)
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                
                non_pad_mask = shift_labels.view(-1) != self.tokenizer.pad_token_id
                
                if non_pad_mask.any():
                    valid_loss = loss[non_pad_mask]
                    # 【关键修复 2】: 防止 BFloat16/Float16 下指数溢出
                    mean_loss = torch.clamp(valid_loss.mean(), max=20.0) 
                    perplexity_tensor = torch.exp(mean_loss)
                    perplexity_val = perplexity_tensor.item()
                else:
                    perplexity_val = 1.0

                if perplexity_val < 1.0 or math.isnan(perplexity_val):
                    perplexity_val = 1.0
                elif perplexity_val > 1e8:
                    perplexity_val = 1e8

                log_perplexity = math.log(perplexity_val + 1e-6)
                entropy = torch.sigmoid(torch.tensor(log_perplexity)).item()
            
            return entropy
        except Exception as e:
            self.logger.error(f"Error in Entropy: {e}")
            return 0.5 

    def calculate_activation_normalized(self, instruction):
        """计算归一化的激活强度 (完美兼容 MHA / GQA)"""
        try:
            cache_key = f"activation_{hashlib.md5(instruction.encode()).hexdigest()}"
            if cache_key in self.cache:
                return self.cache[cache_key]
                
            inputs = self.tokenizer(
                instruction, 
                return_tensors="pt", 
                truncation=True, 
                max_length=512,
                padding=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs, output_attentions=True)
                attentions = outputs.attentions
                
                if attentions is None or len(attentions) == 0:
                    return 0.5
                
                # [batch_size, num_heads, seq_len, seq_len]
                last_layer_attentions = attentions[-1]
                batch_size, num_heads, seq_len, _ = last_layer_attentions.shape
                
                # 【关键修复 3】: 精确剥离 Padding Tokens，仅在有效文本长度内计算KL散度
                attention_mask = inputs["attention_mask"][0]
                valid_seq_len = int(attention_mask.sum().item())
                
                if valid_seq_len <= 1:
                    return 0.5
                
                uniform_dist = torch.ones(valid_seq_len, device=self.device) / valid_seq_len
                
                kl_divergences = []
                for head_idx in range(num_heads):
                    # 仅截取有效 token 部分
                    head_attentions = last_layer_attentions[0, head_idx, :valid_seq_len, :valid_seq_len]
                    head_dist = head_attentions.mean(dim=0)
                    
                    # 【关键修复 4】: Llama-3 注意力极其稀疏，加入极小值 eps 防止 log(0) -> NaN
                    head_dist = head_dist + 1e-12
                    head_dist = head_dist / head_dist.sum()
                    
                    kl_div = F.kl_div(
                        head_dist.log(), 
                        uniform_dist, 
                        reduction='sum'
                    ).item()
                    kl_divergences.append(kl_div)
                
                activation = np.mean(kl_divergences) if kl_divergences else 0.0
                
                # 【关键修复 5】: KL散度理论最大值约为 ln(N)，这里进行真正的 0-1 缩放，抛弃会导致分数聚集的 sigmoid
                max_kl = math.log(valid_seq_len) if valid_seq_len > 1 else 1.0
                normalized_activation = min(max(activation / max_kl, 0.0), 1.0)
            
            self.cache[cache_key] = normalized_activation
            return normalized_activation
        except Exception as e:
            self.logger.error(f"Error in Activation: {e}")
            return 0.5

    def calculate_stability_normalized(self, instruction, num_perturbations=3):
        """计算归一化的语义稳定性"""
        try:
            # 代码逻辑非常棒，这里只微调了代码以适应模型并行和安全加载
            original_inputs = self.tokenizer(instruction, return_tensors="pt", truncation=True, max_length=512).to(self.device)
            with torch.no_grad():
                original_outputs = self.model(**original_inputs, output_hidden_states=True)
                original_embedding = original_outputs.hidden_states[-1].mean(dim=1)
            
            similarities = []
            for _ in range(num_perturbations):
                perturbed_instruction = self.perturb_instruction(instruction)
                perturbed_inputs = self.tokenizer(perturbed_instruction, return_tensors="pt", truncation=True, max_length=512).to(self.device)
                
                with torch.no_grad():
                    perturbed_outputs = self.model(**perturbed_inputs, output_hidden_states=True)
                    perturbed_embedding = perturbed_outputs.hidden_states[-1].mean(dim=1)
                
                cos_sim = F.cosine_similarity(original_embedding, perturbed_embedding)
                if torch.isnan(cos_sim).any():
                    cos_sim = torch.tensor(1.0).to(self.device)
                cos_sim = (cos_sim + 1) / 2
                similarities.append(cos_sim.item())
            
            stability = np.mean(similarities) if similarities else 0.5
            return stability
        except Exception as e:
            self.logger.error(f"Error in Stability: {e}")
            return 0.5

    def perturb_instruction(self, instruction):
        # 保持您原本出色的扰动逻辑不变
        words = instruction.split()
        if len(words) <= 1:
            return instruction
        if np.random.random() < 0.3:
            i, j = np.random.choice(len(words), 2, replace=False)
            words[i], words[j] = words[j], words[i]
        elif np.random.random() < 0.6:
            idx = np.random.randint(0, len(words))
            words[idx] = words[idx] + "."
        else:
            idx = np.random.randint(0, len(words))
            if np.random.random() < 0.5 and len(words[idx]) > 1:
                words[idx] = words[idx][:-1]
            else:
                words[idx] = words[idx] + " "
        return " ".join(words)
    
    def calculate_geps_normalized(self, instruction):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()    
        entropy_score = self.calculate_entropy_normalized(instruction)
        activation_score = self.calculate_activation_normalized(instruction)
        stability_score = self.calculate_stability_normalized(instruction)
        
        geps_score = (entropy_score * activation_score * stability_score) ** (1/3)
        return {"geps": geps_score, "entropy": entropy_score, "activation": activation_score, "stability": stability_score}
    

def load_dataset(file_path):
    """加载数据集"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        logging.error(f"Error loading dataset: {e}")
        return []

def process_instructions(data, geps_calculator, output_file, start=0, pace=None, logger=None):
    """处理指令并计算GePS分数"""
    results = {}
    
    # 确定处理范围
    end = start + pace if pace else len(data)
    if end > len(data):
        end = len(data)
    
    if logger:
        logger.info(f"Processing instructions from {start} to {end}")
    
    for idx in tqdm(range(start, end), desc="Calculating GePS scores"):
        item = data[idx]
        
        # 获取指令文本
        if isinstance(item, dict):
            instruction = item.get("instruction", "")
            if "input" in item and item["input"]:
                instruction += "\n" + item["input"]
        else:
            instruction = str(item)
        
        if not instruction.strip():
            if logger:
                logger.warning(f"Empty instruction at index {idx}")
            continue
            
        try:
            # 计算GePS分数（使用归一化版本）
            scores = geps_calculator.calculate_geps_normalized(instruction)
            
            # 保存结果
            results[idx] = {
                "instruction": instruction,
                "geps_scores": scores
            }
            
            if logger:
                logger.info(f"Processed instruction {idx}: GePS={scores['geps']:.6f}")
            
        except Exception as e:
            if logger:
                logger.error(f"Error processing instruction {idx}: {e}")
            continue
    
    # 保存结果到文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    
    return results

def analyze_results(results):
    """分析结果并生成统计信息"""
    if not results:
        return {
            "geps": {"mean": 0, "std": 0, "min": 0, "max": 0, "median": 0},
            "entropy": {"mean": 0, "std": 0, "min": 0, "max": 0},
            "activation": {"mean": 0, "std": 0, "min": 0, "max": 0},
            "stability": {"mean": 0, "std": 0, "min": 0, "max": 0}
        }
    
    geps_scores = [item["geps_scores"]["geps"] for item in results.values()]
    entropy_scores = [item["geps_scores"]["entropy"] for item in results.values()]
    activation_scores = [item["geps_scores"]["activation"] for item in results.values()]
    stability_scores = [item["geps_scores"]["stability"] for item in results.values()]
    
    stats = {
        "geps": {
            "mean": np.mean(geps_scores),
            "std": np.std(geps_scores),
            "min": np.min(geps_scores),
            "max": np.max(geps_scores),
            "median": np.median(geps_scores)
        },
        "entropy": {
            "mean": np.mean(entropy_scores),
            "std": np.std(entropy_scores),
            "min": np.min(entropy_scores),
            "max": np.max(entropy_scores)
        },
        "activation": {
            "mean": np.mean(activation_scores),
            "std": np.std(activation_scores),
            "min": np.min(activation_scores),
            "max": np.max(activation_scores)
        },
        "stability": {
            "mean": np.mean(stability_scores),
            "std": np.std(stability_scores),
            "min": np.min(stability_scores),
            "max": np.max(stability_scores)
        }
    }
    
    return stats

def main():
    # 创建参数解析器
    parser = argparse.ArgumentParser(description="Calculate GePS scores for instructions")
    parser.add_argument("--dataset", type=str, default="alpaca", help="Dataset name")
    parser.add_argument("--model_path", type=str, required=True, help="Path to local model files")
    parser.add_argument("--prompt_path", type=str, required=True, help="Path to prompt data")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save results")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size (set to 1 for large models)")
    parser.add_argument("--gpus", type=str, default="0", help="GPU IDs to use")
    parser.add_argument("--start", type=int, default=0, help="Start index")
    parser.add_argument("--pace", type=int, default=10, help="Number of instructions to process")
    parser.add_argument("--num_perturbations", type=int, default=3, help="Number of perturbations for stability calculation")
    
    args = parser.parse_args()
    
    # 设置环境
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    
    # 设置日志
    ensure_folder(args.save_path, parents=True)
    logger = setup_logger(args.save_path, "geps_calculation.log", console_output=True)
    
    logger.info("Starting GePS calculation")
    logger.info(f"Arguments: {args}")
    
    try:
        # 加载数据集
        data = load_dataset(args.prompt_path)
        logger.info(f"Loaded dataset with {len(data)} instructions")
        
        if len(data) == 0:
            logger.error("No data loaded. Exiting.")
            return
        
        # 初始化GePS计算器
        geps_calculator = GePSCalculator(
            model_path=args.model_path,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        
        # 处理指令
        results = process_instructions(
            data, 
            geps_calculator, 
            os.path.join(args.save_path, f"geps_scores_{args.start}_{args.start+args.pace}.json"),
            start=args.start,
            pace=args.pace,
            logger=logger
        )
        
        # 分析结果
        stats = analyze_results(results) 
        # 保存统计信息
        with open(os.path.join(args.save_path, f"geps_stats_{args.start}_{args.start+args.pace}.json"), 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=4, ensure_ascii=False)
        
        logger.info("GePS calculation completed successfully")
        logger.info(f"GePS scores range: {stats['geps']['min']:.6f} - {stats['geps']['max']:.6f}")
        logger.info(f"Average GePS score: {stats['geps']['mean']:.6f}")
        
    except Exception as e:
        logger.error(f"Error in GePS calculation: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()
