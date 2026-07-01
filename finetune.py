import os
import torch
import json
import argparse
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainerCallback
)
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

# ===== 1. 通用数据加载器 (兼容 JSON/JSONL 及 Alpaca/ShareGPT 格式) =====
def load_and_format_dataset(data_path):
    raw_data = []
    
    # 步骤 A: 智能读取 (尝试解析 JSON，如果报错就按 JSONL 逐行解析)
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            raw_data = list(data.values()) if isinstance(data, dict) else data
    except json.JSONDecodeError:
        print("[*] 检测到 JSONL 格式，正在逐行读取...")
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    raw_data.append(json.loads(line.strip()))
    
    # 步骤 B: 智能字段提取
    data_list = []
    for item in raw_data:
        instruction = ""
        output = ""
        
        # 格式 1: Alpaca 格式 (有 instruction 和 output)
        if "instruction" in item:
            instruction = item.get("instruction", "")
            inp = item.get("input", "")
            output = item.get("output", "")
            if inp and inp.strip():
                instruction += "\n" + inp
                
        # 格式 2: ShareGPT/DEITA 格式 (多轮对话 conversations)
        elif "conversations" in item:
            conversations = item["conversations"]
            for turn in conversations:
                # 兼容不同数据集的键名习惯 (from/role, value/content)
                role = turn.get("from", turn.get("role", "")).lower()
                val = turn.get("value", turn.get("content", ""))
                
                # 提取第一轮的问答
                if role in ["human", "user"] and not instruction:
                    instruction = val
                elif role in ["gpt", "assistant", "bot"] and not output:
                    output = val
                    
        # 只要成功提取到问答，就加入训练集
        if instruction and output:
            data_list.append({
                "instruction": instruction,
                "output": output
            })
            
    return Dataset.from_list(data_list)

# ===== 2. 动态模板生成器 (Llama-2 vs Llama-3) =====
def create_formatting_func(model_type):
    def format_instruction(example):
        if model_type.lower() == "llama3":
            # Llama-3 官方专属模板
            return (
                f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
                f"You are a helpful AI assistant.<|eot_id|>"
                f"<|start_header_id|>user<|end_header_id|>\n\n"
                f"{example['instruction']}<|eot_id|>"
                f"<|start_header_id|>assistant<|end_header_id|>\n\n"
                f"{example['output']}<|eot_id|>"
            )
        else:
            # Llama-2 官方专属模板
            return (
                f"<s>[INST] <<SYS>>\nYou are a helpful AI assistant.\n<</SYS>>\n\n"
                f"{example['instruction']} [/INST] {example['output']} </s>"
            )
    return format_instruction

# ===== 3. 主函数 =====
def main():
    parser = argparse.ArgumentParser(description="Universal LLM Finetuning Script")
    parser.add_argument("--model_path", type=str, required=True, help="基础模型路径")
    parser.add_argument("--dataset_path", type=str, required=True, help="微调数据集路径")
    parser.add_argument("--output_dir", type=str, required=True, help="微调后模型保存路径")
    parser.add_argument("--model_type", type=str, choices=["llama2", "llama3"], required=True, help="模型架构类型")
    parser.add_argument("--epochs", type=int, default=3, help="训练轮数 (Alpaca基线通常用3)")
    parser.add_argument("--deepspeed", type=str, default=None, help="DeepSpeed 配置文件路径")
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", -1))

    # 【核心修复 1】: 必须先声明 SFTConfig！
    # 让 DeepSpeed 提前就绪，准备在下一步加载模型时进行实时切片
    training_arguments = SFTConfig(
        output_dir=args.output_dir,
        max_length=1024,             # 修复了 TRL 新版本的规范参数名
        num_train_epochs=args.epochs,
        per_device_train_batch_size=2,   # ZeRO-3 切片后显存极其充裕，设为 2 甚至 4 都没问题
        gradient_accumulation_steps=8,   
        optim="adamw_torch",             
        save_strategy="epoch",
        logging_steps=5,
        learning_rate=2e-5,              
        weight_decay=0.01,
        fp16=True,                       
        bf16=False,                      
        max_grad_norm=1.0,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        remove_unused_columns=False,
        dataset_text_field="instruction",
        ddp_find_unused_parameters=False,
        gradient_checkpointing_kwargs={'use_reentrant': True},
        deepspeed=args.deepspeed         # 传入 DeepSpeed 配置
    )

    print(f"[*] 准备加载数据集: {args.dataset_path}")
    train_dataset = load_and_format_dataset(args.dataset_path)
    print(f"[*] 成功加载 {len(train_dataset)} 条训练数据！")

    print(f"[*] 正在从 {args.model_path} 加载模型...")
    # 【核心修复 2】: 使用 ZeRO-3 时，千万不能用 device_map="auto"
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16, 
        device_map=None,  # DeepSpeed 会完全接管显存分配，必须置为 None
        local_files_only=True
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, local_files_only=True)
    
    if tokenizer.pad_token is None:
        if args.model_type.lower() == "llama3":
            tokenizer.pad_token = "<|eot_id|>"
        else:
            tokenizer.pad_token = tokenizer.unk_token or tokenizer.eos_token
    tokenizer.padding_side = "right"

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    peft_config = LoraConfig(
        lora_alpha=32,
        lora_dropout=0.05,
        r=16,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    class LoggingCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if 'loss' in logs and local_rank in [-1, 0]: 
                print(f"Step {state.global_step}: Loss = {logs['loss']:.4f}")

    formatting_func = create_formatting_func(args.model_type)

    trainer = SFTTrainer(
        model=model,
        args=training_arguments,
        train_dataset=train_dataset,
        peft_config=peft_config,
        formatting_func=formatting_func,
        callbacks=[LoggingCallback()],
    )

    print(f"[*] 开始进行微调，使用架构: {args.model_type.upper()}")
    trainer.train()

# 【核心修复】：使用 trainer.save_model，它能自动识别 DeepSpeed 并跨卡收集权重！
    trainer.save_model(args.output_dir) 
    
    if local_rank in [-1, 0]:
        tokenizer.save_pretrained(args.output_dir)
        print(f"[*] 微调完成！模型已完整拼接并保存至: {args.output_dir}")

if __name__ == "__main__":
    main()
