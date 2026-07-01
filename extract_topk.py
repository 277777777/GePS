import json
import glob
import os

def extract_top_k(original_data_path, scores_dir, output_path, top_k=6000):
    print(f"[*] 正在处理: {scores_dir}")
    
    # 1. 读取原始数据 (为了获取完整的 instruction, input, output)
    with open(original_data_path, 'r', encoding='utf-8') as f:
        original_data = json.load(f)

    # 2. 读取所有并行的打分片段
    all_scores = {}
    score_files = glob.glob(os.path.join(scores_dir, "geps_scores_*.json"))
    
    if not score_files:
        print(f"[!] 警告：在 {scores_dir} 中没有找到打分文件！")
        return

    for file_path in score_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            scores_chunk = json.load(f)
            all_scores.update(scores_chunk)

    print(f"[*] 共收集到 {len(all_scores)} 条打分记录")

    # 3. 按 GePS 分数从高到低排序 (all_scores 的 key 是原始数据的索引)
    sorted_items = sorted(
        all_scores.items(),
        key=lambda x: x[1]['geps_scores']['geps'],
        reverse=True
    )

    # 4. 提取 Top-K 的原始数据，并附上分数留作分析
    top_k_data = []
    for idx_str, score_info in sorted_items[:top_k]:
        idx = int(idx_str)
        item = original_data[idx].copy() # 复制原始数据
        item['geps_score'] = score_info['geps_scores']['geps']
        top_k_data.append(item)

    # 5. 保存最终的微调数据集
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(top_k_data, f, indent=4, ensure_ascii=False)

    print(f"[*] 成功提取 Top-{top_k} 数据并保存至: {output_path}")
    print(f"[*] 最高分: {sorted_items[0][1]['geps_scores']['geps']:.4f}")
    print(f"[*] 第{top_k}名分数: {sorted_items[top_k-1][1]['geps_scores']['geps']:.4f}\n")

if __name__ == "__main__":
    # 分别为 Llama-2 和 Llama-3 提取前 5000 名数据
    extract_top_k(
        original_data_path="./datasets/alpaca_gpt4_data.json", 
        scores_dir="./results/llama2_scores/", 
        output_path="./results/llama2_geps_6000.json"
    )
    
    extract_top_k(
        original_data_path="./datasets/alpaca_gpt4_data.json", 
        scores_dir="./results/llama3_scores/", 
        output_path="./results/llama3_geps_6000.json"
    )
