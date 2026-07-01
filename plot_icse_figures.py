import matplotlib.pyplot as plt
import numpy as np
import os

# 确保输出目录存在
os.makedirs('./images', exist_ok=True)

# ==========================================
# 图 1: 通用性能对比 (极简故事流)
# ==========================================
# 核心阵列：底线(52k) -> 上限(9k) -> 竞品(DEITA) -> GePS缩放规律
models_gen = ['Source\n(52k)', 'ChatGPT\n(9k)', 'DEITA\n(6k)', 'GePS\n(4k)', 'GePS\n(5k, Ours)', 'GePS\n(6k)']

# 对应的数据 (已剔除冗余基线)
alpaca_scores = [54.97, 61.18, 59.01, 62.30, 61.68, 60.37]
mt_bench_scores = [5.75, 6.45, 6.18, 6.48, 6.96, 6.41]

# 优化排版比例
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

# 颜色策略：52k(深灰警告), 9k/DEITA(浅蓝基线), GePS系列(橙色渐变强调)
colors = ['#9E9E9E', '#90CAF9', '#90CAF9', '#FFCC80', '#FF8A65', '#E64A19']

# 子图 1: Alpaca-Eval
bars1 = ax1.bar(models_gen, alpaca_scores, color=colors, edgecolor='black', width=0.55)
ax1.set_ylim(52, 64) # 适配52k的低分
ax1.set_ylabel('Alpaca-Eval Win Rate (%)', fontsize=11, fontweight='bold')
ax1.set_title('(a) Alpaca-Eval Performance', fontsize=13, fontweight='bold')
ax1.tick_params(axis='x', labelsize=10) # 不再需要倾斜，因为只有6个且带换行
ax1.grid(axis='y', linestyle='--', alpha=0.7)

for bar in bars1:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, yval + 0.15, f'{yval}%', ha='center', va='bottom', fontsize=9.5)

# 子图 2: MT-Bench
bars2 = ax2.bar(models_gen, mt_bench_scores, color=colors, edgecolor='black', width=0.55)
ax2.set_ylim(5.5, 7.2)
ax2.set_ylabel('MT-Bench Score (1-10)', fontsize=11, fontweight='bold')
ax2.set_title('(b) MT-Bench Performance', fontsize=13, fontweight='bold')
ax2.tick_params(axis='x', labelsize=10)
ax2.grid(axis='y', linestyle='--', alpha=0.7)

for bar in bars2:
    yval = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2, yval + 0.03, f'{yval}', ha='center', va='bottom', fontsize=9.5)

plt.tight_layout()
plt.savefig('./images/general_performance.pdf', format='pdf', bbox_inches='tight', dpi=300)
plt.close()

# ==========================================
# 图 2: HumanEval 代码能力保真度对比
# ==========================================
# 加入 52k 作为技术债务的视觉冲击点
models_he = ['Source\n(52k)', 'ChatGPT\n(9k)', 'IFD\n(5k)', 'GePS\n(5k, Ours)']
pass_1 = [9.33, 12.62, 10.85, 12.50]
pass_10 = [47.56, 60.37, 54.88, 56.10]

x = np.arange(len(models_he))
width = 0.3

fig, ax = plt.subplots(figsize=(7, 4.5))
rects1 = ax.bar(x - width/2, pass_1, width, label='Pass@1', color='#90CAF9', edgecolor='black')
rects2 = ax.bar(x + width/2, pass_10, width, label='Pass@10', color='#FF8A65', edgecolor='black')

ax.set_ylabel('HumanEval Pass Rate (%)', fontsize=11, fontweight='bold')
ax.set_title('Downstream SE Task Preservation (Code Generation)', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(models_he, fontsize=10)
ax.set_ylim(0, 70) 
ax.legend(fontsize=10)
ax.grid(axis='y', linestyle='--', alpha=0.7)

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9.5)

autolabel(rects1)
autolabel(rects2)

plt.tight_layout()
plt.savefig('./images/humaneval_preservation.pdf', format='pdf', bbox_inches='tight', dpi=300)
plt.close()

print("[*] 成功生成极简高清图表: general_performance.pdf 和 humaneval_preservation.pdf")
