# 下面针对生成的ID嵌入进行分析

import json
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict, Counter
from matplotlib.ticker import PercentFormatter
import pandas as pd
import powerlaw

sequence_path = "./data/Instruments/Instruments.inter.json"
data_path = "./data/Instruments/Instruments.index.json"
self_trained_collision_rate_oriented = "./data/Instruments/Instruments.index.epoch10.alpha1e-1-beta1e-4.json"
self_trained_loss_oriented = "./data/Instruments/Instruments.index.epoch19999.alpha1e-1-beta1e-4.json"

with open(self_trained_loss_oriented, "r") as f:
    data = json.load(f)  # load means loading json object from documents , loads means loading from the str


with open(sequence_path, "r") as f:
    seq_data = json.load(f)

# 辅助函数：提取语义ID
def extract_id(item, pos):
    return item.split('_')[1].strip('>') if item.startswith(f'<{pos}_') else None

def all_position_distribution(seq_data, data):
    position_counts = {pos: defaultdict(int) for pos in ['a', 'b', 'c', 'd']}
    
    for user, seq_items in seq_data.items():
        for items in seq_items:
            semetic_id = data.get(str(items), None)
            if semetic_id is None:
                print(f"Item {items} missing")
            else:
                # do something
                1;
            for i, item in enumerate(semetic_id):
                pos = ['a', 'b', 'c', 'd'][i]
                item_id = extract_id(item, pos)
                if item_id:
                    position_counts[pos][item_id] += 1
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, (pos, counts) in enumerate(position_counts.items()):
        sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        ids, freqs = zip(*sorted_counts)
        
        ax = axes[i]
        # ax.bar(ids, freqs, color='skyblue')
        ax.plot(range(len(ids)), freqs, color='skyblue', marker='o')
        ax.set_title(f'Position {pos.upper()} - all IDs')
        ax.set_xlabel('Semantic ID index')
        ax.set_ylabel('Frequency')
        # ax.tick_params(axis='x', rotation=45)

        step = max(1, len(ids) // 20);
        indexs = range(1, len(ids), step)
        x_tick = [ids[id] for id in indexs]
        ax.set_xticks(indexs)
        ax.set_xticklabels(x_tick, rotation=45, ha='right')

        
        # 添加统计信息
        total = sum(counts.values())
        unique = len(counts)
        ax.annotate(f'Total items: {total}\nUnique IDs: {unique}', 
                   xy=(0.7, 0.9), xycoords='axes fraction')
    
    plt.tight_layout()
    plt.savefig('position_distribution_self.png')
    plt.close()
    
    return position_counts

# 2. ID交互频率统计与长尾分布验证
def frequency_analysis(position_counts):
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, (pos, counts) in enumerate(position_counts.items()):
        freqs = sorted(counts.values(), reverse=True)
        
        fit = powerlaw.Fit(freqs, discrete=True)
        print("alpha (幂律指数):", fit.power_law.alpha)
        print("xmin (幂律起点):", fit.power_law.xmin)

        cumulative = np.cumsum(freqs) / sum(freqs)
        
        # 长尾分布可视化
        ax = axes[i]
        ax.plot(range(1, len(freqs)+1), cumulative, 'b-')
        ax.set_title(f'Position {pos.upper()} - Cumulative Frequency')
        ax.set_xlabel('Rank of ID (by frequency)')
        ax.set_ylabel('Cumulative Proportion')
        ax.grid(True)
        ax.set_yscale('log')
        ax.set_xscale('log')
        
        # 计算长尾分布指标
        top_20 = cumulative[int(len(freqs)*0.2)-1]
        ax.axvline(x=len(freqs)*0.2, color='r', linestyle='--')
        ax.annotate(f'Top 20% IDs cover {top_20:.1%} of interactions', 
                   xy=(0.4, 0.1), xycoords='axes fraction')
    
    plt.tight_layout()
    plt.savefig('long_tail_analysis_self.png')
    plt.close()

# 3. 按行为序列长度分类
def analyze_by_length(data):
    # 计算序列长度
    length_groups = defaultdict(list)
    for user, items in data.items():
        length = len(items)
        length_groups[length].append(user)
    
    # 打印分组信息
    print("\n用户按行为序列长度分组:")
    for length, users in sorted(length_groups.items()):
        print(f"长度 {length}: {len(users)} 用户")
    
    # 4. 分组统计物品数和冲突数量
    results = {}
    for length, users in length_groups.items():
        group_data = {user: data[user] for user in users}
        total_items = set()
        conflict_count = 0
        
        for items in group_data.values():
            # 统计所有物品
            ids = [item.split('_')[1].strip('>') for item in items]
            total_items.update(ids)
            
            # 检测冲突（同一用户相同ID出现在不同位置）
            pos_ids = defaultdict(set)
            for item in items:
                pos = item[1]
                id_val = item.split('_')[1].strip('>')
                pos_ids[pos].add(id_val)
            
            # 检查不同位置间的ID冲突
            all_ids = [id_set for id_set in pos_ids.values()]
            for i in range(len(all_ids)):
                for j in range(i+1, len(all_ids)):
                    if all_ids[i] & all_ids[j]:
                        conflict_count += 1
        
        results[length] = {
            'num_users': len(users),
            'total_items': len(total_items),
            'conflicts': conflict_count
        }
    
    # 输出结果
    print("\n分组统计结果:")
    for length, stats in results.items():
        print(f"序列长度 {length}:")
        print(f"  用户数: {stats['num_users']}")
        print(f"  唯一物品数: {stats['total_items']}")
        print(f"  跨位置冲突数: {stats['conflicts']}")
    
    return results

def analyze_distribution_by_length(seq_data, data):
    # 计算序列长度
    length_groups = defaultdict(list)
    all_lengths = []
    for user, items in seq_data.items():
        length = len(items)
        all_lengths.append(length)
        length_groups[length].append(user)

    quantiles = np.array([10, 20])

    categories = {
        "short": [],
        "medium": [],
        "long": []
    }
    
    for length, users in length_groups.items():
        if length <= quantiles[0]:
            categories["short"].extend(users)
        elif length <= quantiles[1]:
            categories["medium"].extend(users)
        else:
            categories["long"].extend(users)

    print("\n用户按行为序列长度分组:")
    for cat, users in categories.items():
        print(f"{cat}序列: {len(users)} 用户 (长度范围: {min([len(seq_data[u]) for u in users]) if users else 'N/A'}-{max([len(seq_data[u]) for u in users]) if users else 'N/A'})")
     
    # 2. 分组统计物品数和冲突数量
    results = {}
    for cat, users in categories.items():
        group_data = {user: seq_data[user] for user in users}
        total_items = set()
        conflict_count = 0
        
        for user_items in group_data.values():
            # 统计所有物品
            for item in user_items:
                total_items.add(item)
            
            # 检测冲突（同一用户相同ID出现在不同位置）
            pos_ids = defaultdict(set)
            for item in user_items:
                item = str(item)
                semantic_ids = data.get(str(item), None)
                if semantic_ids is None:
                    print(f"Item {item} missing")
                else:
                    # do something
                    1;
                # print(list(data.keys())[:10])  # 看看都是字符串还是整数

                # semantic_ids = data[item]

                for sid in semantic_ids:
                    pos = sid.split('_')[0].strip('<')
                    id_val = sid.split('_')[1].strip('>')
                    pos_ids[pos].add(id_val)
            
            # 检查不同位置间的ID冲突
            all_positions = list(pos_ids.values())
            for i in range(len(all_positions)):
                for j in range(i+1, len(all_positions)):
                    if all_positions[i] & all_positions[j]:
                        conflict_count += 1
        
        results[cat] = {
            'num_users': len(users),
            'total_items': len(total_items),
            'conflicts': conflict_count,
            'avg_sequence_length': np.mean([len(seq_data[u]) for u in users]) if users else 0
        }
    
    # 3. 对每个位置上的ID进行交互频率统计和长尾分析
    position_freqs = {}
    position_long_tail = {}
    
    # 为每个位置创建频率计数器
    positions = ['a', 'b', 'c', 'd']
    for pos in positions:
        position_freqs[pos] = {}
        position_long_tail[pos] = {}
        
        for cat in categories:
            position_freqs[pos][cat] = Counter()
    
    # 统计每个位置每个类别的频率
    for cat, users in categories.items():
        for user in users:
            for item in seq_data[user]:
                # semantic_ids = data[item]
                semantic_ids = data.get(str(item), None)
                if semantic_ids is None:
                    print(f"Item {item} missing")
                else:
                    # do something
                    1;
                for sid in semantic_ids:
                    pos = sid.split('_')[0].strip('<')
                    id_val = sid.split('_')[1].strip('>')
                    if pos in positions:
                        position_freqs[pos][cat][id_val] += 1
    
    # 4. 验证长尾分布并绘图
    plt.figure(figsize=(15, 10))
    
    for i, pos in enumerate(positions, 1):
        plt.subplot(2, 2, i)
        
        for cat in categories:
            # 获取排序后的频率
            freqs = sorted(position_freqs[pos][cat].values(), reverse=True)
            if not freqs:
                continue
                
            # 计算累积频率
            cum_freq = np.cumsum(freqs) / sum(freqs)
            
            # 绘制长尾分布曲线
            plt.plot(range(1, len(freqs)+1), cum_freq, label=f"{cat} (n={results[cat]['num_users']})")
            
            # 计算并存储长尾指标
            top_20_index = int(len(freqs) * 0.2)
            top_20_ratio = cum_freq[top_20_index] if top_20_index < len(cum_freq) else 1.0
            position_long_tail[pos][cat] = top_20_ratio
            
            # 添加长尾指标标注
            plt.annotate(f"{cat}: Top 20% IDs cover {top_20_ratio:.1%}", 
                         xy=(0.05, 0.1 + 0.1 * list(categories.keys()).index(cat)),
                         xycoords='axes fraction')
        
        plt.title(f'Position {pos.upper()} - Cumulative Frequency Distribution')
        plt.xlabel('ID Rank (by frequency)')
        plt.ylabel('Cumulative Proportion')
        plt.legend()
        plt.grid(True)
        plt.xscale('log')
    
    plt.tight_layout()
    plt.savefig('./analysis/png/long_tail_by_category_position.png')
    plt.close()
    
    # 5. 绘制长尾指标对比图
    plt.figure(figsize=(12, 8))
    width = 0.2
    x = np.arange(len(positions))
    
    for i, cat in enumerate(categories):
        ratios = [position_long_tail[pos][cat] * 100 for pos in positions]
        plt.bar(x + i*width, ratios, width=width, label=f"{cat} sequence")
    
    plt.title('Top 20% ID Coverage by Position and Sequence Length')
    plt.xlabel('Position')
    plt.ylabel('Coverage (%)')
    plt.xticks(x + width, [pos.upper() for pos in positions])
    plt.legend()
    plt.grid(axis='y')
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig('./analysis/png/long_tail_coverage.png')
    plt.close()
    
    # 6. 输出结果
    print("\n分组统计结果:")
    for cat, stats in results.items():
        print(f"{cat}序列:")
        print(f"  用户数: {stats['num_users']}")
        print(f"  平均序列长度: {stats['avg_sequence_length']:.2f}")
        print(f"  唯一物品数: {stats['total_items']}")
        print(f"  跨位置冲突数: {stats['conflicts']}")
    
    # 7. 保存结果到CSV
    # 保存分组统计
    group_df = pd.DataFrame(results).T
    group_df.reset_index(inplace=True)
    group_df.rename(columns={'index': 'category'}, inplace=True)
    group_df.to_csv('./analysis/csv/group_analysis.csv', index=False)
    
    # 保存位置频率
    freq_data = []
    for pos in positions:
        for cat in categories:
            for id_val, count in position_freqs[pos][cat].items():
                freq_data.append({
                    'position': pos,
                    'category': cat,
                    'id': id_val,
                    'count': count
                })
    
    freq_df = pd.DataFrame(freq_data)
    freq_df.to_csv('./analysis/csv/position_frequencies.csv', index=False)
    
    return results, position_freqs



# 执行分析
print("开始分析...")
pos_counts = all_position_distribution(seq_data, data)
frequency_analysis(pos_counts)
length_results = analyze_distribution_by_length(seq_data, data)

print("\n分析完成! 结果已保存到:")
print("- position_distribution.png")
print("- long_tail_analysis.png")