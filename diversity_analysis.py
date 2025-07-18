import json
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict, Counter
import pandas as pd

def load_data(sequence_path, index_path):
    """加载序列数据和索引数据"""
    with open(index_path, "r") as f:
        data = json.load(f)
    
    with open(sequence_path, "r") as f:
        seq_data = json.load(f)
    
    return data, seq_data

def categorize_users(seq_data, quantiles=[10, 20]):
    """将用户按序列长度分类"""
    length_groups = defaultdict(list)
    for user, items in seq_data.items():
        length = len(items)
        length_groups[length].append(user)
    
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
    
    return categories

def analyze_id_distribution(seq_data, data, categories):
    """分析不同类别用户的ID分布情况"""
    position_freqs = {pos: {cat: Counter() for cat in categories} 
                     for pos in ['a', 'b', 'c', 'd']}
    
    # 统计每个位置每个类别的ID频率
    cat_accu = defaultdict(int)

    for cat, users in categories.items():

        for user in users:
            for item in seq_data[user]:
                semantic_ids = data.get(str(item), [])
                for sid in semantic_ids:
                    pos = sid.split('_')[0].strip('<')
                    id_val = sid.split('_')[1].strip('>')
                    if pos in position_freqs:
                        position_freqs[pos][cat][id_val] += 1
                        cat_accu[cat] += 1
    
    # 绘制ID分布图
    plt.figure(figsize=(15, 10))
    for i, pos in enumerate(['a', 'b', 'c', 'd'], 1):
        plt.subplot(2, 2, i)
        
        for cat in categories:
            freqs = position_freqs[pos][cat]
            # sorted_ids = sorted(freqs.keys(), key=lambda x: freqs[x], reverse=True)
            ids = list(freqs.keys())
            counts = list(a / cat_accu[cat] for a in freqs.values())
            # sorted_counts = [freqs[id] for id in sorted_ids]
            
            # plt.plot(range(len(sorted_ids)), sorted_counts, label=cat)
            plt.plot(range(len(ids)), counts, label=cat)
            plt.title(f'Position {pos.upper()} - ID Distribution')
            plt.xlabel('Semantic ID')
            plt.ylabel('Frequency')
            plt.legend()
            plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('./analysis/png/id_distribution_by_category.png')
    plt.close()
    
    return position_freqs

def calculate_diversity(position_freqs, categories, data, seq_data):
    """计算每个位置每个类别的ID重复率"""
    diversity_results = {pos: {} for pos in ['a', 'b', 'c', 'd']}
    
    for pos in diversity_results:
        for cat,users in categories.items():

            # freqs = position_freqs[pos][cat]
            # total = sum(freqs.values())
            unique_ids = set()
            total_ids = 0
            sum_freq = 0

            for user in users:
                for item in seq_data[user]:
                    semantic_ids = data.get(str(item), [])
                    for sid in semantic_ids:
                        pos_ = sid.split('_')[0].strip('<')
                        id_val = sid.split('_')[1].strip('>')
                        if pos == pos_:
                            total_ids += 1
                            unique_ids.add(id_val)
                unique = len(unique_ids)
                sum_freq += unique/total_ids if total_ids > 0 else 0
                total_ids = 0
                unique_ids = set()
            
            sum_freq /= len(users) if users else 1
                                
            # 重复率 = 1 - (唯一ID数 / 总交互数)
            repetition_rate = 1 - (unique_ids / total_ids) if total_ids > 0 else 0
            diversity_results[pos][cat] = {
                'unique_ids': 1,
                'total_interactions': 1,
                'repetition_rate': 1 - sum_freq
            }
            print(f"Position {pos.upper()}, Category {cat}: "
                  f"Unique IDs = {1 - sum_freq}, ")
    
    # 绘制重复率热图
    fig, ax = plt.subplots(figsize=(10, 6))
    # 构建正确的二维数组：行=位置，列=类别
    data = [[diversity_results[pos][cat]['repetition_rate'] for cat in categories] 
            for pos in diversity_results]
    im = ax.imshow(data, cmap='YlOrRd')
    
    ax.set_xticks(np.arange(len(categories)))
    ax.set_yticks(np.arange(len(diversity_results)))
    ax.set_xticklabels(categories)
    ax.set_yticklabels(diversity_results.keys())
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    for i in range(len(diversity_results)):
        for j in range(len(categories)):
            text = ax.text(j, i, f"{data[i][j]:.2f}",
                          ha="center", va="center", color="black")
    
    ax.set_title("ID Repetition Rate by Position and User Category")
    fig.tight_layout()
    plt.savefig('./analysis/png/repetition_rate_heatmap.png')
    plt.close()
    
    # 保存结果到CSV
    results = []
    for pos in diversity_results:
        for cat in categories:
            results.append({
                'position': pos,
                'category': cat,
                'unique_ids': diversity_results[pos][cat]['unique_ids'],
                'total_interactions': diversity_results[pos][cat]['total_interactions'],
                'repetition_rate': diversity_results[pos][cat]['repetition_rate']
            })
    
    df = pd.DataFrame(results)
    df.to_csv('./analysis/csv/diversity_analysis.csv', index=False)
    
    return diversity_results

def main():
    sequence_path = "./data/Instruments/Instruments.inter.json"
    data_path = "./data/Instruments/Instruments.index.json"
    index_path = "./data/Instruments/Instruments.index.epoch19999.alpha1e-1-beta1e-4.json"
    
    data, seq_data = load_data(sequence_path, data_path)
    categories = categorize_users(seq_data)
    
    print("Analyzing ID distribution by user category...")
    position_freqs = analyze_id_distribution(seq_data, data, categories)
    
    print("Calculating diversity metrics...")
    diversity_results = calculate_diversity(position_freqs, categories, data, seq_data)
    
    print("\nDiversity Analysis Results:")
    for pos in diversity_results:
        print(f"\nPosition {pos.upper()}:")
        for cat in categories:
            res = diversity_results[pos][cat]
            print(f"  {cat}: Repetition Rate = {res['repetition_rate']:.2%} "
                  f"(Unique IDs: {res['unique_ids']}, Interactions: {res['total_interactions']})")
    
    print("\nAnalysis complete! Results saved to:")
    print("- ./analysis/png/id_distribution_by_category.png")
    print("- ./analysis/png/repetition_rate_heatmap.png")
    print("- ./analysis/csv/diversity_analysis.csv")

if __name__ == "__main__":
    main()
