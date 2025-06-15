import numpy as np
import os
import json
from pathlib import Path

def analyze_seed_results(seed_dir):
    """分析单个种子的训练结果"""
    try:
        # 加载评估结果
        eval_file = Path(seed_dir) / "evaluations.npz"
        if not eval_file.exists():
            print(f"No evaluation file found in {seed_dir}")
            return None
            
        data = np.load(eval_file)
        results = data['results']
        
        # 获取最后一次评估的结果
        final_results = results[-1]
        
        # 计算统计信息
        mean_reward = final_results.mean()
        std_reward = final_results.std()
        min_reward = final_results.min()
        max_reward = final_results.max()
        above_200 = np.sum(final_results > 200)
        total_episodes = len(final_results)
        
        # 加载训练配置
        meta_file = Path(seed_dir) / "meta.json"
        if meta_file.exists():
            with open(meta_file) as f:
                config = json.load(f)
        else:
            config = None
            
        return {
            'mean_reward': mean_reward,
            'std_reward': std_reward,
            'min_reward': min_reward,
            'max_reward': max_reward,
            'above_200': above_200,
            'total_episodes': total_episodes,
            'config': config
        }
    except Exception as e:
        print(f"Error analyzing {seed_dir}: {str(e)}")
        return None

def main():
    base_dir = "/var/scratch/fhu100/rlzoo-logs/lunar_dqn_improved"
    seeds = [101, 307, 911, 1747, 2029, 2861, 3253, 4099, 7919, 9011]
    
    print("Analyzing improved training results:")
    print("-" * 50)
    
    all_results = []
    for seed in seeds:
        seed_dir = os.path.join(base_dir, f"seed_{seed}")
        results = analyze_seed_results(seed_dir)
        if results:
            print(f"\nSeed {seed}:")
            print(f"  Mean reward: {results['mean_reward']:.2f}")
            print(f"  Std reward: {results['std_reward']:.2f}")
            print(f"  Min reward: {results['min_reward']:.2f}")
            print(f"  Max reward: {results['max_reward']:.2f}")
            print(f"  Episodes above 200: {results['above_200']}/{results['total_episodes']}")
            print(f"  Percentage above 200: {(results['above_200']/results['total_episodes'])*100:.1f}%")
            all_results.append(results)
    
    if all_results:
        # 计算总体统计信息
        mean_rewards = [r['mean_reward'] for r in all_results]
        above_200_counts = [r['above_200'] for r in all_results]
        total_episodes = sum(r['total_episodes'] for r in all_results)
        
        print("\nOverall statistics:")
        print(f"Total seeds: {len(all_results)}")
        print(f"Mean reward across seeds: {np.mean(mean_rewards):.2f} ± {np.std(mean_rewards):.2f}")
        print(f"Seeds with mean reward > 200: {np.sum(np.array(mean_rewards) > 200)}/{len(mean_rewards)}")
        print(f"Total episodes above 200: {sum(above_200_counts)}/{total_episodes}")
        print(f"Percentage of episodes above 200: {(sum(above_200_counts)/total_episodes)*100:.1f}%")
        
        # 比较新旧配置
        print("\nComparing with old configuration:")
        print("Old config (from previous analysis):")
        print("- Mean reward: 171.54")
        print("- Std reward: 132.16")
        print("- Episodes above 200: 30/50 (60.0%)")
        print("- Seeds with mean reward > 200: 4/10")
        
        print("\nNew config:")
        print(f"- Mean reward: {np.mean(mean_rewards):.2f}")
        print(f"- Std reward: {np.std(mean_rewards):.2f}")
        print(f"- Episodes above 200: {sum(above_200_counts)}/{total_episodes} ({(sum(above_200_counts)/total_episodes)*100:.1f}%)")
        print(f"- Seeds with mean reward > 200: {np.sum(np.array(mean_rewards) > 200)}/{len(mean_rewards)}")

if __name__ == "__main__":
    main() 