
import os

def check_dataset_structure(dataset_dir='./datasets'):
    """检查数据集结构，支持可选的模态"""
    
    datasets = {}
    
    # 扫描数据集文件夹
    if not os.path.exists(dataset_dir):
        print(f"❌ 数据集目录不存在: {dataset_dir}")
        return
    
    dataset_names = [d for d in os.listdir(dataset_dir) 
                     if os.path.isdir(os.path.join(dataset_dir, d)) and not d.startswith('.')]
    
    print("\n" + "="*80)
    print("数据集检查报告")
    print("="*80 + "\n")
    
    all_consistent = True
    
    for dataset_name in dataset_names:
        print(f"【{dataset_name}】")
        print("-" * 80)
        
        dataset_path = os.path.join(dataset_dir, dataset_name)
        
        # 检查 split 目录
        splits = [d for d in os.listdir(dataset_path) 
                 if os.path.isdir(os.path.join(dataset_path, d)) and not d.startswith('.')]
        splits = sorted(splits)
        
        if not splits:
            print(f"  ❌ 没有找到 split 目录")
            all_consistent = False
            continue
        
        for split in splits:
            split_path = os.path.join(dataset_path, split)
            print(f"\n  [{split}] 分割:")
            
            # 检测可用的模态
            possible_modalities = ['uav', 'sat', 'ground']
            available_modalities = {}
            
            for modality in possible_modalities:
                mod_path = os.path.join(split_path, modality)
                if os.path.exists(mod_path):
                    image_files = [f for f in os.listdir(mod_path) 
                                  if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp'))]
                    if image_files:
                        available_modalities[modality] = len(image_files)
                        print(f"    ✓ {modality:10s}: {len(image_files):6d} 张图片")
            
            if not available_modalities:
                print(f"    ❌ 该分割没有任何模态的图片")
                all_consistent = False
                continue
            
            # 检查模态之间的一致性
            counts = list(available_modalities.values())
            if len(set(counts)) > 1:
                print(f"    ⚠️  警告：模态之间图片数量不一致！")
                print(f"       {available_modalities}")
                all_consistent = False
            else:
                print(f"    ✓ 所有模态图片数量一致 ({counts[0]} 张)")
    
    # 总结
    print("\n" + "="*80)
    if all_consistent:
        print("✓ 检查通过：所有数据集结构正确！")
    else:
        print("⚠️  检查完成：请注意上面的警告信息。")
    print("="*80 + "\n")
    
    return all_consistent

if __name__ == '__main__':
    check_dataset_structure('./datasets')
