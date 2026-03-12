
import os
import shutil
from tqdm import tqdm

VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff'}

def get_all_images(directory):
    """递归获取目录中的所有图片"""
    images = []
    if not os.path.exists(directory):
        return images
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if os.path.splitext(file)[1].lower() in VALID_EXTENSIONS:
                images.append(os.path.join(root, file))
    return images

def cleanup_and_reorganize():
    """彻底清理和重新组织"""
    
    print("\n" + "="*80)
    print("数据集清理和重新组织")
    print("="*80 + "\n")
    
    base_path = './datasets'
    
    # ============ SUES-200 处理 ============
    print("【处理 SUES-200】\n")
    sues_base = os.path.join(base_path, 'SUES-200')
    
    # 备份原始数据
    drone_original = os.path.join(sues_base, 'drone_view_512')
    sat_original = os.path.join(sues_base, 'satellite-view')
    
    drone_backup = os.path.join(sues_base, '_backup_drone')
    sat_backup = os.path.join(sues_base, '_backup_sat')
    
    # 移动原始数据到备份
    if os.path.exists(drone_original) and not os.path.exists(drone_backup):
        print("✓ 备份 drone_view_512...")
        shutil.move(drone_original, drone_backup)
    
    if os.path.exists(sat_original) and not os.path.exists(sat_backup):
        print("✓ 备份 satellite-view...")
        shutil.move(sat_original, sat_backup)
    
    # 删除旧的 train/val/test 目录
    for split in ['train', 'val', 'test', 'drone_view_512', 'satellite-view']:
        split_path = os.path.join(sues_base, split)
        if os.path.exists(split_path):
            print(f"✓ 删除旧目录: {split}...")
            shutil.rmtree(split_path)
    
    # 获取备份中的所有图片
    print("\n从备份中获取图片...")
    drone_images = get_all_images(drone_backup)
    sat_images = get_all_images(sat_backup)
    
    print(f"  找到 {len(drone_images)} 张 UAV 图片")
    print(f"  找到 {len(sat_images)} 张卫星图片")
    
    # 分割比例：train:val:test = 70:15:15
    total_pairs = min(len(drone_images), len(sat_images))
    train_size = int(total_pairs * 0.7)
    val_size = int(total_pairs * 0.15)
    test_size = total_pairs - train_size - val_size
    
    print(f"\n分割计划:")
    print(f"  train: {train_size} 对")
    print(f"  val:   {val_size} 对")
    print(f"  test:  {test_size} 对\n")
    
    # 创建新的目录结构
    splits_data = {
        'train': (drone_images[:train_size], sat_images[:train_size]),
        'val': (drone_images[train_size:train_size+val_size], 
                sat_images[train_size:train_size+val_size]),
        'test': (drone_images[train_size+val_size:], 
                 sat_images[train_size+val_size:])
    }
    
    for split, (drone_list, sat_list) in splits_data.items():
        split_path = os.path.join(sues_base, split)
        os.makedirs(os.path.join(split_path, 'uav'), exist_ok=True)
        os.makedirs(os.path.join(split_path, 'sat'), exist_ok=True)
        
        print(f"复制 {split}...")
        
        # 复制 UAV 图片
        pbar = tqdm(drone_list, desc=f"  {split}/uav", leave=False)
        for src in pbar:
            dst = os.path.join(split_path, 'uav', os.path.basename(src))
            shutil.copy2(src, dst)
        
        # 复制卫星图片
        pbar = tqdm(sat_list, desc=f"  {split}/sat", leave=False)
        for src in pbar:
            dst = os.path.join(split_path, 'sat', os.path.basename(src))
            shutil.copy2(src, dst)
    
    print("\n✓ SUES-200 重组完成！\n")
    
    # ============ University-1652 处理 ============
    print("【处理 University-1652】\n")
    uni_base = os.path.join(base_path, 'University-1652')
    
    # 处理 test 目录
    test_path = os.path.join(uni_base, 'test')
    if os.path.exists(test_path):
        print("✓ 检查 test 目录...")
        
        # 重命名原始目录
        for old_name, new_name in [('drone', 'uav'), ('satellite', 'sat'), ('street', 'ground'), ('google', 'ground')]:
            old_dir = os.path.join(test_path, old_name)
            new_dir = os.path.join(test_path, new_name)
            
            if os.path.exists(old_dir) and not os.path.exists(new_dir):
                print(f"  ✓ 重命名: {old_name} -> {new_name}")
                shutil.move(old_dir, new_dir)
            elif os.path.exists(old_dir) and os.path.exists(new_dir) and old_name == 'google':
                # 如果 ground 已存在，合并 google 目录
                print(f"  ✓ 合并: google -> ground")
                google_files = get_all_images(old_dir)
                pbar = tqdm(google_files, desc="  合并 google", leave=False)
                for src in pbar:
                    dst = os.path.join(new_dir, os.path.basename(src))
                    if not os.path.exists(dst):
                        shutil.copy2(src, dst)
                shutil.rmtree(old_dir)
    
    print("\n✓ University-1652 处理完成！\n")
    
    # ============ 验证 ============
    print("="*80)
    print("验证重组结果")
    print("="*80 + "\n")
    
    for dataset_name in ['University-1652', 'SUES-200']:
        dataset_path = os.path.join(base_path, dataset_name)
        print(f"【{dataset_name}】")
        
        splits = sorted([d for d in os.listdir(dataset_path) 
                        if os.path.isdir(os.path.join(dataset_path, d)) and d in ['train', 'val', 'test']])
        
        for split in splits:
            split_path = os.path.join(dataset_path, split)
            print(f"  [{split}]")
            
            modalities = ['uav', 'sat', 'ground']
            counts = {}
            
            for mod in modalities:
                mod_path = os.path.join(split_path, mod)
                if os.path.exists(mod_path):
                    files = [f for f in os.listdir(mod_path) 
                            if os.path.splitext(f)[1].lower() in VALID_EXTENSIONS]
                    if files:
                        counts[mod] = len(files)
                        print(f"    ✓ {mod:10s}: {len(files):6d} 张")
            
            if counts:
                if len(set(counts.values())) == 1:
                    print(f"    ✓ 检查通过：一致性 OK")
                else:
                    print(f"    ⚠️  警告：{counts}")
    
    print("\n" + "="*80)
    print("✓ 清理和重组完成！")
    print("="*80 + "\n")

if __name__ == '__main__':
    cleanup_and_reorganize()
