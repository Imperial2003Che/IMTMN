
import os
import shutil
from pathlib import Path
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

def reorganize_sues_200():
    """重组 SUES-200 - 这是一个配对检索数据集"""
    print("\n" + "="*80)
    print("重组 SUES-200 数据集")
    print("="*80 + "\n")
    
    base_path = './datasets/SUES-200'
    
    # 清空之前的数据
    for split in ['train', 'val', 'test']:
        split_path = os.path.join(base_path, split)
        if os.path.exists(split_path):
            shutil.rmtree(split_path)
    
    # 获取所有图片
    print("扫描原始数据...")
    drone_images = get_all_images(os.path.join(base_path, 'drone_view_512'))
    sat_images = get_all_images(os.path.join(base_path, 'satellite-view'))
    
    print(f"  找到 {len(drone_images)} 张 UAV 图片")
    print(f"  找到 {len(sat_images)} 张卫星图片")
    
    if len(drone_images) == 0 or len(sat_images) == 0:
        print("❌ 错误：没有找到图片")
        return
    
    # 排序以保证配对
    drone_images.sort()
    sat_images.sort()
    
    # 取最少的那个
    total_pairs = min(len(drone_images), len(sat_images))
    
    # 分割：train:val:test = 70:15:15
    train_size = int(total_pairs * 0.7)
    val_size = int(total_pairs * 0.15)
    test_size = total_pairs - train_size - val_size
    
    print(f"\n分割计划:")
    print(f"  train: {train_size} 对")
    print(f"  val:   {val_size} 对")
    print(f"  test:  {test_size} 对\n")
    
    # 创建并填充目录
    splits = {
        'train': (drone_images[:train_size], sat_images[:train_size]),
        'val': (drone_images[train_size:train_size+val_size], 
                sat_images[train_size:train_size+val_size]),
        'test': (drone_images[train_size+val_size:], 
                 sat_images[train_size+val_size:])
    }
    
    for split, (drone_list, sat_list) in splits.items():
        train_uav_path = os.path.join(base_path, split, 'uav')
        train_sat_path = os.path.join(base_path, split, 'sat')
        
        os.makedirs(train_uav_path, exist_ok=True)
        os.makedirs(train_sat_path, exist_ok=True)
        
        # 复制 UAV 图片
        pbar = tqdm(drone_list, desc=f"{split}/uav", leave=False)
        for src in pbar:
            dst = os.path.join(train_uav_path, os.path.basename(src))
            shutil.copy2(src, dst)
        
        # 复制卫星图片
        pbar = tqdm(sat_list, desc=f"{split}/sat", leave=False)
        for src in pbar:
            dst = os.path.join(train_sat_path, os.path.basename(src))
            shutil.copy2(src, dst)
    
    print("✓ SUES-200 重组完成！\n")

def reorganize_university_1652():
    """重组 University-1652 - 这是一个检索数据集"""
    print("\n" + "="*80)
    print("重组 University-1652 数据集")
    print("="*80 + "\n")
    
    base_path = './datasets/University-1652'
    
    # 处理 train
    train_path = os.path.join(base_path, 'train')
    if os.path.exists(train_path):
        print("【train 分割】已存在，跳过")
    
    # 处理 test - 需要重组 query/gallery 结构
    test_path = os.path.join(base_path, 'test')
    
    if os.path.exists(test_path):
        print("【test 分割】重组中...\n")
        
        # 删除旧的 uav, sat, ground 目录
        for modal in ['uav', 'sat', 'ground']:
            modal_path = os.path.join(test_path, modal)
            if os.path.exists(modal_path):
                shutil.rmtree(modal_path)
        
        # 创建新的模态目录
        uav_path = os.path.join(test_path, 'uav')
        sat_path = os.path.join(test_path, 'sat')
        ground_path = os.path.join(test_path, 'ground')
        
        os.makedirs(uav_path, exist_ok=True)
        os.makedirs(sat_path, exist_ok=True)
        os.makedirs(ground_path, exist_ok=True)
        
        # 处理 UAV 图片（query + gallery）
        query_drone = os.path.join(test_path, 'query_drone')
        gallery_drone = os.path.join(test_path, 'gallery_drone')
        
        if os.path.exists(query_drone):
            drone_files = get_all_images(query_drone)
            pbar = tqdm(drone_files, desc="复制 query_drone", leave=False)
            for src in pbar:
                dst = os.path.join(uav_path, os.path.basename(src))
                shutil.copy2(src, dst)
        
        if os.path.exists(gallery_drone):
            drone_files = get_all_images(gallery_drone)
            pbar = tqdm(drone_files, desc="复制 gallery_drone", leave=False)
            for src in pbar:
                dst = os.path.join(uav_path, os.path.basename(src))
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)
        
        # 处理卫星图片
        query_sat = os.path.join(test_path, 'query_satellite')
        gallery_sat = os.path.join(test_path, 'gallery_satellite')
        
        if os.path.exists(query_sat):
            sat_files = get_all_images(query_sat)
            pbar = tqdm(sat_files, desc="复制 query_satellite", leave=False)
            for src in pbar:
                dst = os.path.join(sat_path, os.path.basename(src))
                shutil.copy2(src, dst)
        
        if os.path.exists(gallery_sat):
            sat_files = get_all_images(gallery_sat)
            pbar = tqdm(sat_files, desc="复制 gallery_satellite", leave=False)
            for src in pbar:
                dst = os.path.join(sat_path, os.path.basename(src))
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)
        
        # 处理街景图片
        query_street = os.path.join(test_path, 'query_street')
        gallery_street = os.path.join(test_path, 'gallery_street')
        
        if os.path.exists(query_street):
            street_files = get_all_images(query_street)
            pbar = tqdm(street_files, desc="复制 query_street", leave=False)
            for src in pbar:
                dst = os.path.join(ground_path, os.path.basename(src))
                shutil.copy2(src, dst)
        
        if os.path.exists(gallery_street):
            street_files = get_all_images(gallery_street)
            pbar = tqdm(street_files, desc="复制 gallery_street", leave=False)
            for src in pbar:
                dst = os.path.join(ground_path, os.path.basename(src))
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)
        
        print("✓ test 分割重组完成！\n")
    
    print("✓ University-1652 处理完成！\n")

def verify_structure():
    """验证重组后的结构"""
    print("\n" + "="*80)
    print("验证重组结果")
    print("="*80 + "\n")
    
    datasets = ['University-1652', 'SUES-200']
    
    for dataset_name in datasets:
        dataset_path = os.path.join('./datasets', dataset_name)
        if not os.path.exists(dataset_path):
            continue
        
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
                    print(f"    ✓ 检查通过：一致性 OK ({counts['uav']} 张)")
                else:
                    print(f"    ⚠️  模态间数量不一致: {counts}")
    
    print("\n" + "="*80 + "\n")

if __name__ == '__main__':
    reorganize_sues_200()
    reorganize_university_1652()
    verify_structure()
    
    print("="*80)
    print("✓ 重组完成！现在可以运行训练了")
    print("="*80 + "\n")
