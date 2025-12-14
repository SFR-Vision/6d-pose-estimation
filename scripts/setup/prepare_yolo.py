"""
Prepare LineMOD dataset for YOLO training.
Converts bounding boxes to YOLO format and splits data into train/val/test sets.
"""

import os
import yaml
import shutil
import cv2
import sys

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

try:
    from tqdm import tqdm
except ImportError:
    print("Installing required package: tqdm")
    os.system(f"{sys.executable} -m pip install tqdm")
    from tqdm import tqdm


SOURCE_ROOT = os.path.join(PROJECT_ROOT, "datasets", "Linemod_preprocessed", "data")
DEST_ROOT = os.path.join(PROJECT_ROOT, "datasets", "yolo_ready")


def convert_bbox_to_yolo(size, box):
    """Convert absolute bounding box to YOLO format (normalized center coordinates)."""
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    x_center = box[0] + box[2] / 2.0
    y_center = box[1] + box[3] / 2.0
    return (x_center * dw, y_center * dh, box[2] * dw, box[3] * dh)

def prepare_data():
    """Main function to prepare YOLO dataset from LineMOD data."""
    print("Starting dataset preparation...")
    print(f"Source: {os.path.abspath(SOURCE_ROOT)}")

    if not os.path.exists(SOURCE_ROOT):
        print(f"Error: Source directory '{SOURCE_ROOT}' not found.")
        print("Please run this script from the project root directory.")
        return

    if os.path.exists(DEST_ROOT):
        print(f"Removing existing output directory: {DEST_ROOT}")
        shutil.rmtree(DEST_ROOT)
    
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(DEST_ROOT, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(DEST_ROOT, 'labels', split), exist_ok=True)

    obj_folders = [f for f in sorted(os.listdir(SOURCE_ROOT)) if f.isdigit()]
    
    if not obj_folders:
        print("Error: No object folders found (expected: 01, 02, ...)")
        return

    print(f"Found {len(obj_folders)} object classes: {', '.join(obj_folders)}")
    
    class_mapping = {folder: i for i, folder in enumerate(obj_folders)}
    print(f"Class mapping: {class_mapping}")
    
    stats = {'train': 0, 'val': 0, 'test': 0}

    for obj_folder in tqdm(obj_folders, desc="Processing objects"):
        base_path = os.path.join(SOURCE_ROOT, obj_folder)
        rgb_path = os.path.join(base_path, 'rgb')
        gt_path = os.path.join(base_path, 'gt.yml')
        
        if not os.path.exists(rgb_path) or not os.path.exists(gt_path):
            continue
            
        with open(gt_path, 'r') as f:
            gts = yaml.safe_load(f)
            
        images = sorted([img for img in os.listdir(rgb_path) if img.endswith(".png")])
        
        for i, img_name in enumerate(images):
            frame_id = int(img_name.split('.')[0])
            
            cycle = i % 10
            if cycle == 8:
                split_name = 'val'
            elif cycle == 9:
                split_name = 'test'
            else:
                split_name = 'train'

            if frame_id in gts:
                target_anno = None
                for anno in gts[frame_id]:
                    if str(int(anno['obj_id'])).zfill(2) == obj_folder:
                        target_anno = anno
                        break
                
                if target_anno:
                    src_img = os.path.join(rgb_path, img_name)
                    dst_img = os.path.join(DEST_ROOT, 'images', split_name, 
                                          f"{obj_folder}_{img_name}")
                    dst_label = os.path.join(DEST_ROOT, 'labels', split_name, 
                                            f"{obj_folder}_{img_name.replace('.png', '.txt')}")
                    
                    shutil.copy(src_img, dst_img)
                    
                    img_h, img_w, _ = cv2.imread(src_img).shape
                    class_id = class_mapping[obj_folder]
                    bbox = target_anno['obj_bb']
                    yolo_bbox = convert_bbox_to_yolo((img_w, img_h), bbox)
                    
                    with open(dst_label, 'w') as out_f:
                        out_f.write(f"{class_id} {yolo_bbox[0]:.6f} {yolo_bbox[1]:.6f} "
                                  f"{yolo_bbox[2]:.6f} {yolo_bbox[3]:.6f}\n")
                        
                    stats[split_name] += 1

    yaml_content = f"""path: {os.path.abspath(DEST_ROOT)} 
train: images/train
val: images/val
test: images/test

nc: {len(obj_folders)}
names: {obj_folders}
"""
    
    with open(os.path.join(DEST_ROOT, "dataset.yaml"), "w") as f:
        f.write(yaml_content)

    print("\nDataset preparation complete!")
    print(f"Output directory: {DEST_ROOT}")
    print(f"  Training samples:   {stats['train']}")
    print(f"  Validation samples: {stats['val']}")
    print(f"  Test samples:       {stats['test']}")

if __name__ == "__main__":
    prepare_data()