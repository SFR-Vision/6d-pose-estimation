"""Per-object ADD metrics comparison for all 3 pose estimation models."""

import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm
import tkinter as tk
from tkinter import ttk

from data.dataset_rgb import LineMODDatasetRGB
from data.dataset_rgbd import LineMODDatasetRGBD
from models.add_loss import ADDLoss

# Configuration
DATA_ROOT = os.path.join(PROJECT_ROOT, "datasets", "Linemod_preprocessed", "data")
MODEL_DIR = os.path.join(PROJECT_ROOT, "datasets", "Linemod_preprocessed", "models")
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Object names mapping (obj_id = folder_number - 1)
OBJ_NAMES = {
    0: "01-ape",
    1: "02-benchvise",
    3: "04-camera",
    4: "05-can",
    5: "06-cat",
    7: "08-driller",
    8: "09-duck",
    9: "10-eggbox",
    10: "11-glue",
    11: "12-holepuncher",
    12: "13-iron",
    13: "14-lamp",
    14: "15-phone"
}

# Symmetric objects (eggbox and glue)
SYMMETRIC_OBJECTS = {9, 10}  # 10-eggbox, 11-glue

WEIGHTS = {
    'RGB': os.path.join(PROJECT_ROOT, "weights_rgb", "best_pose_model.pth"),
    'RGBD': os.path.join(PROJECT_ROOT, "weights_rgbd", "best_pose_model.pth"),
}


def load_pose_model(model_name, weights_path):
    """Load a pose estimation model."""
    if not os.path.exists(weights_path):
        return None
    
    try:
        if model_name == 'RGB':
            from models.pose_net_rgb import PoseNetRGB
            model = PoseNetRGB(pretrained=False)
        elif model_name == 'RGBD':
            from models.pose_net_rgbd import PoseNetRGBD
            model = PoseNetRGBD(pretrained=False)
        else:
            return None
        
        checkpoint = torch.load(weights_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(DEVICE).eval()
        return model
    except Exception as e:
        print(f"  Error loading {model_name}: {e}")
        return None


def evaluate_per_object(model, model_name, dataset, add_loss, is_rgbd=False, needs_geometry=False, is_4ch=False, is_5ch=False):
    """Evaluate model and return per-object ADD metrics using ADDLoss class."""
    if model is None:
        return None
    
    # Per-object metrics storage
    obj_add_sums = {}
    obj_counts = {}
    obj_acc_2cm = {}
    
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(len(dataset)), desc=f"Evaluating {model_name}", leave=False):
            sample = dataset[i]
            
            if is_rgbd:
                # RGBD dataset returns: rgbdm (5ch), z_sensor, gt_rot, gt_trans, obj_id, bbox_center, cam_matrix
                rgbdm, z_sensor, gt_rot, gt_trans, obj_id, bbox_center, cam_matrix = sample
                rgbdm = rgbdm.unsqueeze(0).to(DEVICE)
                z_sensor = z_sensor.unsqueeze(0).to(DEVICE)
                bbox_center = bbox_center.unsqueeze(0).to(DEVICE)
                cam_matrix = cam_matrix.unsqueeze(0).to(DEVICE)
                
                if needs_geometry:
                    pred_rot, pred_trans = model(rgbdm, z_sensor, bbox_center, cam_matrix)
            else:
                # RGB: (rgbm_4ch, gt_rot, gt_trans, obj_id, bbox_center, cam_matrix)
                rgbm, gt_rot, gt_trans, obj_id, bbox_center, cam_matrix = sample
                rgbm = rgbm.unsqueeze(0).to(DEVICE)
                bbox_center = bbox_center.unsqueeze(0).to(DEVICE)
                cam_matrix = cam_matrix.unsqueeze(0).to(DEVICE)
                pred_rot, pred_trans = model(rgbm, bbox_center, cam_matrix)
            
            obj_id_val = int(obj_id.item()) if hasattr(obj_id, 'item') else int(obj_id)
            gt_rot = gt_rot.unsqueeze(0).to(DEVICE)
            gt_trans = gt_trans.unsqueeze(0).to(DEVICE)
            obj_ids_tensor = torch.tensor([obj_id_val], device=DEVICE)
            
            # Use ADDLoss class for evaluation
            metrics = add_loss.eval_metrics(pred_rot, pred_trans, gt_rot, gt_trans, obj_ids_tensor)
            add_dist = metrics['add_mean']  # Already in mm
            acc_2cm = metrics['add_2cm_acc']  # Accuracy %
            
            if obj_id_val not in obj_add_sums:
                obj_add_sums[obj_id_val] = 0.0
                obj_counts[obj_id_val] = 0
                obj_acc_2cm[obj_id_val] = 0.0
            
            obj_add_sums[obj_id_val] += add_dist
            obj_counts[obj_id_val] += 1
            obj_acc_2cm[obj_id_val] += acc_2cm
    
    # Compute averages
    results = {}
    for obj_id in obj_add_sums:
        results[obj_id] = {
            'add_mm': obj_add_sums[obj_id] / obj_counts[obj_id],
            'acc_2cm': obj_acc_2cm[obj_id] / obj_counts[obj_id],
            'count': obj_counts[obj_id]
        }
    
    return results


def main():
    print(f"\nPer-Object ADD Metrics Comparison on {DEVICE}\n")
    
    # Create ADDLoss evaluator
    print("Loading 3D models...")
    add_loss = ADDLoss(MODEL_DIR, DEVICE)
    print(f"  Loaded models\n")
    
    # Transforms (datasets handle ToTensor internally)
    val_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    # Load datasets
    print("Loading datasets...")
    rgb_dataset = LineMODDatasetRGB(DATA_ROOT, mode='test', transform=val_transform)
    
    try:
        rgbd_dataset = LineMODDatasetRGBD(DATA_ROOT, mode='test', transform=val_transform)
    except:
        rgbd_dataset = None
        print("  RGBD dataset not available")
    
    print(f"  {len(rgb_dataset)} test samples\n")
    
    # Load models
    print("Loading models...")
    models = {}
    for name, path in WEIGHTS.items():
        models[name] = load_pose_model(name, path)
        status = "✓" if models[name] is not None else "✗"
        print(f"  {name}: {status}")
    print()
    
    # Evaluate each model
    all_results = {}
    
    if models['RGB'] is not None:
        all_results['RGB'] = evaluate_per_object(
            models['RGB'], 'RGB', rgb_dataset, add_loss,
            is_rgbd=False, needs_geometry=True, is_4ch=True
        )
    
    if rgbd_dataset is not None:
        if models['RGBD'] is not None:
            all_results['RGBD'] = evaluate_per_object(
                models['RGBD'], 'RGBD', rgbd_dataset, add_loss,
                is_rgbd=True, needs_geometry=True, is_5ch=True
            )
    
    # Print results table - ADD (mm)
    print("\n" + "=" * 80)
    print("Per-Object ADD Metrics (mm) - Lower is better")
    print("=" * 80)
    
    model_names = [name for name in ['RGB', 'RGBD'] if name in all_results]
    
    header = f"{'Object':<18}"
    for name in model_names:
        header += f"{name:<12}"
    print(header)
    print("-" * 80)
    
    # Get all object IDs
    all_obj_ids = set()
    for results in all_results.values():
        if results:
            all_obj_ids.update(results.keys())
    
    model_avgs = {name: [] for name in model_names}
    
    for obj_id in sorted(all_obj_ids):
        obj_name = OBJ_NAMES.get(obj_id, f"obj_{obj_id}")
        sym_marker = "*" if obj_id in SYMMETRIC_OBJECTS else ""
        row = f"{obj_name}{sym_marker:<18}"
        
        for model_name in model_names:
            if model_name in all_results and all_results[model_name] and obj_id in all_results[model_name]:
                add_val = all_results[model_name][obj_id]['add_mm']
                row += f"{add_val:<12.1f}"
                model_avgs[model_name].append(add_val)
            else:
                row += f"{'N/A':<12}"
        
        print(row)
    
    print("-" * 80)
    
    # Averages
    avg_row = f"{'AVERAGE':<18}"
    for model_name in model_names:
        if model_avgs[model_name]:
            avg_row += f"{np.mean(model_avgs[model_name]):<12.1f}"
        else:
            avg_row += f"{'N/A':<12}"
    print(avg_row)
    
    # Print ADD-2cm accuracy table
    print("\n" + "=" * 80)
    print("Per-Object ADD-2cm Accuracy (%) - Higher is better")
    print("=" * 80)
    print(header)
    print("-" * 80)
    
    model_acc_avgs = {name: [] for name in model_names}
    
    for obj_id in sorted(all_obj_ids):
        obj_name = OBJ_NAMES.get(obj_id, f"obj_{obj_id}")
        sym_marker = "*" if obj_id in SYMMETRIC_OBJECTS else ""
        row = f"{obj_name}{sym_marker:<18}"
        
        for model_name in model_names:
            if model_name in all_results and all_results[model_name] and obj_id in all_results[model_name]:
                acc_val = all_results[model_name][obj_id]['acc_2cm']
                row += f"{acc_val:<12.1f}"
                model_acc_avgs[model_name].append(acc_val)
            else:
                row += f"{'N/A':<12}"
        
        print(row)
    
    print("-" * 80)
    
    # Averages
    avg_row = f"{'AVERAGE':<18}"
    for model_name in model_names:
        if model_acc_avgs[model_name]:
            avg_row += f"{np.mean(model_acc_avgs[model_name]):<12.1f}"
        else:
            avg_row += f"{'N/A':<12}"
    print(avg_row)
    
    print("\n* = Symmetric object (uses ADD-S metric)")
    print("=" * 80)
    
    # Show Excel-like GUI table (skip in headless environments like Colab)
    try:
        show_excel_table(all_results, model_names, all_obj_ids)
    except Exception as e:
        # GUI not available (headless/Colab environment)
        pass


def show_excel_table(all_results, model_names, all_obj_ids):
    """Display results in an Excel-like GUI table."""
    
    # Create main window
    root = tk.Tk()
    root.title("6D Pose Estimation - Per-Object Metrics Comparison")
    root.geometry("1200x800")
    
    # Create notebook for tabs
    notebook = ttk.Notebook(root)
    notebook.pack(fill='both', expand=True, padx=10, pady=10)
    
    # Tab 1: ADD Metrics (mm)
    frame_add = ttk.Frame(notebook)
    notebook.add(frame_add, text='ADD Metrics (mm)')
    
    # Create Treeview for ADD
    tree_add = ttk.Treeview(frame_add, columns=['Object'] + model_names, show='headings', height=20)
    tree_add.pack(side='left', fill='both', expand=True)
    
    # Scrollbars
    vsb_add = ttk.Scrollbar(frame_add, orient="vertical", command=tree_add.yview)
    vsb_add.pack(side='right', fill='y')
    tree_add.configure(yscrollcommand=vsb_add.set)
    
    # Define columns
    tree_add.heading('Object', text='Object')
    tree_add.column('Object', width=200, anchor='w')
    
    for model_name in model_names:
        tree_add.heading(model_name, text=model_name)
        tree_add.column(model_name, width=150, anchor='center')
    
    # Add data rows for ADD
    model_avgs = {name: [] for name in model_names}
    
    for obj_id in sorted(all_obj_ids):
        obj_name = OBJ_NAMES.get(obj_id, f"obj_{obj_id}")
        sym_marker = " *" if obj_id in SYMMETRIC_OBJECTS else ""
        
        values = [obj_name + sym_marker]
        for model_name in model_names:
            if model_name in all_results and all_results[model_name] and obj_id in all_results[model_name]:
                add_val = all_results[model_name][obj_id]['add_mm']
                values.append(f"{add_val:.1f}")
                model_avgs[model_name].append(add_val)
            else:
                values.append("N/A")
        
        tree_add.insert('', 'end', values=values)
    
    # Add average row
    avg_values = ["AVERAGE"]
    for model_name in model_names:
        if model_avgs[model_name]:
            avg_values.append(f"{np.mean(model_avgs[model_name]):.1f}")
        else:
            avg_values.append("N/A")
    tree_add.insert('', 'end', values=avg_values, tags=('average',))
    tree_add.tag_configure('average', background='#E8F4F8', font=('TkDefaultFont', 9, 'bold'))
    
    # Tab 2: ADD-2cm Accuracy (%)
    frame_acc = ttk.Frame(notebook)
    notebook.add(frame_acc, text='ADD-2cm Accuracy (%)')
    
    # Create Treeview for Accuracy
    tree_acc = ttk.Treeview(frame_acc, columns=['Object'] + model_names, show='headings', height=20)
    tree_acc.pack(side='left', fill='both', expand=True)
    
    # Scrollbars
    vsb_acc = ttk.Scrollbar(frame_acc, orient="vertical", command=tree_acc.yview)
    vsb_acc.pack(side='right', fill='y')
    tree_acc.configure(yscrollcommand=vsb_acc.set)
    
    # Define columns
    tree_acc.heading('Object', text='Object')
    tree_acc.column('Object', width=200, anchor='w')
    
    for model_name in model_names:
        tree_acc.heading(model_name, text=model_name)
        tree_acc.column(model_name, width=150, anchor='center')
    
    # Add data rows for Accuracy
    model_acc_avgs = {name: [] for name in model_names}
    
    for obj_id in sorted(all_obj_ids):
        obj_name = OBJ_NAMES.get(obj_id, f"obj_{obj_id}")
        sym_marker = " *" if obj_id in SYMMETRIC_OBJECTS else ""
        
        values = [obj_name + sym_marker]
        for model_name in model_names:
            if model_name in all_results and all_results[model_name] and obj_id in all_results[model_name]:
                acc_val = all_results[model_name][obj_id]['acc_2cm']
                values.append(f"{acc_val:.1f}")
                model_acc_avgs[model_name].append(acc_val)
            else:
                values.append("N/A")
        
        tree_acc.insert('', 'end', values=values)
    
    # Add average row
    avg_values = ["AVERAGE"]
    for model_name in model_names:
        if model_acc_avgs[model_name]:
            avg_values.append(f"{np.mean(model_acc_avgs[model_name]):.1f}")
        else:
            avg_values.append("N/A")
    tree_acc.insert('', 'end', values=avg_values, tags=('average',))
    tree_acc.tag_configure('average', background='#E8F4F8', font=('TkDefaultFont', 9, 'bold'))
    
    # Add note at bottom
    note_frame = ttk.Frame(root)
    note_frame.pack(fill='x', padx=10, pady=5)
    
    note_label = ttk.Label(note_frame, text="* = Symmetric object (uses ADD-S metric) | Lower ADD is better | Higher Accuracy is better", 
                          font=('TkDefaultFont', 9, 'italic'))
    note_label.pack()
    
    # Add export button
    button_frame = ttk.Frame(root)
    button_frame.pack(fill='x', padx=10, pady=5)
    
    def export_to_csv():
        """Export data to CSV files."""
        try:
            # Export ADD metrics
            with open('per_object_ADD_metrics.csv', 'w') as f:
                # Header
                f.write('Object,' + ','.join(model_names) + '\n')
                
                # Data rows
                for obj_id in sorted(all_obj_ids):
                    obj_name = OBJ_NAMES.get(obj_id, f"obj_{obj_id}")
                    row = [obj_name]
                    for model_name in model_names:
                        if model_name in all_results and all_results[model_name] and obj_id in all_results[model_name]:
                            row.append(f"{all_results[model_name][obj_id]['add_mm']:.1f}")
                        else:
                            row.append('N/A')
                    f.write(','.join(row) + '\n')
                
                # Average row
                avg_row = ['AVERAGE']
                for model_name in model_names:
                    vals = [all_results[model_name][obj_id]['add_mm'] 
                           for obj_id in sorted(all_obj_ids) 
                           if model_name in all_results and all_results[model_name] and obj_id in all_results[model_name]]
                    if vals:
                        avg_row.append(f"{np.mean(vals):.1f}")
                    else:
                        avg_row.append('N/A')
                f.write(','.join(avg_row) + '\n')
            
            # Export accuracy metrics
            with open('per_object_accuracy_metrics.csv', 'w') as f:
                # Header
                f.write('Object,' + ','.join(model_names) + '\n')
                
                # Data rows
                for obj_id in sorted(all_obj_ids):
                    obj_name = OBJ_NAMES.get(obj_id, f"obj_{obj_id}")
                    row = [obj_name]
                    for model_name in model_names:
                        if model_name in all_results and all_results[model_name] and obj_id in all_results[model_name]:
                            row.append(f"{all_results[model_name][obj_id]['acc_2cm']:.1f}")
                        else:
                            row.append('N/A')
                    f.write(','.join(row) + '\n')
                
                # Average row
                avg_row = ['AVERAGE']
                for model_name in model_names:
                    vals = [all_results[model_name][obj_id]['acc_2cm'] 
                           for obj_id in sorted(all_obj_ids) 
                           if model_name in all_results and all_results[model_name] and obj_id in all_results[model_name]]
                    if vals:
                        avg_row.append(f"{np.mean(vals):.1f}")
                    else:
                        avg_row.append('N/A')
                f.write(','.join(avg_row) + '\n')
            
            export_label.config(text="✓ Exported to per_object_ADD_metrics.csv and per_object_accuracy_metrics.csv", foreground='green')
        except Exception as e:
            export_label.config(text=f"✗ Export failed: {e}", foreground='red')
    
    export_btn = ttk.Button(button_frame, text="Export to CSV", command=export_to_csv)
    export_btn.pack(side='left', padx=5)
    
    export_label = ttk.Label(button_frame, text="")
    export_label.pack(side='left', padx=10)
    
    # Start GUI
    root.mainloop()


if __name__ == '__main__':
    main()
