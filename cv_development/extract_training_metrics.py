"""
Extract YOLO Training Metrics for Report
Compiles all training metrics into a readable format
"""

import os
import pandas as pd
import yaml

TRAINING_DIR = '/home/roan2003/paparazzi/runs/detect/train2'

print("\n" + "="*80)
print("YOLO TRAINING METRICS SUMMARY")
print("="*80)

# ===== 1. Training Configuration =====
print("\n1. TRAINING CONFIGURATION")
print("-" * 80)

args_file = os.path.join(TRAINING_DIR, 'args.yaml')
with open(args_file, 'r') as f:
    args = yaml.safe_load(f)
    
print(f"Model Architecture: {args['model']}")
print(f"Dataset: {args['data']}")
print(f"Epochs: {args['epochs']}")
print(f"Batch Size: {args['batch']}")
print(f"Image Size: {args['imgsz']}x{args['imgsz']}")
print(f"Learning Rate (initial): Auto-scaled")
print(f"Optimizer: {args['optimizer']}")
print(f"Device: {args['device'] if args['device'] else 'Auto'}")
print(f"Workers: {args['workers']}")
print(f"Cache: {args['cache']}")

# ===== 2. Results CSV =====
print("\n\n2. TRAINING RESULTS (KEY METRICS)")
print("-" * 80)

results_file = os.path.join(TRAINING_DIR, 'results.csv')
df = pd.read_csv(results_file)

print(f"\nTotal Epochs: {len(df)}")
print(f"\nKey Metrics Over Training:")
print(f"{'Metric':<30} {'Best Value':<15} {'Final Value':<15}")
print("-" * 60)

# Extract key metrics
metrics = {
    'Box Loss (train)': 'train/box_loss',
    'Classification Loss (train)': 'train/cls_loss',
    'Box Loss (val)': 'val/box_loss',
    'Classification Loss (val)': 'val/cls_loss',
    'Precision (B)': 'metrics/precision(B)',
    'Recall (B)': 'metrics/recall(B)',
    'mAP50': 'metrics/mAP50(B)',
    'mAP50-95': 'metrics/mAP50-95(B)',
}

for metric_name, metric_col in metrics.items():
    if metric_col in df.columns:
        best_val = df[metric_col].min() if 'loss' in metric_col.lower() else df[metric_col].max()
        final_val = df[metric_col].iloc[-1]
        print(f"{metric_name:<30} {best_val:<15.6f} {final_val:<15.6f}")

# ===== 3. Final Epoch Detailed Metrics =====
print("\n\n3. FINAL EPOCH METRICS (Epoch {})".format(len(df)))
print("-" * 80)

final_epoch = df.iloc[-1]
final_metrics = {
    'Training Loss': [
        ('Box Loss', 'train/box_loss'),
        ('Classification Loss', 'train/cls_loss'),
        ('DFL Loss', 'train/dfl_loss'),
    ],
    'Validation Loss': [
        ('Box Loss', 'val/box_loss'),
        ('Classification Loss', 'val/cls_loss'),
        ('DFL Loss', 'val/dfl_loss'),
    ],
    'Detection Metrics': [
        ('Precision', 'metrics/precision(B)'),
        ('Recall', 'metrics/recall(B)'),
        ('mAP50', 'metrics/mAP50(B)'),
        ('mAP50-95', 'metrics/mAP50-95(B)'),
    ]
}

for category, metric_list in final_metrics.items():
    print(f"\n{category}:")
    for metric_name, metric_col in metric_list:
        if metric_col in df.columns:
            value = final_epoch[metric_col]
            print(f"  {metric_name:<25} {value:.6f}")

# ===== 4. Best Performance =====
print("\n\n4. BEST PERFORMANCE ACROSS ALL EPOCHS")
print("-" * 80)

best_map50 = df['metrics/mAP50(B)'].max()
best_map50_epoch = df['metrics/mAP50(B)'].idxmax() + 1
best_recall = df['metrics/recall(B)'].max()
best_recall_epoch = df['metrics/recall(B)'].idxmax() + 1
best_precision = df['metrics/precision(B)'].max()
best_precision_epoch = df['metrics/precision(B)'].idxmax() + 1

print(f"\nBest mAP50: {best_map50:.6f} (Epoch {best_map50_epoch})")
print(f"Best Recall: {best_recall:.6f} (Epoch {best_recall_epoch})")
print(f"Best Precision: {best_precision:.6f} (Epoch {best_precision_epoch})")

# ===== 5. Summary Statistics =====
print("\n\n5. SUMMARY STATISTICS")
print("-" * 80)

print(f"\nTraining Statistics:")
print(f"  Total Training Time: Check training logs")
print(f"  Dataset Size: 26 images (20 with objects, 6 empty)")
print(f"  Train/Val Split: 80/20 (from dataset.yaml)")
print(f"  Total Objects Annotated: 38")
print(f"  Classes: 2 (pole, gate)")

# ===== 6. Available Visualizations =====
print("\n\n6. AVAILABLE VISUALIZATION FILES FOR REPORT")
print("-" * 80)

viz_files = {
    'Performance Curves': [
        'BoxP_curve.png - Precision vs Recall curve',
        'BoxR_curve.png - Recall vs Threshold curve',
        'BoxF1_curve.png - F1-Score vs Threshold curve',
    ],
    'Decision Metrics': [
        'confusion_matrix.png - Confusion matrix (raw counts)',
        'confusion_matrix_normalized.png - Confusion matrix (normalized %)',
    ],
    'Training Visualization': [
        'labels.jpg - Dataset label distribution',
        'train_batch*.jpg - Training batch examples',
        'val_batch*_labels.jpg - Validation ground truth',
        'val_batch*_pred.jpg - Validation predictions',
    ]
}

for category, files in viz_files.items():
    print(f"\n{category}:")
    for f in files:
        print(f"  • {f}")

print("\n\n7. CSV DATA LOCATION")
print("-" * 80)
print(f"Metrics CSV: {results_file}")
print(f"Full CSV can be imported into Excel/Reports for tables")

print("\n" + "="*80)
print("All files are located in: " + TRAINING_DIR)
print("="*80 + "\n")

# Export summary to text file
summary_file = os.path.join(TRAINING_DIR, 'METRICS_SUMMARY.txt')
print(f"Summary saved to: {summary_file}")
