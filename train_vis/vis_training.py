import pandas as pd

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# Read the CSV file into a DataFrame
df = pd.read_csv('../runs/pose/uvdoc_yolo11s_midstrong/results.csv')

import matplotlib.pyplot as plt
import seaborn as sns

# Clean column names
df.columns = df.columns.str.strip()

# Set style
sns.set(style="whitegrid")

# Define epochs
epochs = df['epoch']

# 1. Losses Plot
fig1, axes1 = plt.subplots(2, 3, figsize=(18, 10))
fig1.suptitle('Training and Validation Losses', fontsize=16)

loss_pairs = [
    ('Box Loss', 'train/box_loss', 'val/box_loss'),
    ('Pose Loss', 'train/pose_loss', 'val/pose_loss'),
    ('Kobj Loss', 'train/kobj_loss', 'val/kobj_loss'),
    ('Cls Loss', 'train/cls_loss', 'val/cls_loss'),
    ('Dfl Loss', 'train/dfl_loss', 'val/dfl_loss')
]

axes1 = axes1.flatten()
for i, (title, train_col, val_col) in enumerate(loss_pairs):
    ax = axes1[i]
    if train_col in df.columns and val_col in df.columns:
        ax.plot(epochs, df[train_col], label='Train', marker='o')
        ax.plot(epochs, df[val_col], label='Val', marker='o')
        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'Data not found', ha='center', va='center')

# Hide the empty 6th subplot
axes1[-1].axis('off')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('losses_plot.png')

# 2. Metrics (Bounding Box) Plot
fig2, axes2 = plt.subplots(2, 2, figsize=(12, 10))
fig2.suptitle('Metrics (Bounding Box)', fontsize=16)

metrics_b = [
    ('Precision (B)', 'metrics/precision(B)'),
    ('Recall (B)', 'metrics/recall(B)'),
    ('mAP50 (B)', 'metrics/mAP50(B)'),
    ('mAP50-95 (B)', 'metrics/mAP50-95(B)')
]

axes2 = axes2.flatten()
for i, (title, col) in enumerate(metrics_b):
    ax = axes2[i]
    if col in df.columns:
        ax.plot(epochs, df[col], label=title, color='blue', marker='o')
        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Value')
    else:
        ax.text(0.5, 0.5, 'Data not found', ha='center', va='center')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('metrics_b_plot.png')

# 3. Metrics (Pose) Plot
fig3, axes3 = plt.subplots(2, 2, figsize=(12, 10))
fig3.suptitle('Metrics (Pose)', fontsize=16)

metrics_p = [
    ('Precision (P)', 'metrics/precision(P)'),
    ('Recall (P)', 'metrics/recall(P)'),
    ('mAP50 (P)', 'metrics/mAP50(P)'),
    ('mAP50-95 (P)', 'metrics/mAP50-95(P)')
]

axes3 = axes3.flatten()
for i, (title, col) in enumerate(metrics_p):
    ax = axes3[i]
    if col in df.columns:
        ax.plot(epochs, df[col], label=title, color='green', marker='o')
        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Value')
    else:
        ax.text(0.5, 0.5, 'Data not found', ha='center', va='center')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('metrics_p_plot.png')

# 4. Learning Rate Plot
fig4, ax4 = plt.subplots(figsize=(10, 6))
fig4.suptitle('Learning Rates', fontsize=16)

lr_cols = ['lr/pg0', 'lr/pg1', 'lr/pg2']
for col in lr_cols:
    if col in df.columns:
        ax4.plot(epochs, df[col], label=col, marker='o')

ax4.set_xlabel('Epoch')
ax4.set_ylabel('Learning Rate')
ax4.legend()
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('lr_plot.png')
