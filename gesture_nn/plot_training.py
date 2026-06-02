"""
Plot training curves from saved results.json for full and cnn_only models.
"""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False


def load_history(path):
    with open(path, 'r') as f:
        data = json.load(f)
    history = data['history']
    epochs = range(len(history['train']))
    train_loss = [e['loss'] for e in history['train']]
    val_loss = [e['loss'] for e in history['val']]
    train_acc = [e['accuracy'] for e in history['train']]
    val_acc = [e['accuracy'] for e in history['val']]
    train_f1 = [e['macro_f1'] for e in history['train']]
    val_f1 = [e['macro_f1'] for e in history['val']]
    return {
        'epochs': list(epochs),
        'train_loss': train_loss, 'val_loss': val_loss,
        'train_acc': train_acc, 'val_acc': val_acc,
        'train_f1': train_f1, 'val_f1': val_f1,
        'best_epoch': data['best_epoch'],
        'test_acc': data['test_metrics']['accuracy'],
        'test_f1': data['test_metrics']['macro_f1'],
    }


full = load_history('checkpoints/full/results.json')
cnn = load_history('checkpoints/cnn_only/results.json')

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# --- Plot 1: Loss ---
ax = axes[0, 0]
ax.plot(full['epochs'], full['train_loss'], 'b-', alpha=0.5, linewidth=0.8, label='Full Train')
ax.plot(full['epochs'], full['val_loss'], 'b-', linewidth=1.5, label='Full Val')
ax.plot(cnn['epochs'], cnn['train_loss'], 'r-', alpha=0.5, linewidth=0.8, label='CNN-only Train')
ax.plot(cnn['epochs'], cnn['val_loss'], 'r-', linewidth=1.5, label='CNN-only Val')
ax.axvline(full['best_epoch'], color='b', linestyle='--', alpha=0.4, label=f'Full best={full["best_epoch"]}')
ax.axvline(cnn['best_epoch'], color='r', linestyle='--', alpha=0.4, label=f'CNN best={cnn["best_epoch"]}')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('Training & Validation Loss')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# --- Plot 2: Accuracy ---
ax = axes[0, 1]
ax.plot(full['epochs'], full['train_acc'], 'b-', alpha=0.5, linewidth=0.8, label='Full Train')
ax.plot(full['epochs'], full['val_acc'], 'b-', linewidth=1.5, label='Full Val')
ax.plot(cnn['epochs'], cnn['train_acc'], 'r-', alpha=0.5, linewidth=0.8, label='CNN-only Train')
ax.plot(cnn['epochs'], cnn['val_acc'], 'r-', linewidth=1.5, label='CNN-only Val')
ax.axvline(full['best_epoch'], color='b', linestyle='--', alpha=0.4)
ax.axvline(cnn['best_epoch'], color='r', linestyle='--', alpha=0.4)
ax.set_xlabel('Epoch')
ax.set_ylabel('Accuracy')
ax.set_title('Training & Validation Accuracy')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# --- Plot 3: Macro F1 ---
ax = axes[1, 0]
ax.plot(full['epochs'], full['train_f1'], 'b-', alpha=0.5, linewidth=0.8, label='Full Train')
ax.plot(full['epochs'], full['val_f1'], 'b-', linewidth=1.5, label='Full Val')
ax.plot(cnn['epochs'], cnn['train_f1'], 'r-', alpha=0.5, linewidth=0.8, label='CNN-only Train')
ax.plot(cnn['epochs'], cnn['val_f1'], 'r-', linewidth=1.5, label='CNN-only Val')
ax.axvline(full['best_epoch'], color='b', linestyle='--', alpha=0.4)
ax.axvline(cnn['best_epoch'], color='r', linestyle='--', alpha=0.4)
ax.set_xlabel('Epoch')
ax.set_ylabel('Macro F1')
ax.set_title('Training & Validation Macro F1')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# --- Plot 4: Test set bar comparison ---
ax = axes[1, 1]
models = ['CNN+Attention\n(Full)', 'CNN-only']
accs = [full['test_acc'], cnn['test_acc']]
f1s = [full['test_f1'], cnn['test_f1']]
x = range(len(models))
w = 0.35
bars1 = ax.bar([i - w/2 for i in x], accs, w, label='Test Accuracy', color=['#4472C4', '#ED7D31'])
bars2 = ax.bar([i + w/2 for i in x], f1s, w, label='Test Macro F1', color=['#70AD47', '#FFC000'])
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002, f'{bar.get_height():.4f}',
            ha='center', va='bottom', fontsize=10)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002, f'{bar.get_height():.4f}',
            ha='center', va='bottom', fontsize=10)
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_ylim(0.90, 1.0)
ax.set_title('Test Set Performance Comparison')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis='y')

fig.suptitle('DBEW-NN Gesture Classifier — Training Curves', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
print('Saved: training_curves.png')
