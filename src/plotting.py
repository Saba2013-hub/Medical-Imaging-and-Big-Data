
import numpy as np, matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
GRADE_SHORT = ['0 No DR', '1 Mild', '2 Moderate', '3 Severe', '4 Prolif.']

def plot_confusion(ax, y, pred, title, subtitle=None, cmap='Blues'):
    cm = confusion_matrix(y, pred, labels=range(5))
    cmn = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    ax.imshow(cmn, cmap=cmap, vmin=0, vmax=1)
    for i in range(5):
        for j in range(5):
            dark = cmn[i, j] > 0.55
            ax.text(j, i - 0.12, f'{cm[i, j]}', ha='center', va='center', fontsize=13,
                    fontweight='semibold', color='white' if dark else '#1F2328')
            ax.text(j, i + 0.22, f'{100 * cmn[i, j]:.0f}%', ha='center', va='center', fontsize=9.5,
                    color='white' if dark else '#6B7280')
    ax.set_xticks(range(5), GRADE_SHORT, rotation=0, fontsize=10)
    ax.set_yticks(range(5), GRADE_SHORT, fontsize=10)
    ax.set_xticks(np.arange(-.5, 5), minor=True); ax.set_yticks(np.arange(-.5, 5), minor=True)
    ax.grid(which='minor', color='white', linewidth=2); ax.grid(which='major', visible=False)
    ax.tick_params(which='both', length=0)
    for s in ax.spines.values(): s.set_visible(False)
    ax.set_xlabel('Predicted grade'); ax.set_ylabel('True grade')
    ax.set_title(title, loc='left', pad=26 if subtitle else 12)
    if subtitle:
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=11, color='#6B7280', va='bottom')



def plot_learning(axes, log, accent="#2F6690", muted="#6B7280", bad="#B23A48", good="#4F8A5B"):
    """axes = (ax_loss, ax_metric). Marks the epoch selected on validation QWK."""
    b = int(log.best_epoch.iloc[0])
    ax1, ax2 = axes
    ax1.plot(log.epoch, log.train_loss, color=muted, label="Train")
    ax1.plot(log.epoch, log.val_loss, color=accent, label="Validation")
    i = log.val_loss.idxmin()
    ax1.scatter(log.epoch[i], log.val_loss[i], s=60, color=accent, zorder=3)
    ax1.annotate(f"lowest validation loss (epoch {int(log.epoch[i])})", (log.epoch[i], log.val_loss[i]),
                 xytext=(0, -22), textcoords="offset points", ha="center", fontsize=10, color=accent)
    ax2.plot(log.epoch, log.val_QWK, color=accent, label="QWK")
    ax2.plot(log.epoch, log.val_referable_AUROC, color=good, ls="--", label="Referable AUROC")
    for ax in axes:
        ax.axvline(b, color=bad, lw=1.2, ls=":")
        ax.text(b, 1.01, f"selected epoch {b}", transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=10, color=bad)
        ax.set_xlabel("Epoch")
    ax1.legend(loc="upper right"); ax2.legend(loc="lower right")
    ax1.set_ylabel("Weighted cross-entropy"); ax2.set_ylabel("Validation score")
    ax1.set_title("Loss", loc="left", pad=22); ax2.set_title("Validation metrics", loc="left", pad=22)
