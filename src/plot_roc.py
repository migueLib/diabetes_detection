import matplotlib.pyplot as plt

def plot_roc_curve(fpr, tpr, roc_auc):
    # Create canvas
    fig, ax = plt.subplots(ncols=1, constrained_layout=True)
    
    # Create label
    label = f"ROC curve (area = {roc_auc:.2f})"
    
    # Plot roc
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=label)
    
    # Plot line
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    
    # Ax labels
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver operating characteristic')
    
    # Legend
    ax.legend(loc="lower right")
    
    return fig, ax