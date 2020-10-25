import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

def plot_loss(data, ax=None):
    # If not ax create one
    if ax is None:
        fig, ax = plt.subplots(figsize=(16,9))
    
    # Take adata and plot line for the data (train or train and validation)
    sns.lineplot(ax=ax, data=data, lw=2, dashes=False)
    ax.set(xlabel='Epochs', ylabel='Loss', title="Loss x Epoch")
    sns.despine()
    return ax