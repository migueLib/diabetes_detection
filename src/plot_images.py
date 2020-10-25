import torchvision
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def imshow(img, title, save, show=False):
    "Show sample images for tensor"
    img = img/2 +0.5
    npimg = img.numpy().transpose((1, 2, 0))
    plt.imshow(npimg)
    plt.axis("off")

    if title is not None:
        plt.title(title)
    
    if save is not None:
        Path("./output/").mkdir(parents=True, exist_ok=True)
        plt.savefig(f'./output/{save}-sample.png', bbox_inches='tight')

    if show:
        plt.show()
    
    
def plot_images(loader, classes, title=None, save=None):
    "loading a batch of images and labels"
    images, labels = next(iter(loader))
    images, labels = images[:8], labels[:8]
    
    # Show images
    # TODO: Remove Axis and add option to save
    imshow(torchvision.utils.make_grid(images), title=title, save=save)
    
    # Print labels
    # TODO:  Add labels directly to the plot
    print(" ".join('%5s' % classes[labels[j]] for j in range(len(images))))