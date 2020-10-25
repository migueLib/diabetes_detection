# Build-in
import os

# External
import torch
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

# Local
from src.batch_summary import batch_summary

def test_phase(model, model_path, dataloader, device, logger=None):
    all_labels = list()
    all_output = list()
    
    # Load moodel weights
    model.load_state_dict(torch.load(model_path))
    
    # Set model to evaluition mode
    model.eval()
    
    for i, data in enumerate(dataloader):        
        # Move data to device
        inputs = data[0].to(device)
        labels = data[1].to(device)
        
        # Set gradient to false
        with torch.set_grad_enabled(False):
            
            # IV3 can output an auxiliary layer
            try:
                outputs, aux = model(inputs)
            except:
                outputs = model(inputs)
                
        # Save a copy of the labels and outputs 
        all_labels.extend(data[1].tolist())
        all_output.extend(outputs.cpu().tolist())
        
    # Binarize 
    _, all_predicted = torch.max(torch.Tensor(all_output), 1)
    all_predicted = all_predicted.tolist()
    
    # Calculate AUC 
    auc = roc_auc_score(all_labels, all_predicted)

    results = {
        "test_labels": all_labels,
        "test_outputs": all_output,
        "test_predicted": all_predicted,
        "test_auc": auc
    }
    
    return results

        

        
            