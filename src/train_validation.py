# Build-in
import os
from copy import deepcopy
from pathlib import Path
import pickle

# External
import torch

# Machine Learning
from sklearn.metrics import roc_auc_score

# Local
from src.batch_summary import batch_summary


# Get in arguments from command line
def train_validation_phase(model, dataset, dataloader, device, epochs,
                           criterion, optimizer, save, logger):
    vec_acc_trn = []
    vec_lss_trn = []

    vec_acc_val = []
    vec_lss_val = []
    vec_auc_val = []

    best_acc = 0.0
    best_model = None
    
    # If model path doesn't exist create it 
    Path(os.path.split(save)[0]).mkdir(parents=True, exist_ok=True)

    # Iterate
    for epoch in range(epochs):
        for phase in ["train", "valid"]:
            # Set model to training mode
            if phase == "train":
                model.train()
            else:
                model.eval()

            # Initializing per/epoch variables
            running_loss = 0.0
            running_true = 0
            auc_vec = []

            # Iterate over data
            # Train a mini-batch a time
            for i, data in enumerate(dataloader[phase]):
                # Move the data to the specified device
                inputs = data[0].to(device)
                labels = data[1].to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                with torch.set_grad_enabled(phase == "train"):
                    # Inception-v3 can output an auxiliary layer
                    try:
                        outputs, aux = model(inputs)
                    except ValueError:
                        outputs = model(inputs)

                    _, predicted = torch.max(outputs, 1)

                    # Calculate loss function
                    loss = criterion(outputs, labels)

                    # Backward and optimize if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # Batch summary
                # Print every x mini-batches
                batch_summary(100, epoch, epochs, i, labels, dataset[phase],
                              loss, logger)

                # Get the loss anf trues for the running batch
                running_loss += loss.item() * inputs.size(0)

                # noinspection PyTypeChecker
                running_true += torch.sum(labels.data == predicted)

                if phase == "valid":
                    auc = roc_auc_score(labels.cpu(), predicted.cpu())
                    auc_vec.append(auc)

            # Calculate epoch loss and epoch accuracy
            epoch_lss = running_loss / len(dataset[phase])
            epoch_acc = int(running_true) / len(dataset[phase])
            logger.info(f"Current lr: {optimizer.param_groups[0]['lr']}")
            logger.info(f"{phase}, Loss: {epoch_lss:.4f} Acc: {epoch_acc:.4f}")

            # For training get loss and accuracy
            if phase == "train":
                vec_lss_trn.append(epoch_lss)
                vec_acc_trn.append(epoch_acc)

            # For validation get AUC, loss and accuracy
            if phase == "valid":
                epoch_auc = sum(auc_vec) / len(auc_vec)
                print(epoch_auc)
                vec_auc_val.append(epoch_auc)
                vec_acc_val.append(epoch_acc)
                vec_lss_val.append(epoch_lss)

            if phase == "valid" and epoch >= 20:
                if epoch_acc - 0.005 >= best_acc:
                    best_acc = epoch_acc
                    print(f"new best model  with acc = {best_acc}")
                    best_model = deepcopy(model.state_dict())
                    torch.save(model.state_dict(), save)

            # If
            if phase == "valid":
                if len(vec_acc_val) > 5:
                    # past_average_validation_accuracy =  `pava`
                    pava = sum(vec_acc_val[-5:]) / len(vec_acc_val[-5:])
                    if (epoch_acc - pava) < 0.005:
                        logger.info(f"Fine-tune accuracy has not improved "
                                    f"by 0.5% in the last 5 epochs")

    results = {
        "train_accuracy": vec_acc_trn,
        "train_loss": vec_lss_trn,
        "validation_accuracy": vec_acc_val,
        "validation_loss": vec_lss_val,
        "validation_auc": vec_auc_val,
        "best_acc": best_acc
    }

    # save model per epoch
    # load best model weights
    model.load_state_dict(best_model)
    torch.save(model.state_dict(), save)

    # Save the results to a pickle file
    # Save the results to a pickle file
    with open(f"{save}_results.pkl", "wb") as RESULTS:
        pickle.dump(results, RESULTS, protocol=pickle.HIGHEST_PROTOCOL)
    
    return results
