import torch
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import pandas as pd

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=5):
    # Initialize lists to track per-epoch training and validation loss, accuracy, and other metrics
    train_losses, train_accuracies, train_precisions, train_recalls, train_f1s = [], [], [], [], []
    val_losses, val_accuracies, val_precisions, val_recalls, val_f1s = [], [], [], [], []

    for epoch in range(num_epochs):
        # Training Phase
        model.train()
        train_loss = 0.0
        train_corrects = 0
        total_train = 0
        train_preds, train_labels = [], []

        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            labels = labels.unsqueeze(1).float()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            preds = torch.round(outputs)
            train_loss += loss.item() * images.size(0)
            train_corrects += torch.sum(preds == labels.data)
            total_train += labels.size(0)
            train_preds.extend(preds.view(-1).tolist())
            train_labels.extend(labels.view(-1).tolist())

        epoch_train_loss = train_loss / total_train
        epoch_train_acc = train_corrects.float() / total_train
        epoch_train_precision = precision_score(train_labels, train_preds)
        epoch_train_recall = recall_score(train_labels, train_preds)
        epoch_train_f1 = f1_score(train_labels, train_preds)
        train_losses.append(epoch_train_loss)
        train_accuracies.append(epoch_train_acc)
        train_precisions.append(epoch_train_precision)
        train_recalls.append(epoch_train_recall)
        train_f1s.append(epoch_train_f1)

        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        total_val = 0
        val_preds, val_labels = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images)
                labels = labels.unsqueeze(1).float()
                loss = criterion(outputs, labels)

                preds = torch.round(outputs)
                val_loss += loss.item() * images.size(0)
                val_corrects += torch.sum(preds == labels.data)
                total_val += labels.size(0)
                val_preds.extend(preds.view(-1).tolist())
                val_labels.extend(labels.view(-1).tolist())

        epoch_val_loss = val_loss / total_val
        epoch_val_acc = val_corrects.float() / total_val
        epoch_val_precision = precision_score(val_labels, val_preds)
        epoch_val_recall = recall_score(val_labels, val_preds)
        epoch_val_f1 = f1_score(val_labels, val_preds)
        val_losses.append(epoch_val_loss)
        val_accuracies.append(epoch_val_acc)
        val_precisions.append(epoch_val_precision)
        val_recalls.append(epoch_val_recall)
        val_f1s.append(epoch_val_f1)

        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)
        print(f'Train Loss: {epoch_train_loss:.4f} - Train Accuracy: {epoch_train_acc:.4f} - Train Precision: {epoch_train_precision:.4f} - Train Recall: {epoch_train_recall:.4f} - Train F1: {epoch_train_f1:.4f}')
        print(f'Validation Loss: {epoch_val_loss:.4f} - Validation Accuracy: {epoch_val_acc:.4f} - Validation Precision: {epoch_val_precision:.4f} - Validation Recall: {epoch_val_recall:.4f} - Validation F1: {epoch_val_f1:.4f}')
    
    return model, (train_losses, train_accuracies, val_losses, val_accuracies, train_precisions, train_recalls, train_f1s, val_precisions, val_recalls, val_f1s)


# Evaluation function
def evaluate_model(model, test_loader):
    model.eval()
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            predicted = (outputs.data > 0.5).float()
            all_labels.extend(labels.numpy())
            all_preds.extend(predicted.numpy())
    cm = confusion_matrix(all_labels, all_preds)
    return pd.DataFrame(cm, index=['Actual Normal', 'Actual Pneumonia'], 
                        columns=['Predicted Normal', 'Predicted Pneumonia'])