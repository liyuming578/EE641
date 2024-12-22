import torch
from torch import nn, optim
from evaluate import test
import pandas as pd
import matplotlib.pyplot as plt


def train(model, train_dataloader, test_dataloader, device, epochs=100, lr=1e-4):
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    pos_weight = torch.tensor([6.45], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    model.to(device)

    results = {
        "Epoch": [],
        "Train Loss": [],
        "Train Accuracy": [],
        "Train Precision": [],
        "Train Recall": [],
        "Train F1": [],
        "Test Loss": [],
        "Test Accuracy": [],
        "Test Precision": [],
        "Test Recall": [],
        "Test F1": []
    }

    for epoch in range(epochs):
        model.train()
        average_loss, accuracy, recall, precision, F1 = train_one_epoch(train_dataloader, model, device, optimizer, loss_fn)
        test_loss, test_accuracy, test_recall, test_precision, test_f1 = test(test_dataloader, model, device, loss_fn)

        # Save results for this epoch
        results["Epoch"].append(epoch + 1)
        results["Train Loss"].append(average_loss)
        results["Train Accuracy"].append(accuracy)
        results["Train Recall"].append(recall)
        results["Train Precision"].append(precision)
        results["Train F1"].append(F1)
        results["Test Loss"].append(test_loss)
        results["Test Accuracy"].append(test_accuracy)
        results["Test Recall"].append(test_recall)
        results["Test Precision"].append(test_precision)
        results["Test F1"].append(test_f1)

        # Print train results
        print(f"Epoch {epoch + 1}: Train Results:")
        print(f"Train Average Loss: {average_loss:.4f}, Train Accuracy: {accuracy:.4f}")
        print(f"Train Recall: {recall:.4f}, Train Precision:{precision:.4f}, Train F1 Score: {F1:.4f}\n")

        # Print test results
        print(f"Epoch {epoch + 1}: Test Results:")
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
        print(f"Test Recall: {test_recall:.4f}, Test Precision:{test_precision:.4f}, Test F1 Score: {test_f1:.4f}\n")

    # Save model parameters after each epoch
    torch.save(model.state_dict(), f'model.pth')

    # Save training results to a DataFrame and then to a CSV file
    df = pd.DataFrame(results)
    df.to_csv('training_results.csv', index=False)

    # Plotting the first graph: Loss over epochs
    plt.figure(figsize=(10, 5))
    plt.plot(results["Epoch"], results["Train Loss"], label='Train Loss')  # Plot training loss
    plt.plot(results["Epoch"], results["Test Loss"], label='Test Loss')  # Plot testing loss
    plt.title('Loss vs. Epochs')  # Title of the plot
    plt.xlabel('Epochs')  # X-axis label
    plt.ylabel('Loss')  # Y-axis label
    plt.legend()  # Show legend
    plt.savefig('training_loss.png', dpi=600)  # Save the loss graph to a file
    plt.show()  # Display the graph

    # Plotting the second graph: Accuracy, Recall, and F1 Score
    plt.figure(figsize=(10, 5))

    # Plot Accuracy
    plt.plot(results["Epoch"], results["Train Accuracy"], 'r-',
             label='Train Accuracy')  # Training accuracy (solid red line)
    plt.plot(results["Epoch"], results["Test Accuracy"], 'r--',
             label='Test Accuracy')  # Testing accuracy (dashed red line)

    # Plot Recall
    plt.plot(results["Epoch"], results["Train Recall"], 'g-',
             label='Train Recall')  # Training recall (solid green line)
    plt.plot(results["Epoch"], results["Test Recall"], 'g--', label='Test Recall')  # Testing recall (dashed green line)

    # Plot Precision
    plt.plot(results["Epoch"], results["Train Precision"], 'b-',
             label='Train Precision')  # Training recall (solid green line)
    plt.plot(results["Epoch"], results["Test Precision"], 'b--', label='Test Precision')

    # Plot F1 Score
    plt.plot(results["Epoch"], results["Train F1"], 'y-', label='Train F1')  # Training F1 Score (solid blue line)
    plt.plot(results["Epoch"], results["Test F1"], 'y--', label='Test F1')  # Testing F1 Score (dashed blue line)

    plt.title('Performance Metrics vs. Epochs')  # Title of the plot
    plt.xlabel('Epochs')  # X-axis label
    plt.ylabel('Metrics')  # Y-axis label
    plt.legend(loc='upper right')  # Show legend
    plt.ylim(0, 2)
    plt.savefig('training_performance_metrics.png', dpi=600)  # Save the performance metrics graph to a file
    plt.show()  # Display the graph


def train_one_epoch(dataloader, model, device, optimizer, loss_fn):
    total_loss = 0
    correct = 0
    total_samples = 0
    total_TP = 0
    total_FN = 0
    total_FP = 0

    for batch, (features, labels) in enumerate(dataloader):
        features, labels = features.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(features)
        outputs = outputs.view(-1, outputs.size(-1))  # Flatten outputs but keep class dimension
        labels = labels.view(-1, labels.size(-1))  # Flatten labels but keep class dimension

        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * features.size(0)

        preds = torch.sigmoid(outputs)
        preds = (preds >= 0.5).float()
        correct += (preds == labels).sum().item()
        total_samples += labels.numel()

        TP = ((preds == 1) & (labels == 1)).sum().item()
        FN = ((preds == 0) & (labels == 1)).sum().item()
        FP = ((preds == 1) & (labels == 0)).sum().item()

        total_TP += TP
        total_FN += FN
        total_FP += FP

    average_loss = total_loss / len(dataloader.dataset)
    accuracy = correct / total_samples
    recall = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0
    precision = total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0
    F1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return average_loss, accuracy, recall, precision, F1
