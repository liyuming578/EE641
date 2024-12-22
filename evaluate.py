import torch


def test(dataloader, model, device, loss_fn):
    model.eval()
    total_loss = 0
    correct = 0
    total_samples = 0
    total_TP = 0
    total_FN = 0
    total_FP = 0

    with torch.no_grad():
        for features, labels in dataloader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)

            outputs = outputs.view(-1, outputs.size(-1))
            labels = labels.view(-1, labels.size(-1))

            loss = loss_fn(outputs, labels)
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
