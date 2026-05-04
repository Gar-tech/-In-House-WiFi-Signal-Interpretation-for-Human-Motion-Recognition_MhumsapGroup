import torch
import os
import time
import psutil
import numpy as np
import matplotlib.pyplot as plt  # Added
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from Model import AmplitudeOnlyModel
from Utils import CSIDataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from collections import Counter

if __name__ == '__main__':
    # -------- Load Data --------
    csv_path = "D:/WIMANS/archive/annotation.csv"
    mat_path = "archive/wifi_csi/amp"

    print("Preparing Dataset...")
    dataset = CSIDataset(csv_path, mat_path)
    
    # Split data
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    # --- Balance the training batches ---
    train_indices = train_ds.indices
    labels = [dataset.label_map[str(dataset.df.iloc[i][dataset.activity_col]).strip()] for i in train_indices]
    class_counts = np.bincount(labels)
    weights = 1. / class_counts
    samples_weights = torch.from_numpy(weights[labels])
    sampler = WeightedRandomSampler(samples_weights, len(samples_weights))

    train_loader = DataLoader(train_ds, batch_size=32, sampler=sampler)
    
    # -------- Device & Process Setup --------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    process = psutil.Process(os.getpid())

    # -------- Model Setup --------
    model = AmplitudeOnlyModel(num_classes=9).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    # -------- OS Metrics Tracking Lists --------
    epoch_axis = []
    mem_history = []
    cpu_history = []
    disk_history = []
    latency_history = []

    # -------- Training Loop --------
    epochs = 50
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        all_labels = []
        all_preds = []
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch")
        start_time = time.time()

        for batch_idx, (amp, label) in enumerate(loop):
            batch_start = time.time()
            
            amp, label = amp.to(device), label.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(amp)
            loss = criterion(outputs, label)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Metrics calculation
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            
            all_labels.extend(label.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            
            batch_time = (time.time() - batch_start) * 1000 # ms
            progress = (batch_idx + 1) / len(train_loader) * 100

            # OS Metrics reporting and recording at the end of the epoch
            if (batch_idx + 1) == len(train_loader):
                mem_info = process.memory_info().rss / (1024 * 1024)
                cpu_usage = psutil.cpu_percent()
                io_counters = psutil.disk_io_counters()
                read_mb = io_counters.read_bytes / (1024 * 1024) if io_counters else 0

                # Store values for plotting
                epoch_axis.append(epoch + 1)
                mem_history.append(mem_info)
                cpu_history.append(cpu_usage)
                disk_history.append(read_mb)
                latency_history.append(batch_time)

                tqdm.write("\n--------------------------------------------------")
                tqdm.write(f"Epoch [{epoch + 1}/{epochs}], Batch [{batch_idx + 1}/{len(train_loader)}]")
                tqdm.write(f"Progress: {progress:.2f}%")
                tqdm.write(f"Loss: {loss.item():.4f}")
                tqdm.write("OS METRICS:")
                tqdm.write(f"   [Memory] RSS Usage: {mem_info:.2f} MB")
                tqdm.write(f"   [CPU]     Load: {cpu_usage}%")
                tqdm.write(f"   [Disk]    Total Read: {read_mb:.2f} MB")
                tqdm.write(f"   [Speed]   Latency: {batch_time:.2f} ms/batch")
                tqdm.write("--------------------------------------------------\n")

        # -------- Epoch Summary Metrics --------
        avg_loss = total_loss / len(train_loader)
        acc = accuracy_score(all_labels, all_preds)
        error_rate = 1.0 - acc
        precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

        print("\n=========================================")
        print(f"Result Epoch {epoch + 1}/{epochs}")
        print(f"Average Loss : {avg_loss:.4f}")
        print(f"Accuracy     : {acc * 100:.2f}%")
        print(f"Error Rate   : {error_rate * 100:.2f}%")
        print(f"Precision    : {precision:.4f}")
        print(f"Recall       : {recall:.4f}")
        print(f"F1-Score     : {f1:.4f}")
        print("Label distribution:", Counter(all_labels))
        print("Prediction distribution:", Counter(all_preds))
        print("=========================================\n")

    # -------- Visualizing OS Metrics --------
    print("Generating OS Metrics Plots...")
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Hardware Resource Monitoring Over 50 Epochs', fontsize=16)

    # Memory Usage Plot
    axs[0, 0].plot(epoch_axis, mem_history, color='tab:blue', linewidth=2, marker='o', markersize=4)
    axs[0, 0].set_title('Memory Usage (RSS)')
    axs[0, 0].set_ylabel('MB')
    axs[0, 0].grid(True, linestyle='--', alpha=0.7)

    # CPU Load Plot
    axs[0, 1].plot(epoch_axis, cpu_history, color='tab:red', linewidth=2, marker='s', markersize=4)
    axs[0, 1].set_title('CPU Load')
    axs[0, 1].set_ylabel('Percentage (%)')
    axs[0, 1].grid(True, linestyle='--', alpha=0.7)

    # Disk Read Plot
    axs[1, 0].plot(epoch_axis, disk_history, color='tab:green', linewidth=2, marker='^', markersize=4)
    axs[1, 0].set_title('Disk IO (Total Read)')
    axs[1, 0].set_ylabel('MB')
    axs[1, 0].grid(True, linestyle='--', alpha=0.7)

    # Latency Plot
    axs[1, 1].plot(epoch_axis, latency_history, color='tab:purple', linewidth=2, marker='d', markersize=4)
    axs[1, 1].set_title('Batch Processing Latency')
    axs[1, 1].set_ylabel('ms')
    axs[1, 1].grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('training_os_metrics.png')
    print("Graph saved as training_os_metrics.png")
   

    # -------- Save Model --------
    torch.save(model.state_dict(), "wifi_model.pth")
    print("Finished saving wifi_model.pth")