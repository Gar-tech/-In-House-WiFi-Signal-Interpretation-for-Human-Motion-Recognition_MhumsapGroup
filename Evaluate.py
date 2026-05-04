import torch
from torch.utils.data import DataLoader
from Model import AmplitudeOnlyModel
from Utils import CSIDataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import numpy as np
from collections import Counter

if __name__ == '__main__':
    
    # 1. Define the file paths (Ensuring paths match your local environment)
    csv_path = "D:/WIMANS/archive/annotation.csv"
    mat_path = "archive/wifi_csi/amp" # Updated to 'amp' folder to match training
    
    print("Preparing for evaluation...")
    
    # 2. Initialize Dataset
    # This now automatically applies Differential CSI and Z-score normalization 
    # as defined in your latest Utils.py
    test_dataset = CSIDataset(csv_path, mat_path)
    
    # 3. Create DataLoader
    # Using a larger batch size for evaluation speeds up inference
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 4. Device Management
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing With: {device}")
        
    # 5. Model Initialization (Updated to AmplitudeOnlyModel)
    model = AmplitudeOnlyModel(num_classes=9).to(device)
    
    # 6. Load the pre-trained weights
    # Ensure the filename matches what you saved in Train.py
    try:
        model.load_state_dict(torch.load("wifi_model.pth", map_location=device))
        print("Weights loaded successfully.")
    except FileNotFoundError:
        print("Error: wifi_model.pth not found. Please train the model first.")
        exit()
    
    # 7. Evaluation Mode Activation
    model.eval()
    
    print("Starting Inference...")
    
    all_labels = []
    all_preds = []
    
    # 8. Inference Loop
    with torch.no_grad():
        for batch_idx, (amp, label) in enumerate(test_loader):
            
            amp = amp.to(device)
            label = label.to(device)
            
            # Forward Pass
            outputs = model(amp)
            
            # Prediction Extraction
            _, predicted = torch.max(outputs, 1)
            
            all_labels.extend(label.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            
            if (batch_idx + 1) % 50 == 0:
                print(f"Processed {len(all_labels)} samples...")

    # 9. Comprehensive Metrics Calculation
    accuracy = accuracy_score(all_labels, all_preds) * 100
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    conf_matrix = confusion_matrix(all_labels, all_preds)

    print("\n" + "="*40)
    print("FINAL EVALUATION RESULTS")
    print("="*40)
    print(f"Accuracy     : {accuracy:.2f}%")
    print(f"Precision    : {precision:.4f}")
    print(f"Recall       : {recall:.4f}")
    print(f"F1-Score     : {f1:.4f}")
    print("-" * 40)
    print("Class Distribution in Test Set:")
    print(Counter(all_labels))
    print("\nConfusion Matrix:")
    print(conf_matrix)
    print("="*40)