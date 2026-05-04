import torch
import numpy as np
import os
from scipy.io import loadmat
from Model import AmplitudeOnlyModel # Changed to the high-accuracy model
from collections import Counter

# Updated Action Names for better readability
ACTION_NAMES = {
    0: 'Nothing (Static Room)',
    1: 'Walking',
    2: 'Rotation',
    3: 'Jump',
    4: 'Wave',
    5: 'Lie Down',
    6: 'Pick Up',
    7: 'Sit Down',
    8: 'Stand Up'
}

def predict_action(mat_filepath):
    """
    Reads a .mat CSI file, applies Differential preprocessing, 
    and predicts action using the Attention-based model.
    """
    
    if not os.path.exists(mat_filepath):
        print(f"Error: Path '{mat_filepath}' not found.")
        return

    print(f"\nAnalyzing Wi-Fi signals from: {mat_filepath} ...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Model (Must be AmplitudeOnlyModel to match your 63% accuracy training)
    model = AmplitudeOnlyModel(num_classes=9).to(device)
    try:
        model.load_state_dict(torch.load("wifi_model.pth", map_location=device))
        model.eval()
    except Exception as e:
        print(f"Critical Error: Could not load wifi_model.pth. {e}")
        return
    
    # 2. Extract Data from .mat
    max_time = 1000
    try:
        mat_data = loadmat(mat_filepath)
        trace_data = mat_data['trace']
        num_packets = trace_data.shape[0]
        
        raw_amp_list = []
        for i in range(min(num_packets, max_time)):
            csi_complex = trace_data[i, 0]['csi'][0, 0]
            
            # Extract first 30 subcarriers to match training features
            if csi_complex.ndim == 3:
                subcarriers = csi_complex[0, 0, :30]
            else:
                subcarriers = csi_complex.flatten()[:30]
            
            raw_amp_list.append(np.abs(subcarriers))
            
        amp_data = np.array(raw_amp_list)

        # 3. Apply Differential CSI Preprocessing (Crucial for Accuracy)
        # This highlights movement and removes static room noise
        diff_amp = np.diff(amp_data, axis=0, prepend=amp_data[:1, :])
        
        # Z-Score Normalization
        amp_final = (diff_amp - np.mean(diff_amp)) / (np.std(diff_amp) + 1e-6)

        # Padding Time Dimension to 1000
        if amp_final.shape[0] < max_time:
            pad_len = max_time - amp_final.shape[0]
            padding = np.zeros((pad_len, 30))
            amp_final = np.vstack((amp_final, padding))
        else:
            amp_final = amp_final[:max_time, :]

    except Exception as e:
        print(f"Data Processing Error: {e}")
        return

    # 4. Prepare Tensor [Batch, Channels, Time]
    # We permute to (1, 30, 1000) because Conv1D expects channels in the middle
    amp_tensor = torch.tensor(amp_final, dtype=torch.float32).permute(1, 0).unsqueeze(0).to(device)
    
    # 5. Inference
    with torch.no_grad():
        outputs = model(amp_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted_class = torch.max(probabilities, 1)
        
    pred_idx = predicted_class.item()
    conf_score = confidence.item() * 100
    
    # 6. Display Result
    print("\n" + "=".center(50, "="))
    print(" WIFI CLASSIFICATION RESULT ".center(50, "#"))
    print("=".center(50, "="))
    print(f" Detected Action : {ACTION_NAMES.get(pred_idx, 'Unknown')}")
    print(f" Confidence      : {conf_score:.2f}%")
    print("=".center(50, "=") + "\n")

if __name__ == '__main__':
    print("CSI Activity Recognizer System")
    print("Type 'exit' to quit.\n")
    while True:
        target_file = input("Enter .mat file path: ").strip()
        
        if target_file.lower() == 'exit':
            break
        elif target_file == "":
            continue
            
        predict_action(target_file)