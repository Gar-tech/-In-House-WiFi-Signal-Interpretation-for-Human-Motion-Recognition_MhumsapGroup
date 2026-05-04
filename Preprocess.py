import numpy as np

# -------- Amplitude Processing --------
def normalize_amplitude(amp):
    """
    Min-Max Normalization: Rescales the amplitude values to a standard range [0, 1].
    Neural networks learn much faster and more stably when input features share the same scale.
    """
    min_val = np.min(amp)
    max_val = np.max(amp)
    
    # The '1e-8' (0.00000001) is a small epsilon added to the denominator 
    # to prevent a 'Division by Zero' error in case all amplitude values are identical.
    return (amp - min_val) / (max_val - min_val + 1e-8)

# -------- Phase Processing --------
def unwrap_phase(phase):
    """
    Phase Unwrapping: Fixes the artificial jumps/discontinuities in phase data.
    Since phase is calculated using arctan, it wraps around at pi and -pi (jumps from 180 to -180 degrees).
    This function detects those jumps and makes the phase curve continuous.
    """
    return np.unwrap(phase, axis=0)

# -------- Gaussian Encoding --------
def gaussian_encoding(x, sigma=1.0):
    """
    Applies a Non-linear Gaussian function to the data.
    This acts as a soft thresholding/smoothing mechanism. It highlights values 
    closer to the mean and suppresses extreme outliers (environmental noise).
    """
    # Formula: e^(-x^2 / 2*sigma^2)
    return np.exp(-(x**2) / (2 * sigma**2))

# -------- Main Pipeline (Integration) --------
def preprocess_csi(csi):
    """
    The main preprocessing pipeline that orchestrates the data cleaning steps.
    
    Input csi shape: (time_steps, subcarriers, 2)
    Index 0 of the last dimension contains the Amplitude.
    Index 1 of the last dimension contains the Phase.
    """
    # 1. Split the raw CSI tensor into separate Amplitude and Phase arrays
    amp = csi[:, :, 0]
    phase = csi[:, :, 1]
    
    # 2. Base Preprocessing
    # Scale amplitude to [0,1] and fix phase discontinuities
    amp = normalize_amplitude(amp)
    phase = unwrap_phase(phase)
    
    # 3. Non-linear Transformation (Feature Enhancement)
    # Encode both signals into a Gaussian space to reduce noise and enhance patterns
    amp = gaussian_encoding(amp)
    phase = gaussian_encoding(phase)
    
    # Return the clean, ready-to-use data matrices
    return amp, phase