import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from scipy.stats import entropy


def generate_psd_comparison():
    """
    Generates and compares two PSDs (Power Spectral Densities) to demonstrate
    Spectral Entropy concepts.

    Scenario 1: Low Entropy (Sleep Delta Waves)
    Scenario 2: High Entropy (Chaotic/Seizure-like Activity)
    """

    # 1. Setup Configuration
    fs = 128.0  # Sampling rate in Hz
    duration = 30.0  # Signal duration in seconds
    n_samples = int(fs * duration)
    t = np.linspace(0, duration, n_samples, endpoint=False)

    # 2. Generate Signals

    # --- Signal A: Low Entropy (Deep Sleep / Delta) ---
    # A dominant, slow oscillation at 2 Hz. Very predictable.
    freq_delta = 2.0
    sig_sleep = np.sin(2 * np.pi * freq_delta * t)
    # Add minimal noise to avoid numerical issues, but keep it clean
    sig_sleep += np.random.normal(0, 0.05, n_samples)

    # --- Signal B: High Entropy (Chaotic / Seizure-like) ---
    # Broadband noise to simulate maximum unpredictability across the spectrum.
    # In a real seizure, this might represent the "ictal chaos" or high-frequency
    # discharge phase.
    np.random.seed(42)  # Ensure reproducibility
    sig_seizure = np.random.normal(0, 1, n_samples)

    # 3. Compute PSD using Welch's Method
    # nperseg defines the window length. 2*fs gives us 0.5 Hz frequency resolution.
    nperseg = int(2 * fs)

    f_sleep, p_sleep = welch(sig_sleep, fs, nperseg=nperseg)
    f_seizure, p_seizure = welch(sig_seizure, fs, nperseg=nperseg)

    # 4. Filter Data to 0-35 Hz Range
    # We only care about the requested band.
    def filter_band(freqs, power, f_min=0, f_max=35):
        idx = np.where((freqs >= f_min) & (freqs <= f_max))
        return freqs[idx], power[idx]

    f_sleep_band, p_sleep_band = filter_band(f_sleep, p_sleep)
    f_seizure_band, p_seizure_band = filter_band(f_seizure, p_seizure)

    # 5. Calculate Spectral Entropy
    # Helper function for Shannon Entropy
    def get_spectral_entropy(power_spectrum):
        # Normalize to create a probability mass function (PMF)
        p_norm = power_spectrum / np.sum(power_spectrum)
        # Calculate entropy in bits (base 2)
        return entropy(p_norm, base=2)

    # 6. Visualize the Results
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)

    # Plot Sleep
    axes[0].plot(f_sleep_band, p_sleep_band, color='#1f77b4', lw=2)
    axes[0].set_title(f"Low Entropy")
    axes[0].set_xlabel("Frequency (Hz)")
    axes[0].set_ylabel("Power Spectral Density")
    axes[0].grid(True, alpha=0.5)

    # Plot Seizure
    axes[1].plot(f_seizure_band, p_seizure_band, color='#d62728', lw=2)
    axes[1].set_title(f"High Entropy")
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].grid(True, alpha=0.5)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    generate_psd_comparison()