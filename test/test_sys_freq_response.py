import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def fmt_freq(value):
    if value is None or not np.isfinite(value):
        return "N/A"
    if value >= 1e6:
        return f"{value / 1e6:.3f} MHz"
    if value >= 1e3:
        return f"{value / 1e3:.3f} kHz"
    return f"{value:.3f} Hz"


def extract_fundamental_amplitude(y, fs, target_freq):
    """
    Extracts the amplitude by finding the local maximum peak near the
    target frequency. Returns the peak amplitude, peak frequency,
    and the full spectrum for debugging.
    """
    y = np.asarray(y) - np.mean(y)  # Remove DC component
    window = np.hanning(len(y))
    spectrum = np.fft.rfft(y * window)
    spectrum_mag = np.abs(spectrum)
    freqs_fft = np.fft.rfftfreq(len(y), d=1 / fs)

    # Search within a +/- 15% tolerance window around the target frequency
    lower_bound = target_freq * 0.85
    upper_bound = target_freq * 1.15
    mask = (freqs_fft >= lower_bound) & (freqs_fft <= upper_bound)

    if np.any(mask):
        mask_indices = np.where(mask)[0]
        peak_idx_in_mask = np.argmax(spectrum_mag[mask])
        peak_idx = mask_indices[peak_idx_in_mask]
    else:
        # Fallback to nearest bin if window contains no bins
        peak_idx = np.argmin(np.abs(freqs_fft - target_freq))

    spectrum_mag = spectrum_mag * (4.0 / len(y))  # Hanning window normalization factor

    return spectrum_mag[peak_idx], freqs_fft[peak_idx], freqs_fft, spectrum_mag


def find_bandpass_cutoffs(freqs, gains_db):
    """
    Finds both the lower (f_L) and upper (f_H) -3dB cutoff frequencies
    relative to the absolute peak gain of the passband.
    """
    pk_idx = np.argmax(gains_db)
    below_3db = np.where(gains_db <= -3.0)[0]

    lower_cutoff = None
    upper_cutoff = None

    # Lower Cutoff (Search frequencies below the peak)
    lower_idxs = below_3db[below_3db < pk_idx]
    if len(lower_idxs) > 0:
        i = lower_idxs[-1]
        if i < len(freqs) - 1:
            f1, f2 = freqs[i], freqs[i + 1]
            g1, g2 = gains_db[i], gains_db[i + 1]
            log_f1, log_f2 = np.log10(f1), np.log10(f2)
            log_fc = log_f1 + (-3.0 - g1) * (log_f2 - log_f1) / (g2 - g1)
            lower_cutoff = 10**log_fc

    # Upper Cutoff (Search frequencies above the peak)
    upper_idxs = below_3db[below_3db > pk_idx]
    if len(upper_idxs) > 0:
        i = upper_idxs[0]
        if i > 0:
            f1, f2 = freqs[i - 1], freqs[i]
            g1, g2 = gains_db[i - 1], gains_db[i]
            log_f1, log_f2 = np.log10(f1), np.log10(f2)
            log_fc = log_f1 + (-3.0 - g1) * (log_f2 - log_f1) / (g2 - g1)
            upper_cutoff = 10**log_fc

    return lower_cutoff, upper_cutoff


def smooth_temporal_signal(data, sampling_rate, target_freq, smoothness_factor=0.02):
    """
    Smooths high-frequency ripple from a time-domain signal using a moving average window.

    Parameters:
    - data: 1D numpy array of your signal
    - sampling_rate: The sampling rate of the file (Hz)
    - target_freq: The square wave frequency of the file (Hz)
    - smoothness_factor: Percentage of one signal cycle to use as the window size.
                         Keep this small (0.01 to 0.05) for square waves to avoid
                         rounding the sharp switching edges.
    """
    # Calculate how many data points make up a single full cycle of your wave
    points_per_cycle = sampling_rate / target_freq

    # Define the window size as a fraction of that cycle
    window_size = int(points_per_cycle * smoothness_factor)
    if window_size < 3:
        window_size = 3  # Minimum window size
    if window_size % 2 == 0:
        window_size += 1  # Ensure the window is odd to maintain alignment

    # Apply a uniform boxcar window convolution
    window = np.ones(window_size) / window_size
    smoothed_data = np.convolve(data, window, mode="same")

    return smoothed_data


# ==============================================================================
# Setup & Data Configuration
# ==============================================================================

freqs = [
    50000,
    100000,
    200000,
    300000,
    400000,
    450000,
    500000,
    550000,
    600000,
    620000,
    650000,
    670000,
    700000,
    800000,
    900000,
    1000000,
    2000000,
    3000000,
    4000000,
    5000000,
    6000000,
    7000000,
]

sampling_rates = [
    25e6,
    25e6,
    50e6,
    125e6,
    125e6,
    125e6,
    125e6,
    125e6,
    125e6,
    125e6,
    125e6,
    125e6,
    125e6,
    125e6,
    250e6,
    250e6,
    500e6,
    500e6,
    500e6,
    500e6,
    500e6,
    500e6,
]

resistencia = 4.7
senyals = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v"]
n = len(senyals)

# ==============================================================================
# Process Signals & Debug
# ==============================================================================

cols = 3
rows = math.ceil(n / cols)

# Time Domain Figure
fig_time, axes_time = plt.subplots(rows, cols, figsize=(18, 4 * rows))
axes_time = axes_time.flatten()

# FFT Debug Figure
fig_fft, axes_fft = plt.subplots(rows, cols, figsize=(18, 4 * rows))
axes_fft = axes_fft.flatten()

system_gains = []

for idx, senyal in enumerate(senyals):
    target_f = freqs[idx]
    print(f"Processing {senyal.upper()} (Target: {target_f/1e3:.0f} kHz)...")

    if not Path(f"test/BW_TX/{senyal}_neta_tx.csv").exists():
        from utils import tx_rx_system

        tx_rx_system.convertir_senyal_osciloscopi(
            senyal_csv=f"test/BW_TX/{senyal}.csv", senyal_neta_csv=f"test/BW_TX/{senyal}_neta.csv", channels=2
        )

    # Load data
    v_in = np.loadtxt(f"test/BW_TX/{senyal}_neta_tx.csv", delimiter=",")
    v_mesurada = np.loadtxt(f"test/BW_TX/{senyal}_neta_rx.csv", delimiter=",")

    # Calculate Output Current (mA)
    i_out = ((v_in - v_mesurada) / resistencia) * 1000

    # Smooth out high-frequency ripple (no effect on the fundamental frequency)
    i_out = smooth_temporal_signal(i_out, sampling_rates[idx], target_f, smoothness_factor=0.035)

    # Peak detection in the frequency domain
    amp_out, freq_out, freqs_fft_out, spec_out = extract_fundamental_amplitude(i_out, sampling_rates[idx], target_f)
    # [!] V_in should be V_gate, not V_cc. But we have only 2 channels in the oscilloscope. Removing this for now.
    # amp_in, freq_in, freqs_fft_in, spec_in = extract_fundamental_amplitude(v_in, sampling_rates[idx], target_f)

    # The frequency resolution (spacing between discrete frequency steps in FFT)
    freq_res = sampling_rates[idx] / len(v_in) / 1e3  # kHz

    # Calculate error percentages
    err_out = abs(freq_out - target_f) / target_f
    if err_out > 0.02:
        print(f"Freq mismatch! Target: {target_f/1e3} kHz | I_out: {freq_out} Hz | Resolution: {freq_res} kHz")

    # # Old Gain: how effectively a modulation on the V_gate line translates into an AC current through the LED at a given frequency f.
    # gain = amp_out / amp_in if amp_in > 0 else 0
    # If no V_gate is available, Gain is simply the raw amplitude of output current
    gain = amp_out

    system_gains.append(gain)
    print(f"Gain: {gain}")

    # --- TIME DOMAIN PLOTS ---
    t = np.arange(len(i_out)) / sampling_rates[idx] * 1e6  # us

    time_to_show = 10  # us
    t_to_show = t[t <= time_to_show]
    signal_to_show = i_out[t <= time_to_show]

    ax_t = axes_time[idx]
    ax_t.plot(t_to_show, signal_to_show, linewidth=0.8, color="tab:green", label="I (mA)")
    ax_t.set_ylabel("Current (mA)", fontsize=8)
    ax_t.set_title(f"Senyal {senyal.upper()} — {target_f/1e3:.0f} kHz", fontsize=10)
    ax_t.set_xlabel("Time (us)", fontsize=8)
    ax_t.legend(fontsize=7, loc="upper right")
    ax_t.tick_params(labelsize=7)
    ax_t.grid(True, alpha=0.3)

    # # Alternatively plot V_in and V_mesurada
    # ax_t.plot(t[:500], v_in[:500], linewidth=0.8, color="tab:blue", label="V_in")
    # ax_t.plot(t[:500], v_mesurada[:500], linewidth=0.8, color="tab:red", label="V_mesurada")
    # ax_t.set_ylabel("Voltage (V)", fontsize=8)
    # ax_t.set_title(f"Senyal {senyal.upper()} — {target_f/1e3:.0f} kHz", fontsize=10)
    # ax_t.set_xlabel("Time (us)", fontsize=8)
    # ax_t.legend(fontsize=7, loc="upper right")
    # ax_t.tick_params(labelsize=7)
    # ax_t.grid(True, alpha=0.3)

    # --- FFT PLOTS ---
    ax_f = axes_fft[idx]
    # Normalize spectra just for visual comparison
    # spec_in_norm = spec_in / np.max(spec_in) if np.max(spec_in) > 0 else spec_in
    spec_out_norm = spec_out / np.max(spec_out) if np.max(spec_out) > 0 else spec_out

    # ax_f.plot(freqs_fft_in, spec_in_norm, color="tab:blue", alpha=0.7, label="V_in Spectrum")
    ax_f.plot(freqs_fft_out, spec_out_norm, color="tab:green", alpha=0.7, label="I_out Spectrum")

    # Target Frequency Line
    ax_f.axvline(target_f, color="black", linestyle="--", label="Target Freq")

    # Mark found peaks
    # ax_f.plot(freq_in, spec_in_norm[np.where(freqs_fft_in == freq_in)[0][0]], "ro", markersize=6, label="Peak Found")
    ax_f.plot(freq_out, spec_out_norm[np.where(freqs_fft_out == freq_out)[0][0]], "ro", markersize=6)

    # Zoom in around the target frequency
    ax_f.set_xlim(target_f * 0.5, target_f * 1.5)
    ax_f.set_title(
        f"FFT Peak Match: {senyal.upper()} (Target: {target_f/1e3:.0f} kHz, Resolution: {freq_res:.0f} kHz)",
        fontsize=10,
    )
    ax_f.set_xlabel("Frequency (Hz)", fontsize=8)
    ax_f.legend(fontsize=7)
    ax_f.grid(True, alpha=0.3)

# Clean up empty subplots
for i in range(len(senyals), len(axes_time)):
    axes_time[i].set_visible(False)
    axes_fft[i].set_visible(False)

# Save Time Domain Plot
fig_time.suptitle(f"Corrent I_out per cada senyal - Plots tallats a {time_to_show:.0f} us", fontsize=14, y=1.01)
fig_time.tight_layout()
fig_time.savefig("test/test.png", dpi=150, bbox_inches="tight")
plt.close(fig_time)

# Save FFT Debug Plot
fig_fft.suptitle("FFT Peak Detection Debugging", fontsize=14, y=1.01)
fig_fft.tight_layout()
fig_fft.savefig("test/fft_debug.png", dpi=150, bbox_inches="tight")
plt.close(fig_fft)
print("Saved: test/fft_debug.png")

# ==============================================================================
# Calculate System Bandwidth
# ==============================================================================

system_gains = np.array(system_gains)

max_gain = np.max(system_gains)
gains_db = 20 * np.log10(system_gains / max_gain)

f_lower, f_upper = find_bandpass_cutoffs(freqs, gains_db)

print("\n==========================================")
print("  LED DRIVER BANDWIDTH")
print("==========================================")
print(f"  Peak Freq:     {fmt_freq(freqs[np.argmax(gains_db)])}")
print(f"  Lower -3 dB Cutoff (f_L):     {fmt_freq(f_lower)}")
print(f"  Upper -3 dB Cutoff (f_H):     {fmt_freq(f_upper)}")

if f_lower is not None and f_upper is not None:
    system_bw = f_upper - f_lower
    print(f"  Passband Bandwidth (BW): {fmt_freq(system_bw)}")
elif f_upper is not None:
    print(f"  Low-pass Bandwidth:  {fmt_freq(f_upper)}")
else:
    print("  Bandwidth error: Gain didn't drop by -3 dB from peak.")
print("==========================================\n")

fig_bw, ax_bw = plt.subplots(figsize=(9, 5))
freqs_np = np.array(freqs)

ax_bw.semilogx(freqs_np, gains_db, "o-", color="tab:blue", linewidth=1.5, label="Measured Driver Gain")
ax_bw.axhline(-3, color="gray", linestyle=":", alpha=0.8, label="-3 dB Cutoff Line")

if f_lower is not None:
    ax_bw.axvline(f_lower, color="tab:orange", linestyle="--", label=f"f_L Lower Cutoff ({fmt_freq(f_lower)})")
if f_upper is not None:
    ax_bw.axvline(f_upper, color="tab:red", linestyle="--", label=f"f_H Upper Cutoff ({fmt_freq(f_upper)})")

ax_bw.set_title("LED Driver Frequency Response Curve")
ax_bw.set_xlabel("Frequency (Hz)")
ax_bw.set_ylabel("Normalized Gain (dB)")
ax_bw.grid(True, which="both", alpha=0.4)
ax_bw.legend()
plt.tight_layout()
plt.savefig("test/system_frequency_response.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: test/system_frequency_response.png")
