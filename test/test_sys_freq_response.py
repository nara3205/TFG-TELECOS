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

    # Calculate Current (mA)
    signal_Id = ((v_in - v_mesurada) / resistencia) * 1000

    # Peak detection in the frequency domain
    amp_in, freq_in, freqs_fft_in, spec_in = extract_fundamental_amplitude(v_in, sampling_rates[idx], target_f)
    amp_out, freq_out, freqs_fft_out, spec_out = extract_fundamental_amplitude(signal_Id, sampling_rates[idx], target_f)

    # The frequency resolution (spacing between discrete frequency steps in your FFT)
    freq_resolution = sampling_rates[idx] / len(v_in) / 1e3  # kHz

    # Calculate error percentages
    err_in = abs(freq_in - target_f) / target_f
    err_out = abs(freq_out - target_f) / target_f
    if err_in > 0.02 or err_out > 0.02:
        print(
            f"Frequency mismatch! Target: {target_f} Hz | Found V_in: {freq_in:.0f} Hz, I_out: {freq_out:.0f} Hz, [!] Resolution: {freq_resolution:.0f} kHz"
        )

    # Gain: how effectively a voltage ripple or modulation on the V_CC line translates into an AC current ripple through the LEDs at a given frequency f.
    gain = amp_out / amp_in if amp_in > 0 else 0
    system_gains.append(gain)

    # --- TIME DOMAIN PLOTS ---
    t = np.arange(len(signal_Id)) / sampling_rates[idx] * 1e6
    ax_t = axes_time[idx]
    ax_t.plot(t[:500], signal_Id[:500], linewidth=0.8, color="tab:green", label="I (mA)")
    ax_t.set_ylabel("Current (mA)", fontsize=8)
    ax_t.set_title(f"Senyal {senyal.upper()} — {target_f/1e3:.0f} kHz", fontsize=10)
    ax_t.set_xlabel("Time (us)", fontsize=8)
    ax_t.legend(fontsize=7, loc="upper right")
    ax_t.tick_params(labelsize=7)
    ax_t.grid(True, alpha=0.3)

    # --- FFT PLOTS ---
    ax_f = axes_fft[idx]
    # Normalize spectra just for visual comparison
    spec_in_norm = spec_in / np.max(spec_in) if np.max(spec_in) > 0 else spec_in
    spec_out_norm = spec_out / np.max(spec_out) if np.max(spec_out) > 0 else spec_out

    ax_f.plot(freqs_fft_in, spec_in_norm, color="tab:blue", alpha=0.7, label="V_in Spectrum")
    ax_f.plot(freqs_fft_out, spec_out_norm, color="tab:green", alpha=0.7, label="I_out Spectrum")

    # Target Frequency Line
    ax_f.axvline(target_f, color="black", linestyle="--", label="Target Freq")

    # Mark found peaks
    ax_f.plot(freq_in, spec_in_norm[np.where(freqs_fft_in == freq_in)[0][0]], "ro", markersize=6, label="Peak Found")
    ax_f.plot(freq_out, spec_out_norm[np.where(freqs_fft_out == freq_out)[0][0]], "ro", markersize=6)

    # Zoom in around the target frequency
    ax_f.set_xlim(target_f * 0.5, target_f * 1.5)
    ax_f.set_title(
        f"FFT Peak Match: {senyal.upper()} (Target: {target_f/1e3:.0f} kHz, Resolution: {freq_resolution:.0f} kHz)",
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
fig_time.suptitle("Corrent I per cada senyal", fontsize=14, y=1.01)
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
