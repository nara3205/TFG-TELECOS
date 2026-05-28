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
    y = np.asarray(y) - np.mean(y)
    window = np.hanning(len(y))
    spectrum = np.fft.rfft(y * window)
    spectrum_mag = np.abs(spectrum)
    freqs_fft = np.fft.rfftfreq(len(y), d=1 / fs)

    freq_res = fs / len(y)
    tolerance = max(0.15, (2 * freq_res) / target_freq)
    lower_bound = target_freq * (1 - tolerance)
    upper_bound = target_freq * (1 + tolerance)
    mask = (freqs_fft >= lower_bound) & (freqs_fft <= upper_bound)

    if np.any(mask):
        mask_indices = np.where(mask)[0]
        peak_idx_in_mask = np.argmax(spectrum_mag[mask])
        peak_idx = mask_indices[peak_idx_in_mask]
    else:
        peak_idx = np.argmin(np.abs(freqs_fft - target_freq))

    spectrum_mag = spectrum_mag * (4.0 / len(y))

    return spectrum_mag[peak_idx], freqs_fft[peak_idx], freqs_fft, spectrum_mag


def find_bandpass_cutoffs(freqs, gains_db):
    pk_idx = np.argmax(gains_db)
    below_3db = np.where(gains_db <= -3.0)[0]

    lower_cutoff = None
    upper_cutoff = None

    lower_idxs = below_3db[below_3db < pk_idx]
    if len(lower_idxs) > 0:
        i = lower_idxs[-1]
        if i < len(freqs) - 1:
            f1, f2 = freqs[i], freqs[i + 1]
            g1, g2 = gains_db[i], gains_db[i + 1]
            log_f1, log_f2 = np.log10(f1), np.log10(f2)
            log_fc = log_f1 + (-3.0 - g1) * (log_f2 - log_f1) / (g2 - g1)
            lower_cutoff = 10**log_fc

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
    points_per_cycle = sampling_rate / target_freq
    window_size = int(points_per_cycle * smoothness_factor)
    if window_size < 3:
        window_size = 3
    if window_size % 2 == 0:
        window_size += 1
    window = np.ones(window_size) / window_size
    smoothed_data = np.convolve(data, window, mode="same")
    return smoothed_data


def process_dataset(freqs, sampling_rates, senyals, folder, resistencia, time_to_show=10):
    n = len(senyals)
    cols = 3
    rows = math.ceil(n / cols)

    fig_time, axes_time = plt.subplots(rows, cols, figsize=(18, 4 * rows))
    axes_time = axes_time.flatten()
    fig_fft, axes_fft = plt.subplots(rows, cols, figsize=(18, 4 * rows))
    axes_fft = axes_fft.flatten()

    system_gains = []

    for idx, senyal in enumerate(senyals):
        target_f = freqs[idx]
        fs = sampling_rates[idx]
        print(f"Processing {senyal.upper()} (Target: {fmt_freq(target_f)})...")

        if not Path(f"fitxers/{folder}/{senyal}_neta_tx.csv").exists():
            from utils import tx_rx_system
            tx_rx_system.convertir_senyal_osciloscopi(
                senyal_csv=f"fitxers/{folder}/{senyal}.csv",
                senyal_neta_csv=f"fitxers/{folder}/{senyal}_neta.csv",
                channels=2
            )

        v_in       = np.loadtxt(f"fitxers/{folder}/{senyal}_neta_tx.csv", delimiter=",")
        v_mesurada = np.loadtxt(f"fitxers/{folder}/{senyal}_neta_rx.csv", delimiter=",")

        i_out = ((v_in - v_mesurada) / resistencia) * 1000
        i_out = smooth_temporal_signal(i_out, fs, target_f, smoothness_factor=0.035)

        amp_out, freq_out, freqs_fft_out, spec_out = extract_fundamental_amplitude(i_out, fs, target_f)

        freq_res = fs / len(v_in) / 1e3  # kHz
        err_out = abs(freq_out - target_f) / target_f
        if err_out > 0.15:
            print(f"  [WARN] Freq mismatch! Target: {fmt_freq(target_f)} | Found: {fmt_freq(freq_out)} | Res: {freq_res:.3f} kHz")

        gain = amp_out
        system_gains.append(gain)
        print(f"  Amplitude: {gain:.4f} mA  |  Found at: {fmt_freq(freq_out)}")

        # Time domain plot
        t = np.arange(len(i_out)) / fs * 1e6


        cycles_to_show = 10
        samples_to_show = min(int(cycles_to_show * fs / target_f), len(i_out))
        signal_to_show = i_out[:samples_to_show]
        t_to_show      = t[:samples_to_show]

        ax_t = axes_time[idx]
        ax_t.plot(t_to_show, signal_to_show, linewidth=0.8, color="tab:green", label="I (mA)")
        ax_t.set_ylabel("Current (mA)", fontsize=8)
        ax_t.set_title(f"Senyal {senyal.upper()} — {fmt_freq(target_f)}", fontsize=10)
        ax_t.set_xlabel("Time (us)", fontsize=8)
        ax_t.legend(fontsize=7, loc="upper right")
        ax_t.tick_params(labelsize=7)
        ax_t.grid(True, alpha=0.3)

        # FFT plot
        spec_out_norm = spec_out / np.max(spec_out) if np.max(spec_out) > 0 else spec_out

        ax_f = axes_fft[idx]
        ax_f.plot(freqs_fft_out, spec_out_norm, color="tab:green", alpha=0.7, label="I_out Spectrum")
        ax_f.axvline(target_f, color="black", linestyle="--", label="Target Freq")
        ax_f.plot(freq_out, spec_out_norm[np.where(freqs_fft_out == freq_out)[0][0]], "ro", markersize=6, label="Peak Found")
        ax_f.set_xlim(max(0, target_f * 0.5), target_f * 1.5)
        ax_f.set_title(f"FFT: {senyal.upper()} ({fmt_freq(target_f)}, Res: {freq_res:.3f} kHz)", fontsize=10)
        ax_f.set_xlabel("Frequency (Hz)", fontsize=8)
        ax_f.legend(fontsize=7)
        ax_f.grid(True, alpha=0.3)

    for i in range(n, len(axes_time)):
        axes_time[i].set_visible(False)
        axes_fft[i].set_visible(False)

    fig_time.suptitle(f"Corrent I_out — {folder} (tallat a {time_to_show} us)", fontsize=14, y=1.01)
    fig_time.tight_layout()
    fig_time.savefig(f"fitxers/{folder}/time_domain.png", dpi=150, bbox_inches="tight")
    plt.close(fig_time)

    fig_fft.suptitle(f"FFT Peak Detection — {folder}", fontsize=14, y=1.01)
    fig_fft.tight_layout()
    fig_fft.savefig(f"fitxers/{folder}/fft_debug.png", dpi=150, bbox_inches="tight")
    plt.close(fig_fft)
    print(f"Saved: fitxers/{folder}/fft_debug.png")

    # Bandwidth calculation
    system_gains = np.array(system_gains)
    max_gain = np.max(system_gains)
    gains_db = 20 * np.log10(system_gains / max_gain)

    f_lower, f_upper = find_bandpass_cutoffs(freqs, gains_db)

    print("\n==========================================")
    print(f"  LED DRIVER BANDWIDTH — {folder}")
    print("==========================================")
    print(f"  Peak Freq:               {fmt_freq(freqs[np.argmax(gains_db)])}")
    print(f"  Lower -3 dB Cutoff (f_L): {fmt_freq(f_lower)}")
    print(f"  Upper -3 dB Cutoff (f_H): {fmt_freq(f_upper)}")
    if f_lower is not None and f_upper is not None:
        print(f"  Passband Bandwidth (BW):  {fmt_freq(f_upper - f_lower)}")
    elif f_upper is not None:
        print(f"  Low-pass Bandwidth:       {fmt_freq(f_upper)}")
    else:
        print("  Bandwidth error: Gain didn't drop by -3 dB from peak.")
    print("==========================================\n")

    fig_bw, ax_bw = plt.subplots(figsize=(9, 5))
    ax_bw.semilogx(np.array(freqs), gains_db, "o-", color="tab:blue", linewidth=1.5, label="Measured Driver Gain")
    ax_bw.axhline(-3, color="gray", linestyle=":", alpha=0.8, label="-3 dB Cutoff Line")
    if f_lower is not None:
        ax_bw.axvline(f_lower, color="tab:orange", linestyle="--", label=f"f_L ({fmt_freq(f_lower)})")
    if f_upper is not None:
        ax_bw.axvline(f_upper, color="tab:red", linestyle="--", label=f"f_H ({fmt_freq(f_upper)})")
    ax_bw.set_title(f"LED Driver Frequency Response — {folder}")
    ax_bw.set_xlabel("Frequency (Hz)")
    ax_bw.set_ylabel("Normalized Gain (dB)")
    ax_bw.grid(True, which="both", alpha=0.4)
    ax_bw.legend()
    plt.tight_layout()
    plt.savefig(f"fitxers/{folder}/system_frequency_response.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: fitxers/{folder}/system_frequency_response.png")


# ==============================================================================
# BW_TX_v1
# ==============================================================================
process_dataset(
    freqs=[
        50000, 100000, 200000, 300000, 400000, 450000, 500000, 550000,
        600000, 620000, 650000, 670000, 700000, 800000, 900000, 1000000,
        2000000, 3000000, 4000000, 5000000, 6000000, 7000000,
    ],
    sampling_rates=[
        25e6, 25e6, 50e6, 125e6, 125e6, 125e6, 125e6, 125e6,
        125e6, 125e6, 125e6, 125e6, 125e6, 125e6, 250e6, 250e6,
        500e6, 500e6, 500e6, 500e6, 500e6, 500e6,
    ],
    senyals=["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v"],
    folder="BW_TX_v1",
    resistencia=4.7,
)

# ==============================================================================
# BW_TX_v2
# ==============================================================================
process_dataset(
    freqs=[
        1, 2, 5, 10, 20, 50, 100, 200, 500,
        1e3, 2e3, 5e3, 10e3, 20e3, 50e3, 100e3, 200e3, 500e3,
        1e6, 2e6, 3e6, 4e6, 5e6, 6e6, 7e6,
    ],
    sampling_rates=[
        25e3, 50e3, 125e3, 250e3, 500e3, 1.25e6, 2.5e6, 5e6, 12.5e6,
        25e6, 50e6, 125e6, 125e6, 125e6, 125e6, 125e6, 125e6, 125e6,
        125e6, 125e6, 125e6, 125e6, 125e6, 125e6, 125e6,
    ],
    senyals=["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y"],
    folder="BW_TX_v2",
    resistencia=4.7,
)