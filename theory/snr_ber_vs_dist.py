import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogLocator, MultipleLocator
from scipy.special import erfc

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "text.latex.preamble": r"\usepackage{amsmath}",
    }
)

# ==============================================================================
# 1. System Parameters & Data Generation
# ==============================================================================
q = 1.602e-19  # Electron charge (C)
B = 14000  # Receiver Bandwidth (Hz)
sigma_floor_sq = 9.00e-20  # Noise floor variance (300 pA)^2
print("[!!!] Only valid for low-pass filter that caps the bandwidth at 14 kHz")

d_max = 10.80  # Critical cutoff distance (m)

# Generate distance array (meters)
distance = np.linspace(0.05, 20.0, 2000)

# Calculate analytical Photocurrent profile
I_ph = 1.530e-7 / (distance**2)

# Compute Noise and Electrical SNR
sigma_shot_sq = 2 * q * I_ph * B
sigma_noise_sq = sigma_shot_sq + sigma_floor_sq
snr_linear = (I_ph**2) / sigma_noise_sq
snr_db = 10 * np.log10(snr_linear)

# Theoretical OOK BER using the complementary error function
ber = 0.5 * erfc(np.sqrt(snr_linear) / 2)
ber_clipped = np.clip(ber, 1e-14, 0.5)

# Extract specific values at the exact boundary point for annotations
I_ph_target = 1.530e-7 / (d_max**2)
sigma_noise_target = 2 * q * I_ph_target * B + sigma_floor_sq
snr_target_lin = (I_ph_target**2) / sigma_noise_target
snr_target_db = 10 * np.log10(snr_target_lin)
ber_target = 0.5 * erfc(np.sqrt(snr_target_lin) / 2)

# ==============================================================================
# 2. Plotting the Subplots (1 Row, 2 Columns)
# ==============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14 / 1.5, 5.5 / 1.5), dpi=300, sharex=True)

# ------------------------------------------------------------------------------
# Left Subplot (a): SNR vs Distance
# ------------------------------------------------------------------------------
ax1.plot(distance, snr_db, color="tab:blue", linewidth=2.5, label="SNR elèctrica")
ax1.axvline(x=d_max, color="black", linestyle=":", alpha=0.7, linewidth=1.2)
ax1.plot(d_max, snr_target_db, "o", color="tab:blue", markersize=7, markeredgecolor="black")

# Annotations for SNR
ax1.annotate(
    f"SNR = {snr_target_db:.2f} dB",
    xy=(d_max, snr_target_db),
    xytext=(d_max - 3.2, snr_target_db + 10),
    arrowprops=dict(arrowstyle="->", color="tab:blue", lw=0.8),
    fontsize=9,
    color="tab:blue",
    weight="bold",
)

ax1.text(d_max + 0.3, 85, f"$d_{{max}} = {d_max:.2f}$ m", color="black", fontsize=9)

# Styling Left Axis
ax1.set_xlabel("Distància de l'enllaç, $d$ (metres)", fontsize=10)
ax1.set_ylabel("SNR (dB)", fontsize=10)
ax1.set_xlim(0, 20)
ax1.set_ylim(0, 100)
ax1.grid(True, which="both", linestyle=":", alpha=0.5)

# ------------------------------------------------------------------------------
# Right Subplot (b): BER vs Distance
# ------------------------------------------------------------------------------
ax2.plot(distance, ber_clipped, color="tab:red", linewidth=2.5, linestyle="-", label="BER teòrica")
ax2.axvline(x=d_max, color="black", linestyle=":", alpha=0.7, linewidth=1.2)
ax2.plot(d_max, ber_target, "s", color="tab:red", markersize=6, markeredgecolor="black")

# Annotations for BER
ax2.annotate(
    f"Límit\nBER = $10^{{-3}}$",
    xy=(d_max, ber_target),
    xytext=(d_max - 3.5, ber_target * 15),
    arrowprops=dict(arrowstyle="->", lw=0.8),  # , color="tab:red"
    fontsize=9,
    # color="tab:red",
    # weight="bold",
)

ax2.text(d_max + 0.3, 1e-7, f"$d_{{max}} = {d_max:.2f}$ m", color="black", fontsize=9)

# Styling Right Axis
ax2.set_xlabel("Distància de l'enllaç, $d$ (metres)", fontsize=10)
ax2.set_ylabel("BER", fontsize=10)
ax2.set_yscale("log")
ax2.set_xlim(0, 20)
ax2.set_ylim(1e-12, 1)
ax2.grid(True, which="both", linestyle=":", alpha=0.5)

# --- Horizontal X-Axis (Distance in meters) ---
ax1.xaxis.set_major_locator(MultipleLocator(1.0))  # Major tick every 1 meter
ax1.xaxis.set_minor_locator(MultipleLocator(0.5))  # Minor tick every 0.5 meters

# --- Left Vertical Y-Axis (SNR in dB) ---
ax1.yaxis.set_major_locator(MultipleLocator(10.0))  # Major tick every 10 dB
ax1.yaxis.set_minor_locator(MultipleLocator(5.0))  # Minor tick every 1 dB

# --- Right Vertical Y-Axis (BER Log Scale) ---
ax2.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,), numticks=15))
ax2.yaxis.set_minor_locator(LogLocator(base=10.0, subs="auto", numticks=15))

# ==============================================================================
# 3. Save and Show Final Figure
# ==============================================================================
plt.suptitle("Rendiment del Canal IM/DD OOK (OPT101 BW=14 kHz, Obscuritat)", fontsize=13, weight="bold", y=0.98)
plt.tight_layout()

# Save as clean high-resolution file
plt.savefig("theory/vlc_snr_ber_analysis.png", dpi=300, bbox_inches="tight")
print("Saved figure successfully to theory/vlc_snr_ber_analysis.png")
plt.close()
