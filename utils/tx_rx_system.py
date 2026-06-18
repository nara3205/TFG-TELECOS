import math
import subprocess
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


N_BITS = 4096

def executar_comanda(comanda):
    """Funció aux per cridar scripts"""
    try:
        print(f"Executing: {comanda}")
        subprocess.run(comanda, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error executant la comanda: {e}")

def codificar(mode=None, message=None, num_repeats=None):
    if mode is None:
        mode = input("Escull mode (OOK / MANCHESTER): ").strip().upper()
    if message is None:
        message = input("Escriu el missatge per enviar: ")
    if num_repeats is None:
        num_repeats = int(input("Quants cops vols repetir-lo?: "))

    start_sequence = "10101010"
    preamble = format(len(message), "08b")
    message_bits = "".join(format(ord(c), "08b") for c in message)

    frame_bits = start_sequence + preamble + message_bits

    raw_bits = ""
    for _ in range(num_repeats):
        raw_bits += frame_bits + "00000000"

    if mode == "MANCHESTER":
        tx_bits_str = "".join(["10" if b == "1" else "01" for b in raw_bits])
    else:
        tx_bits_str = raw_bits

    tx_signal = np.zeros(N_BITS)
    signal_data = np.array([1 if b == "1" else 0.0 for b in tx_bits_str])

    if len(signal_data) > N_BITS:
        print(f"El senyal ({len(signal_data)}) excedeix les {N_BITS} mostres")

    tx_signal[: min(len(signal_data), N_BITS)] = signal_data[:N_BITS]

    nom_csv = "fitxers/senyal.csv"
    np.savetxt(nom_csv, tx_signal, delimiter=",", fmt="%.2f")
    print(f"Senyal guardat a {nom_csv}")
    executar_comanda(f"python3 utils/csv2arb.py {nom_csv}")

def _calculate_ber_v2(tx_bits, rx_bits, mode="OOK"):

    tx_bits = np.array(tx_bits, dtype=int)
    rx_bits = np.array(rx_bits, dtype=int)
    n_tx = len(tx_bits)
    n_rx = len(rx_bits)
    if mode=="OOK":
        if n_rx < n_tx:
            compare_len = n_rx
            errors = int(np.sum(tx_bits[:compare_len] != rx_bits[:compare_len]))
            return errors / compare_len, errors, compare_len

        tx_bipolar = 2 * tx_bits - 1
        rx_bipolar = 2 * rx_bits - 1
        correlation = np.correlate(rx_bipolar, tx_bipolar, mode="valid")
        best_offset = int(np.argmax(correlation))
        total_errors = int(np.sum(tx_bits != rx_bits[best_offset : best_offset + n_tx]))
    
    else: 
        total_errors = 0
        min_zero_run = int(100)
        i = 0
        i_saved = 0
        zero_run = 0
        sync_found = False

        while i < len(rx_bits):
            if rx_bits[i] == 0:
                zero_run += 1
            else:
                if zero_run >= min_zero_run:
                    sync_found = True
                    i_saved=i
                    break
                zero_run = 0
            i += 1

        if not sync_found:
            return np.array([])

        end = min(i + n_tx, n_rx)
        results = []
        while i + 1 < end:
            c1, c2 = rx_bits[i], rx_bits[i + 1]
            if c1 == 1 and c2 == 1 and i==i_saved:
                i -= 1
                continue
            elif   c1 == 1 and c2 == 0:
                results.append(1)
            elif c1 == 0 and c2 == 1:
                results.append(0)
            else:
                total_errors +=1

            i += 2

    
    return total_errors / n_tx, total_errors, n_tx

def _calculate_ber(tx_bits, rx_bits):
    """
    Calcula BER usant correlació per trobar el millor offset d'alineació.
    """
    # Cast to signed int before bipolar arithmetic (2*x-1) to avoid uint8 wrap-around
    # when the caller passes e.g. rng.integers(..., dtype=np.uint8).
    tx_bits = np.array(tx_bits, dtype=int)
    rx_bits = np.array(rx_bits, dtype=int)
    n_tx = len(tx_bits)
    n_rx = len(rx_bits)

    if n_rx < n_tx:
        compare_len = n_rx
        errors = int(np.sum(tx_bits[:compare_len] != rx_bits[:compare_len]))
        return errors / compare_len, errors, compare_len

    # --- Correlació bipolar per trobar el millor offset ---
    tx_bipolar = 2 * tx_bits - 1
    rx_bipolar = 2 * rx_bits - 1

    correlation = np.correlate(rx_bipolar, tx_bipolar, mode="valid")
    # El pic de correlació indica on el TX s'assembla més al RX
    best_offset = int(np.argmax(correlation))

    # --- Compara TOTS els bits rebuts usant l'alineació trobada ---
    # best_offset is the phase of the PRBS at rx_bits[0]: the reference bit
    # for rx position i is prbs_gold[(i + n_tx - best_offset%n_tx) % n_tx].
    # This single modular-index expression covers bits before *and* after
    # best_offset (the AWG loops continuously, so the pre-offset tail is
    # just the end of the previous PRBS period).
    phase = best_offset % n_tx
    ref = tx_bits[(np.arange(n_rx) + n_tx - phase) % n_tx]

    total_errors = int(np.sum(ref != rx_bits))
    total_bits   = n_rx

    return total_errors / total_bits, total_errors, total_bits

def calculate_snr(tx_bits, analog_samples):
    """
    Calcula la SNR analógica a partir de los voltajes recibidos en el centro del bit.
    Normaliza la señal analógica para mapear los voltajes al rango [0, 1] 
    y calcula la potencia del ruido real (varianza).
    """
    tx_bits = np.array(tx_bits, dtype=int)
    analog_samples = np.array(analog_samples, dtype=float)
    n_tx = len(tx_bits)
    
    v_min = np.min(analog_samples)
    v_max = np.max(analog_samples)
    
    if v_max == v_min:
        return -99.0, 0.0, n_tx, n_tx
        
    analog_norm = (analog_samples - v_min) / (v_max - v_min)
    
    rx_bits_hard = (analog_norm > 0.5).astype(int)
    tx_bipolar = 2 * tx_bits - 1
    rx_bipolar = 2 * rx_bits_hard - 1
    
    correlation = np.correlate(rx_bipolar, tx_bipolar, mode="valid")
    best_offset = int(np.argmax(correlation))

    rx_bits_aligned = rx_bits_hard[best_offset : best_offset + n_tx]
    analog_norm_aligned = analog_norm[best_offset : best_offset + n_tx]
    
    error_signal = analog_norm_aligned - tx_bits  
    
    bit_errors = int(np.sum(tx_bits != rx_bits_aligned))
    
    power_signal = np.mean(tx_bits ** 2)
    
    power_noise = np.mean(error_signal ** 2)
        
    snr_linear = power_signal / power_noise
    snr_db = 10 * np.log10(snr_linear)
    
    return snr_db, snr_linear, bit_errors, n_tx

def binaritzar(rx_signal):
    v_max, v_min = np.max(rx_signal), np.min(rx_signal)
    threshold = (v_max + v_min) / 2
    return (rx_signal > threshold).astype(int)

def convertir_senyal_osciloscopi(senyal_csv=None, senyal_neta_csv=None, channels=1):

    if channels == 1:
        #df = pd.read_csv(senyal_csv, usecols=[1], header=None)
        df = pd.read_csv(senyal_csv, usecols=[1], skiprows=1, header=None)  
        df_net = pd.to_numeric(df.iloc[:, 0], errors="coerce").dropna()
        df_net.to_csv(senyal_neta_csv, index=False, header=False, float_format="%.10f")

    elif channels == 2:
        df = pd.read_csv(senyal_csv, usecols=[1, 2], header=None, skiprows=1)
        ch1 = pd.to_numeric(df.iloc[:, 0], errors="coerce")
        ch2 = pd.to_numeric(df.iloc[:, 1], errors="coerce")
        df_net = pd.concat([ch1, ch2], axis=1).dropna()
        base = senyal_neta_csv.replace(".csv", "")
        df_net.iloc[:, 0].to_csv(base + "_rx.csv", index=False, header=False, float_format="%.10f")
        df_net.iloc[:, 1].to_csv(base + "_tx.csv", index=False, header=False, float_format="%.10f")
        print(f"Guardat: {base}_rx.csv i {base}_tx.csv")
    else:
        raise ValueError("channels ha de ser 1 o 2")

def convertir_senyal_osciloscopi_bin(senyal_csv=None, senyal_neta_csv=None, channels=1):

    if channels == 1:
        df = pd.read_csv(senyal_csv, usecols=[1], header=None)
        df_net = pd.to_numeric(df.iloc[:, 0], errors="coerce").dropna()
        llindar = (df_net.max() + df_net.min()) / 2
        df_bits = df_net.apply(lambda x: 1 if x > llindar else 0)
        df_bits.to_csv(senyal_neta_csv, index=False, header=False)

    elif channels == 2:
        df = pd.read_csv(senyal_csv, usecols=[1, 2], header=None)
        ch1 = pd.to_numeric(df.iloc[:, 0], errors="coerce")
        ch2 = pd.to_numeric(df.iloc[:, 1], errors="coerce")
        df_net = pd.concat([ch1, ch2], axis=1).dropna()
        llindar_ch1 = (df_net.iloc[:, 0].max() + df_net.iloc[:, 0].min()) / 2
        llindar_ch2 = (df_net.iloc[:, 1].max() + df_net.iloc[:, 1].min()) / 2
        ch1_bits = df_net.iloc[:, 0].apply(lambda x: 1 if x > llindar_ch1 else 0)
        ch2_bits = df_net.iloc[:, 1].apply(lambda x: 1 if x > llindar_ch2 else 0)
        base = senyal_neta_csv.replace(".csv", "")
        ch1_bits.to_csv(base + "_rx.csv", index=False, header=False)
        ch2_bits.to_csv(base + "_tx.csv", index=False, header=False)
        print(f"Guardat: {base}_rx.csv i {base}_tx.csv")
    else:
        raise ValueError("channels ha de ser 1 o 2")

def generate_prbs(seed: int = 42, nom_csv="fitxers/prbs.csv", mode="OOK", size = N_BITS) -> np.ndarray:
    rng = np.random.default_rng(seed=seed)

    if mode == "OOK":
        bits = rng.integers(0, 2, size=size, dtype=np.uint8)
        np.savetxt(nom_csv, bits, delimiter=",", fmt="%.2f")
        print(f"Saved OOK signal to {nom_csv}")
        executar_comanda(f"python3 utils/csv2arb.py {nom_csv}")
        return bits

    elif mode == "MANCHESTER":
        bits = rng.integers(0, 2, size=size // 2, dtype=np.uint8)

        # Save raw bits before encoding
        base = nom_csv.replace(".csv", "")
        raw_csv = f"{base}_raw.csv"
        np.savetxt(raw_csv, bits, delimiter=",", fmt="%.2f")
        print(f"Saved raw bits (pre-Manchester) to {raw_csv}")

        # Encode: 1 → [1, 0], 0 → [0, 1]
        manchester_bits = np.empty(len(bits) * 2, dtype=np.uint8)
        manchester_bits[0::2] = bits
        manchester_bits[1::2] = 1 - bits

        if size < N_BITS:
            manchester_bits.resize(N_BITS, refcheck=False)  

        # Save encoded signal and send to AWG
        np.savetxt(nom_csv, manchester_bits, delimiter=",", fmt="%.2f")
        print(f"Saved Manchester signal ({len(manchester_bits)} samples) to {nom_csv}")
        executar_comanda(f"python3 utils/csv2arb.py {nom_csv}")

    else:
        raise ValueError(f"Unknown type '{type}'. Use 'ook' or 'manchester'.")
   
def _decode_bits_from_signal(rx_signal, bits_per_sample):
    rx_bits_raw = binaritzar(rx_signal)

    flancs = [i for i in range(1, len(rx_bits_raw)) if rx_bits_raw[i] != rx_bits_raw[i - 1]]
    flancs.append(len(rx_bits_raw))
    flancs = np.array(flancs)

    if len(flancs) < 2:
        return np.array([])

    rx_bits = []
    for i in range(len(flancs) - 1):
        durada = flancs[i + 1] - flancs[i]
        n_bits = max(1, int(round(durada / bits_per_sample)))
        # Sample in the middle of the segment to get a stable value
        sample_idx = min(flancs[i] + int(bits_per_sample / 2), len(rx_bits_raw) - 1)
        valor = rx_bits_raw[sample_idx]
        rx_bits.extend([valor] * n_bits)

    return np.array(rx_bits)

def _decode_raw_from_signal(rx_signal, bits_per_sample):
    rx_bits_raw = binaritzar(rx_signal)
    flancs = [i for i in range(1, len(rx_bits_raw)) if rx_bits_raw[i] != rx_bits_raw[i - 1]]
    flancs.append(len(rx_bits_raw))
    flancs = np.array(flancs)

    if len(flancs) < 2:
        return np.array([])

    rx_analog_samples = []
    for i in range(len(flancs) - 1):
        durada = flancs[i + 1] - flancs[i]
        n_bits = max(1, int(round(durada / bits_per_sample)))
        
        # En lugar de hacer un .extend del mismo valor, remuestreamos 
        # el centro de cada bit real dentro de este pulso ancho
        for b in range(n_bits):
            offset_centro = int((b + 0.5) * bits_per_sample)
            sample_idx = min(flancs[i] + offset_centro, len(rx_signal) - 1)
            
            # Guardamos el voltaje analógico real de cada bit individual
            rx_analog_samples.append(rx_signal[sample_idx])

    return np.array(rx_analog_samples)

def experiment1_BODE(freqs, sampling_rates, senyals, folder, resistencia, time_to_show=10):

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

        convertir_senyal_osciloscopi(
            senyal_csv=f"fitxers/{folder}/{senyal}.csv",
            senyal_neta_csv=f"fitxers/{folder}/{senyal}_neta.csv",
            channels=2,
        )

        v_in = np.loadtxt(f"fitxers/{folder}/{senyal}_neta_tx.csv", delimiter=",")
        v_mesurada = np.loadtxt(f"fitxers/{folder}/{senyal}_neta_rx.csv", delimiter=",")

        i_out = ((v_in - v_mesurada) / resistencia) * 1000
        i_out = smooth_temporal_signal(i_out, fs, target_f, smoothness_factor=0.035)

        amp_out, freq_out, freqs_fft_out, spec_out = extract_fundamental_amplitude(i_out, fs, target_f)

        freq_res = fs / len(v_in) / 1e3  # kHz
        err_out = abs(freq_out - target_f) / target_f
        if err_out > 0.15:
            print(
                f"  [WARN] Freq mismatch! Target: {fmt_freq(target_f)} | Found: {fmt_freq(freq_out)} | Res: {freq_res:.3f} kHz"
            )

        gain = amp_out
        system_gains.append(gain)
        print(f"  Amplitude: {gain:.4f} mA  |  Found at: {fmt_freq(freq_out)}")

        # Time domain plot
        t = np.arange(len(i_out)) / fs * 1e6

        cycles_to_show = 10
        samples_to_show = min(int(cycles_to_show * fs / target_f), len(i_out))
        signal_to_show = i_out[:samples_to_show]
        t_to_show = t[:samples_to_show]

        ax_t = axes_time[idx]
        ax_t.plot(t_to_show, signal_to_show, linewidth=0.8, color="tab:green", label="I (mA)")
        ax_t.set_ylabel("Corrent (mA)", fontsize=8)
        ax_t.set_title(f"Senyal {fmt_freq(target_f)}", fontsize=10)
        ax_t.set_xlabel("Temps (us)", fontsize=8)
        ax_t.legend(fontsize=7, loc="upper right")
        ax_t.tick_params(labelsize=7)
        ax_t.grid(True, alpha=0.3)

        # FFT plot
        spec_out_norm = spec_out / np.max(spec_out) if np.max(spec_out) > 0 else spec_out

        ax_f = axes_fft[idx]
        ax_f.plot(freqs_fft_out, spec_out_norm, color="tab:green", alpha=0.7, label="I_out Spectrum")
        ax_f.axvline(target_f, color="black", linestyle="--", label="Target Freq")
        ax_f.plot(
            freq_out, spec_out_norm[np.where(freqs_fft_out == freq_out)[0][0]], "ro", markersize=6, label="Peak Found"
        )
        ax_f.set_xlim(max(0, target_f * 0.5), target_f * 1.5)
        ax_f.set_title(f"FFT: {senyal.upper()} ({fmt_freq(target_f)}, Res: {freq_res:.3f} kHz)", fontsize=10)
        ax_f.set_xlabel("Frequency (Hz)", fontsize=8)
        ax_f.legend(fontsize=7)
        ax_f.grid(True, alpha=0.3)

    for i in range(n, len(axes_time)):
        axes_time[i].set_visible(False)
        axes_fft[i].set_visible(False)

    fig_time.tight_layout()
    fig_time.savefig(f"fitxers/{folder}/time_domain.png", dpi=150, bbox_inches="tight")
    plt.show()
    # plt.close(fig_time)

    fig_fft.suptitle(f"FFT Peak Detection — {folder}", fontsize=14, y=1.01)
    fig_fft.tight_layout()
    fig_fft.savefig(f"fitxers/{folder}/fft_debug.png", dpi=150, bbox_inches="tight")
    plt.show()
    # plt.close(fig_fft)
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
    ax_bw.semilogx(np.array(freqs), gains_db, "o-", color="tab:blue", linewidth=1.5)
    ax_bw.axhline(-3, color="gray", linestyle=":", alpha=0.8, label="-3dB")
    if f_lower is not None:
        ax_bw.axvline(f_lower, color="tab:orange", linestyle="--", label=f"f_L ({fmt_freq(f_lower)})")
    if f_upper is not None:
        ax_bw.axvline(f_upper, color="tab:red", linestyle="--", label=f"f_H ({fmt_freq(f_upper)})")
    ax_bw.set_title(f"Resposta frequencial del LED Driver")
    ax_bw.set_xlabel("Frequència (Hz)")
    ax_bw.set_ylabel("Guany normalitzat (dB)")
    ax_bw.grid(True, which="both", alpha=0.4)
    ax_bw.legend()
    plt.tight_layout()
    plt.savefig(f"fitxers/{folder}/system_frequency_response.png", dpi=150, bbox_inches="tight")
    plt.show()
    # plt.close()
    print(f"Saved: fitxers/{folder}/system_frequency_response.png")

def experiment2_BERvsBitrate(signals, bits_per_sample_rx, bit_rates, experiment="BER_OOK", ref="prbs.csv", mode="OOK", channels = 2, version=1):

    # Gold reference: the exact PRBS sequence sent by the AWG (seed=42, same as
    # generate_prbs_4096).  Using the stored pattern avoids decoding the noisy
    # scope CH2 (TX probe), which itself suffers bandwidth-limited duty-cycle
    # bias at high bitrates and would corrupt the BER measurement.
    prbs_gold =  np.loadtxt(f"fitxers/{experiment}/{ref}", delimiter=",")

    results = {}
    for s, b, bit_rate in zip(signals, bits_per_sample_rx, bit_rates):
        convertir_senyal_osciloscopi(
          senyal_csv=f"fitxers/{experiment}/{s}.csv", senyal_neta_csv=f"fitxers/{experiment}/{s}_neta.csv", channels=channels
        )
        if channels == 2:
            rx_signal = np.loadtxt(f"fitxers/{experiment}/{s}_neta_tx.csv", delimiter=",")
        else:
            rx_signal = np.loadtxt(f"fitxers/{experiment}/{s}_neta.csv", delimiter=",")
       
        rx_bits = _decode_bits_from_signal(rx_signal, b)
        if version == 1:
            ber, errors, total = _calculate_ber(prbs_gold, rx_bits)
        else:
            ber, errors, total = _calculate_ber_v2(prbs_gold, rx_bits, mode,)
        results[s] = {"ber": ber, "errors": errors, "total": total, "bit_rate": bit_rate}      
        print(f"  {s:30s}  BER = {ber:.6f}  ({errors}/{total} errors)  @{bit_rate:.0f} baud")

    valid = {s: v for s, v in results.items() if v is not None}
    if valid:
        sorted_items = sorted(valid.items(), key=lambda x: x[1]["bit_rate"])
        x = [v["bit_rate"] for _, v in sorted_items]
        y = [v["ber"] for _, v in sorted_items]

        # Interpolation over all signals
        x_interp = np.linspace(min(x), max(x), 500)
        y_interp = np.interp(x_interp, x, y)

        ber_target = 0.01
        d_at_target = None
        for i in range(len(y_interp) - 1):
            if y_interp[i] <= ber_target <= y_interp[i + 1] or \
               y_interp[i] >= ber_target >= y_interp[i + 1]:
                d_at_target = np.interp(ber_target,
                                        [y_interp[i], y_interp[i + 1]],
                                        [x_interp[i], x_interp[i + 1]])
                break

        fig, ax = plt.subplots(figsize=(10, 5))

        # Plot all signals
        ax.plot(x_interp, y_interp, linewidth=2,
                color="steelblue")
        ax.scatter(x, y, color="steelblue", zorder=5, s=50)

        ax.axhline(ber_target, color="orange", linestyle="--",
                   linewidth=1.5, label="BER = 0.01")

        if d_at_target:
            ax.axvline(d_at_target, color="red", linestyle="--",
                       linewidth=1.5,
                       label=f"{d_at_target:.0f} baud")
            #ax.plot(d_at_target, ber_target, "ro", markersize=8, zorder=6)

        ax.set_xlabel("Symbol rate (baud)", fontsize=12)
        ax.set_ylabel("BER", fontsize=12)
        ax.set_title("BER vs Symbol rate ", fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, which="both", alpha=0.3)
        plt.tight_layout()
        plt.show()

    return results

def experiment3_BERvsDISTANCE(signals, distances, bits_per_sample, ref="prbs.csv", mode="OOK"):
    results = {}
    tx_signal = np.loadtxt(f"fitxers/SNRvsDISTANCE/{ref}", delimiter=",")

    for s, d in zip(signals, distances):
        convertir_senyal_osciloscopi(
            senyal_csv=f"fitxers/SNRvsDISTANCE/{s}.csv",
            senyal_neta_csv=f"fitxers/SNRvsDISTANCE/{s}_neta.csv",
            channels=1,
        )
        rx_signal = np.loadtxt(f"fitxers/SNRvsDISTANCE/{s}_neta.csv", delimiter=",")
        rx_bits = _decode_bits_from_signal(rx_signal, bits_per_sample)
        ber, errors, total = _calculate_ber_v2(tx_signal, rx_bits, mode)
        results[s] = {"ber": ber, "errors": errors, "total": total, "distance": d}
        print(f"  {s:30s}  BER = {ber:.6f}  ({errors}/{total} errors)  @{d:.0f} cm")


    valid = {s: v for s, v in results.items() if v is not None}
    if valid:
        sorted_items = sorted(valid.items(), key=lambda x: x[1]["distance"])
        x = [v["distance"] for _, v in sorted_items]
        y = [v["ber"] for _, v in sorted_items]

        # Interpolation over all signals
        x_interp = np.linspace(min(x), max(x), 500)
        y_interp = np.interp(x_interp, x, y)

        # BER = 0.1 crossing calculated only from 3rd signal onwards
        x_from3 = x[2:]
        y_from3 = y[2:]
        x_interp_from3 = np.linspace(min(x_from3), max(x_from3), 500)
        y_interp_from3 = np.interp(x_interp_from3, x_from3, y_from3)

        ber_target = 0.01
        d_at_target = None
        for i in range(len(y_interp_from3) - 1):
            if y_interp_from3[i] <= ber_target <= y_interp_from3[i + 1] or \
               y_interp_from3[i] >= ber_target >= y_interp_from3[i + 1]:
                d_at_target = np.interp(ber_target,
                                        [y_interp_from3[i], y_interp_from3[i + 1]],
                                        [x_interp_from3[i], x_interp_from3[i + 1]])
                break

        fig, ax = plt.subplots(figsize=(10, 5))

        # Plot all signals
        ax.plot(x_interp, y_interp, linewidth=2,
                color="steelblue")
        ax.scatter(x, y, color="steelblue", zorder=5, s=50)

        ax.axhline(ber_target, color="orange", linestyle="--",
                   linewidth=1.5, label="BER = 0.01")

        if d_at_target:
            ax.axvline(d_at_target, color="red", linestyle="--",
                       linewidth=1.5,
                       label=f"{d_at_target:.0f} cm")

        ax.set_xlabel("Distance (cm)", fontsize=12)
        ax.set_ylabel("BER", fontsize=12)
        ax.set_title("BER vs Distance", fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, which="both", alpha=0.3)
        plt.tight_layout()
        plt.show()

    else:
        print("No hi ha resultats vàlids per fer el plot.")

    return results

def experiment4_SNRvsDISTANCE(signals, distances, bits_per_sample, ref="prbs.csv"):
    results = {}
    tx_signal = np.loadtxt(f"fitxers/SNRvsDISTANCE/{ref}", delimiter=",")
    
    for s, d in zip(signals, distances):
        convertir_senyal_osciloscopi(
            senyal_csv=f"fitxers/SNRvsDISTANCE/{s}.csv",
            senyal_neta_csv=f"fitxers/SNRvsDISTANCE/{s}_neta.csv",
            channels=1,
        )
        rx_signal = np.loadtxt(f"fitxers/SNRvsDISTANCE/{s}_neta.csv", delimiter=",")
        rx_bits = _decode_raw_from_signal(rx_signal, bits_per_sample)
        snr_db, snr_linear, errors, total = calculate_snr(tx_signal, rx_bits)
        
        results[s] = {
            "snr_db": snr_db,
            "snr_linear": snr_linear,
            "distance": d,
            "errors": errors,
            "total_signal": total
        }
        print(f"  {s:30s}  SNR = {snr_db:7.2f} dB, {snr_linear} @{d:6.1f} cm BER = {(errors/total):.6f}  ({errors}/{total} errors)")
    
    if results:
        sorted_items = sorted(results.items(), key=lambda x: x[1]["distance"])
        x = np.array([v["distance"] for _, v in sorted_items])
        y = np.array([v["snr_db"] for _, v in sorted_items])
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(x, y, "o-", color="steelblue", linewidth=2.5, markersize=10)
        ax.set_xlabel("Distance (cm)", fontsize=12, fontweight='bold')
        ax.set_ylabel("SNR (dB)", fontsize=12, fontweight='bold')
        ax.set_title("SNR vs Distance", fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add value labels on points
        for xi, yi in zip(x, y):
            ax.annotate(f'{yi:.2f} dB', (xi, yi), textcoords="offset points", 
                       xytext=(0, 10), ha='center', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f"fitxers/SNRvsDISTANCE/snr_vs_distance.png", dpi=150, bbox_inches="tight")
        plt.show()
        
        print(f"\nSaved plot to: fitxers/SNRvsDISTANCE/snr_vs_distance.png")
    
    return results

"""
def _calculate_snr(tx_signal, rx_signal):
    
    ##Calcula la SNR entre la senyal transmesa (neta) i la rebuda (amb soroll).
    ##Retorna la SNR en dB.
    
    # Assegurem la mateixa longitud
    min_len = min(len(tx_signal), len(rx_signal))
    tx = tx_signal[:min_len]
    rx = rx_signal[:min_len]

    noise = rx - tx  # soroll estimat

    power_signal = np.mean(tx**2)  # potència de la senyal
    power_noise = np.mean(noise**2)  # potència del soroll

    if power_noise == 0:
        return float("inf")  # sense soroll

    snr_db = 10 * np.log10(power_signal / power_noise)
    return snr_db

    
def _decode_bits_from_signal(rx_signal, bits_per_sample, mode=None):
    rx_bits_raw = binaritzar(rx_signal)

    if mode == "OOK":
        # Keep your original simpler logic for OOK
        return np.array(rx_bits_raw)
    # --- Manchester decoding ---
    symbol_len = int(2 * bits_per_sample)

    # Find the first flank
    first_flank = 0
    for i in range(1, len(rx_bits_raw)):
        if rx_bits_raw[i] != rx_bits_raw[i - 1]:
            first_flank = i
            break

    # Try offset at the flank; if first symbol is invalid (00 or 11),
    # step back half a symbol — we caught the mid-bit transition, not the start
    offset = first_flank
    if offset + symbol_len <= len(rx_bits_raw):
        first_half  = np.mean(rx_bits_raw[offset : offset + int(bits_per_sample)])
        second_half = np.mean(rx_bits_raw[offset + int(bits_per_sample) : offset + symbol_len])
        c1 = 1 if first_half  > 0.5 else 0
        c2 = 1 if second_half > 0.5 else 0
        if c1 == c2:  # invalid: 00 or 11 → we're misaligned
            offset = max(0, first_flank - int(bits_per_sample))

    results = []
    n_symbols = (len(rx_bits_raw) - offset) // symbol_len

    for i in range(n_symbols):
        start = offset + i * symbol_len
        mid   = start + int(bits_per_sample)
        end   = start + symbol_len

        if end > len(rx_bits_raw):
            break

        first_half  = np.mean(rx_bits_raw[start:mid])
        second_half = np.mean(rx_bits_raw[mid:end])

        c1 = 1 if first_half  > 0.5 else 0
        c2 = 1 if second_half > 0.5 else 0

        if   c1 == 1 and c2 == 0:
            results.append(1)
        elif c1 == 0 and c2 == 1:
            results.append(0)
        else:
            continue

    return np.array(results)

    
def decodificar(senyal_csv=None, mode=None, bits_per_sample=6.1):
    print("\n--- MODE DECODIFICACIÓ (RX) ---")

    if senyal_csv is None:
        senyal_csv = input("Introdueix el nom del CSV del senyal rebut (ex: 'senyal_osv2_neta.csv'): ").strip()
    try:
        rx_signal = np.loadtxt(senyal_csv, delimiter=",")
    except Exception as e:
        print(f"No s'ha pogut llegir el CSV: {e}")
        return

    if mode is None:
        mode = input("En quin mode s'ha envFFat? (OOK / MANCHESTER): ").strip().upper()

    rx_bits_raw = binaritzar(rx_signal)

    if mode == "MANCHESTER":
        base_pattern = np.array([1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1])
    else:
        base_pattern = np.array([1, 0, 1, 0, 1, 0, 1, 0])

    pattern = np.repeat(base_pattern, bits_per_sample)

    rx_bipolar = 2 * rx_bits_raw - 1
    pattern_bipolar = 2 * pattern - 1
    correlation = np.correlate(rx_bipolar, pattern_bipolar, mode="valid")
    threshold_corr = len(pattern) * 0.85
    potential_starts = np.where(correlation >= threshold_corr)[0]

    found_indices = []
    if len(potential_starts) > 0:
        found_indices.append(potential_starts[0])
        for idx in potential_starts[1:]:
            if idx > found_indices[-1] + (bits_per_sample * 20):
                found_indices.append(idx)
    print(f"S'han trobat {len(found_indices)} frames a {bits_per_sample} samples/bit.")

    for start_idx in found_indices:
        try:

            def get_bit_at(bit_pos):
                sample_idx = start_idx + int((bit_pos + 0.5) * bits_per_sample)
                if sample_idx >= len(rx_bits_raw):
                    return None
                return rx_bits_raw[sample_idx]

            bits_dec = ""

            if mode == "MANCHESTER":
                bits_capçalera = ""
                for b in range(16):
                    p1 = get_bit_at(b * 2)
                    p2 = get_bit_at(b * 2 + 1)
                    bits_capçalera += "1" if (p1 == 1 and p2 == 0) else "0"

                msg_len = int(bits_capçalera[8:16], 2)
                total_bits_dades = 16 + (msg_len * 8)

                bits_dec = bits_capçalera
                for b in range(16, total_bits_dades):
                    p1 = get_bit_at(b * 2)
                    p2 = get_bit_at(b * 2 + 1)
                    bits_dec += "1" if (p1 == 1 and p2 == 0) else "0"
            else:
                bits_capçalera = ""
                for b in range(16):
                    bit = get_bit_at(b)
                    if bit is None:
                        break
                    bits_capçalera += str(bit)

                msg_len = int(bits_capçalera[8:16], 2)
                total_bits_a_llegir = 16 + (msg_len * 8)

                bits_dec = bits_capçalera
                for b in range(16, total_bits_a_llegir):
                    bit = get_bit_at(b)
                    if bit is None:
                        break
                    bits_dec += str(bit)

            msg_len = int(bits_dec[8:16], 2)
            msg_bits = bits_dec[16 : 16 + msg_len * 8]
            res = "".join([chr(int(msg_bits[k : k + 8], 2)) for k in range(0, len(msg_bits), 8)])

            msg_bits = bits_dec[16 : 16 + msg_len]
            bpm_final = int(msg_bits, 2)
            print(f"--> Frame trobat a mostra {start_idx}:")
            print(f"    Bits totals: {bits_dec}")
            print(f"    BPM detectat: {bpm_final}")
            print(f" Missatge a mostra {start_idx}: '{res}'")
            return bpm_final
        except Exception as e:
            print(f"DEBUG: Error en frame a {start_idx}: {e}")
            continue

def decodificar_v2(senyal_csv=None, mode=None, bits_per_sample=6.1):
    print("\n--- MODE DECODIFICACIÓ V2 (per flancs) ---")
    if senyal_csv is None:
        senyal_csv = input("Nom del CSV: ").strip()
    try:
        rx_signal = np.loadtxt(senyal_csv, delimiter=",")
    except Exception as e:
        print(f"Error llegint CSV: {e}")
        return

    if mode is None:
        mode = input("Mode (OOK / MANCHESTER): ").strip().upper()

    rx_bits_raw = binaritzar(rx_signal)

    flancs = []
    for i in range(1, len(rx_bits_raw)):
        if rx_bits_raw[i] != rx_bits_raw[i - 1]:
            flancs.append(i)

    flancs.append(len(rx_bits_raw))
    flancs = np.array(flancs)

    if len(flancs) < 2:
        print("No s'han detectat prou flancs.")
        return

    rx_bits = []
    for i in range(len(flancs) - 1):
        durada = flancs[i + 1] - flancs[i]
        n_bits = max(1, int(round(durada / bits_per_sample)))
        valor = rx_bits_raw[flancs[i] + int(bits_per_sample / 2)]
        rx_bits.extend([valor] * n_bits)

    rx_bits = np.array(rx_bits)

    if mode == "MANCHESTER":
        base_pattern = np.array([1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1])
    else:
        base_pattern = np.array([1, 0, 1, 0, 1, 0, 1, 0])

    rx_bipolar = 2 * rx_bits - 1
    pattern_bipolar = 2 * base_pattern - 1
    correlation = np.correlate(rx_bipolar, pattern_bipolar, mode="valid")
    threshold = len(base_pattern) * 0.8

    starts = np.where(correlation >= threshold)[0]

    if len(starts) == 0:
        print("No s'han trobat frames.")
        return

    for start_idx in starts:
        try:
            if mode == "OOK":
                bits_cap = "".join(str(b) for b in rx_bits[start_idx : start_idx + 16])
                msg_len_bits = int(bits_cap[8:16], 2)
                total_bits = 16 + msg_len_bits
                bits_dec = "".join(str(b) for b in rx_bits[start_idx : start_idx + total_bits])
                msg_bits = bits_dec[16:]
            else:
                bits_temp = rx_bits[start_idx : start_idx + 300]
                bits_manchester = []
                for i in range(0, len(bits_temp) - 1, 2):
                    p1, p2 = bits_temp[i], bits_temp[i + 1]
                    if p1 == 1 and p2 == 0:
                        bits_manchester.append(1)
                    elif p1 == 0 and p2 == 1:
                        bits_manchester.append(0)

                bits_cap = "".join(str(b) for b in bits_manchester[:16])
                msg_len_bytes = int(bits_cap[8:16], 2)
                total_bits_utils = 16 + (msg_len_bytes * 8)
                bits_dec = "".join(str(b) for b in bits_manchester[:total_bits_utils])
                msg_bits = bits_dec[16:]

            if len(msg_bits) > 0:
                resultat = int(msg_bits, 2)
                print(f"\n--> Frame a bit {start_idx}")
                print(f"Bits Totals: {bits_dec}")
                print(f"Missatge: '{resultat}'")

            return resultat

        except Exception as e:
            print(f"Error en frame {start_idx}: {e}")


"""
