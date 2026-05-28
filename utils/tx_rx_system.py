import numpy as np
import pandas as pd
import subprocess
import matplotlib.pyplot as plt
import math

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

    start_sequence = '10101010'
    preamble = format(len(message), '08b')
    message_bits = ''.join(format(ord(c), '08b') for c in message)
    
    frame_bits = start_sequence + preamble + message_bits
    
    raw_bits = ""
    for _ in range(num_repeats):
        raw_bits += frame_bits + '00000000'

    if mode == "MANCHESTER":
        tx_bits_str = "".join(["10" if b == '1' else "01" for b in raw_bits])
    else:
        tx_bits_str = raw_bits

    tx_signal = np.zeros(4096)
    signal_data = np.array([1 if b == '1' else 0.0 for b in tx_bits_str])
    
    if len(signal_data) > 4096:
        print(f"El senyal ({len(signal_data)}) excedeix les 4096 mostres")
    
    tx_signal[:min(len(signal_data), 4096)] = signal_data[:4096]

    nom_csv = "fitxers/senyal.csv"
    np.savetxt(nom_csv, tx_signal, delimiter=',', fmt='%.2f')
    print(f"Senyal guardat a {nom_csv}")
    executar_comanda(f"python3 utils/csv2arb.py {nom_csv}")

def decodificar(senyal_csv=None, mode=None, bits_per_sample=6.1):
    print("\n--- MODE DECODIFICACIÓ (RX) ---")

    if senyal_csv is None:
        senyal_csv = input("Introdueix el nom del CSV del senyal rebut (ex: 'senyal_osv2_neta.csv'): ").strip()  
    try:
        rx_signal = np.loadtxt(senyal_csv, delimiter=',')
    except Exception as e:
        print(f"No s'ha pogut llegir el CSV: {e}")
        return

    if mode is None:
        mode = input("En quin mode s'ha envFFat? (OOK / MANCHESTER): ").strip().upper()
    
    rx_bits_raw = binaritzar(rx_signal)
    
    if mode == "MANCHESTER":
        base_pattern = np.array([1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1])
    else:
        base_pattern = np.array([1,0,1,0,1,0,1,0])

    pattern = np.repeat(base_pattern, bits_per_sample)

    rx_bipolar = 2 * rx_bits_raw - 1
    pattern_bipolar = 2 * pattern - 1    
    correlation = np.correlate(rx_bipolar, pattern_bipolar, mode='valid')
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
            msg_bits = bits_dec[16 : 16 + msg_len*8]
            res = "".join([chr(int(msg_bits[k:k+8], 2)) for k in range(0, len(msg_bits), 8)])
    
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
        rx_signal = np.loadtxt(senyal_csv, delimiter=',')
    except Exception as e:
        print(f"Error llegint CSV: {e}")
        return
    
    if mode is None:
        mode = input("Mode (OOK / MANCHESTER): ").strip().upper()

    rx_bits_raw = binaritzar(rx_signal)
    
    flancs = []
    for i in range(1, len(rx_bits_raw)):
        if rx_bits_raw[i] != rx_bits_raw[i-1]:
            flancs.append(i)
    
    flancs.append(len(rx_bits_raw)) 
    flancs = np.array(flancs)
    
    if len(flancs) < 2:
        print("No s'han detectat prou flancs.")
        return
    
    rx_bits = []
    for i in range(len(flancs)-1):
        durada = flancs[i+1] - flancs[i]
        n_bits = max(1, int(round(durada / bits_per_sample)))
        valor = rx_bits_raw[flancs[i] + int(bits_per_sample/2)] 
        rx_bits.extend([valor] * n_bits)

    rx_bits = np.array(rx_bits)

    if mode == "MANCHESTER":
        base_pattern = np.array([1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1])
    else:
        base_pattern = np.array([1,0,1,0,1,0,1,0])
        
    rx_bipolar = 2 * rx_bits - 1
    pattern_bipolar = 2 * base_pattern - 1
    correlation = np.correlate(rx_bipolar, pattern_bipolar, mode='valid')
    threshold = len(base_pattern) * 0.8

    starts = np.where(correlation >= threshold)[0]

    if len(starts) == 0:
        print("No s'han trobat frames.")
        return

    for start_idx in starts:
        try:
            if mode == "OOK":
                bits_cap = "".join(str(b) for b in rx_bits[start_idx:start_idx+16])
                msg_len_bits = int(bits_cap[8:16], 2)
                total_bits = 16 + msg_len_bits 
                bits_dec = "".join(str(b) for b in rx_bits[start_idx:start_idx+total_bits])
                msg_bits = bits_dec[16:]
            else:
                bits_temp = rx_bits[start_idx:start_idx+300] 
                bits_manchester = []
                for i in range(0, len(bits_temp)-1, 2):
                    p1, p2 = bits_temp[i], bits_temp[i+1]
                    if p1 == 1 and p2 == 0: bits_manchester.append(1)
                    elif p1 == 0 and p2 == 1: bits_manchester.append(0)

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

def _calculate_ber(tx_bits, rx_bits):
    """
    Calcula BER usant correlació per trobar el millor offset d'alineació.
    """
    tx_bits = np.array(tx_bits)
    n_tx = len(tx_bits)
    n_rx = len(rx_bits)

    if n_rx < n_tx:
        compare_len = n_rx
        errors = int(np.sum(tx_bits[:compare_len] != rx_bits[:compare_len]))
        return errors / compare_len, errors, compare_len

    # --- Correlació bipolar per trobar el millor offset ---
    tx_bipolar = 2 * tx_bits - 1
    rx_bipolar = 2 * rx_bits - 1

    correlation = np.correlate(rx_bipolar, tx_bipolar, mode='valid')
    # El pic de correlació indica on el TX s'assembla més al RX
    best_offset = int(np.argmax(correlation))

    # --- Acumula BER sobre tots els cicles complets a partir del millor offset ---
    total_errors = 0
    total_bits   = 0
    pos = best_offset

    while pos + n_tx <= n_rx:
        total_errors += int(np.sum(tx_bits != rx_bits[pos:pos + n_tx]))
        total_bits   += n_tx
        pos          += n_tx

    if total_bits == 0:
        return float('nan'), 0, 0

    return total_errors / total_bits, total_errors, total_bits

def binaritzar(rx_signal):
    v_max, v_min = np.max(rx_signal), np.min(rx_signal)
    threshold = (v_max + v_min) / 2
    return (rx_signal > threshold).astype(int)

def convertir_senyal_osciloscopi(senyal_csv=None, senyal_neta_csv=None, channels=1):

    if channels == 1:
        df = pd.read_csv(senyal_csv, usecols=[1], header=None)
        df_net = pd.to_numeric(df.iloc[:, 0], errors='coerce').dropna()
        df_net.to_csv(senyal_neta_csv, index=False, header=False, float_format='%.10f')

    elif channels == 2:
        df = pd.read_csv(senyal_csv, usecols=[1, 2],header=None, skiprows=1)
        ch1 = pd.to_numeric(df.iloc[:, 0], errors='coerce')
        ch2 = pd.to_numeric(df.iloc[:, 1], errors='coerce')
        df_net = pd.concat([ch1, ch2], axis=1).dropna()
        base = senyal_neta_csv.replace(".csv", "")
        df_net.iloc[:, 0].to_csv(base + "_rx.csv", index=False, header=False, float_format='%.10f')
        df_net.iloc[:, 1].to_csv(base + "_tx.csv", index=False, header=False, float_format='%.10f')
        print(f"Guardat: {base}_rx.csv i {base}_tx.csv")
    else:
        raise ValueError("channels ha de ser 1 o 2") 

def convertir_senyal_osciloscopi_bin(senyal_csv=None, senyal_neta_csv=None, channels=1):

    if channels == 1:
        df = pd.read_csv(senyal_csv, usecols=[1], header=None)
        df_net = pd.to_numeric(df.iloc[:, 0], errors='coerce').dropna()
        llindar = (df_net.max() + df_net.min()) / 2
        df_bits = df_net.apply(lambda x: 1 if x > llindar else 0)
        df_bits.to_csv(senyal_neta_csv, index=False, header=False)

    elif channels == 2:
        df = pd.read_csv(senyal_csv, usecols=[1, 2], header=None)
        ch1 = pd.to_numeric(df.iloc[:, 0], errors='coerce')
        ch2 = pd.to_numeric(df.iloc[:, 1], errors='coerce')
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

def generate_prbs_4096(n_bits: int, seed: int = 0xACE1):

    N_TOTAL = 4096

    if N_TOTAL % n_bits != 0:
        raise ValueError(
            f"n_bits = {n_bits} must divide 4096 exactly "
            f"(valid: { [d for d in range(1, 4097) if 4096 % d == 0] })"
        )

    samples_per_bit = N_TOTAL // n_bits
    bits = []
    lfsr = seed & 0xFFFF

    for _ in range(n_bits):
        bit = lfsr & 1
        bits.append(bit)
        feedback = ((lfsr >> 0) ^ (lfsr >> 2) ^ (lfsr >> 3) ^ (lfsr >> 5)) & 1
        lfsr = ((lfsr >> 1) | (feedback << 15)) & 0xFFFF

    print(f"Generated PRBS bits: {bits}")
    signal = np.repeat(bits, samples_per_bit).astype(float)
    signal = signal[:N_TOTAL]

    nom_csv = "fitxers/prbs.csv"
    np.savetxt(nom_csv, signal, delimiter=',', fmt='%.2f')
    print(f"Samples per bit: {samples_per_bit}")
    print(f"Saved signal to {nom_csv}")
    executar_comanda(f"python3 utils/csv2arb.py {nom_csv}")

def experiment1_BODE(senyals, signal_Ids, sampling_rates, freqs):
    n    = len(senyals)
    cols = 3
    rows = math.ceil(n / cols)
    fig,axes = plt.subplots(rows, cols, figsize=(18, 4*rows))
    axes = axes.flatten()

    for idx, signal_Id in enumerate(signal_Ids):
        fs  = sampling_rates[idx]
        N   = len(signal_Id)
        fft_vals = np.fft.rfft(signal_Id)
        fft_mag  = np.abs(fft_vals) * 2 / N
        fft_freq = np.fft.rfftfreq(N, d=1/fs)
        
        ax = axes[idx]
        ax.plot(fft_freq / 1e3, fft_mag, linewidth=0.8, color='crimson')
        ax.axvline(freqs[idx] / 1e3, color='navy', linestyle='--', linewidth=1,
                label=f'f={freqs[idx]/1e3:.0f} kHz')
        ax.set_title(f"{senyals[idx].upper()} — {freqs[idx]/1e3:.0f} kHz", fontsize=10)
        ax.set_xlabel("Freqüència (kHz)", fontsize=8)
        ax.set_ylabel("|I| (mA)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(True, alpha=0.3)

    for i in range(n, len(axes)):
        axes[i].set_visible(False)

    plt.suptitle("FFT del Corrent I per cada senyal", fontsize=13, y=1.005)
    plt.tight_layout()
    plt.show()

    for idx, signal_Id in enumerate(signal_Ids):
        I_dc = np.mean(signal_Id)
        print(f"[{senyals[idx].upper()}] {freqs[idx]/1e3:>7.0f} kHz | I_dc = {I_dc:.3f} mA")

def _decode_bits_from_signal(rx_signal, bits_per_sample, mode=None):
    rx_bits_raw = binaritzar(rx_signal)
    
    # Find edges
    flancs = []
    for i in range(1, len(rx_bits_raw)):
        if rx_bits_raw[i] != rx_bits_raw[i-1]:
            flancs.append(i)
    flancs.append(len(rx_bits_raw))
    flancs = np.array(flancs)

    
    if len(flancs) < 2:
        return None
    
    rx_bits = []
    for i in range(len(flancs) - 1):
        durada = flancs[i+1] - flancs[i]
        
        # How many bits fit in this run?
        n_bits = max(1, int(round(durada / bits_per_sample)))
        
        # Sample in the middle of the run
        sample_idx = min(flancs[i] + int(bits_per_sample / 2), len(rx_bits_raw) - 1)
        valor = rx_bits_raw[sample_idx]
        
        rx_bits.extend([valor] * n_bits)
    
    return np.array(rx_bits)

def experiment2_BERvsBitrate(tx_signal_sequence, rx_signals, bits_per_sample_rx, bit_rates):
    print(f"\n--- EXPERIMENT 2: BER vs BITRATE ---")

    tx_bits = np.array(tx_signal_sequence)
    results = {}

    # --- Preprocessa tots els RX ---
    for s in rx_signals:
        signal_path = "fitxers/BW_RX/" + s
        convertir_senyal_osciloscopi(signal_path + ".csv", signal_path + "_neta.csv", channels=1)

    # --- Processa cada RX ---
    for s, b, bit_rate in zip(rx_signals, bits_per_sample_rx, bit_rates):
        senyal_csv = "fitxers/BW_RX/" + s + "_neta.csv"
        try:
            rx_signal = np.loadtxt(senyal_csv, delimiter=',')
        except Exception as e:
            print(f"Error llegint {senyal_csv}: {e}")
            results[s] = None
            continue

        rx_bits = _decode_bits_from_signal(rx_signal, b, mode=None)

        if rx_bits is None or len(rx_bits) == 0:
            print(f"[WARNING] No s'han pogut decodificar bits de {s}")
            results[s] = None
            continue

        ber, errors, total = _calculate_ber(tx_bits, rx_bits)

        results[s] = {
            'ber':          ber,
            'total_errors': errors,
            'total_bits':   total,
            'bit_rate':     bit_rate,
        }
        print(f"  {s:30s}  BER = {ber:.6f}  ({errors}/{total} errors)  @{bit_rate:.0f} bps")

    # --- Plot BER vs Bitrate ---
    valid = {s: v for s, v in results.items() if v is not None}

    if valid:
        sorted_items = sorted(valid.items(), key=lambda x: x[1]['bit_rate'])
        x      = [v['bit_rate'] for _, v in sorted_items]
        y      = [v['ber']      for _, v in sorted_items]
        labels = [s             for s, _ in sorted_items]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(x, y, marker='o', linewidth=2, color='steelblue', markersize=7)

        for xi, yi, label in zip(x, y, labels):
            ax.annotate(label, (xi, yi),
                        textcoords="offset points", xytext=(6, 6),
                        fontsize=8, color='dimgray')

        # Mark Rb_max: last bitrate where BER < 0.01 (1%)
        rb_max = None
        for xi, yi in zip(x, y):
            if yi < 0.01:
                rb_max = xi
        if rb_max:
            ax.axvline(rb_max, color='red', linestyle='--', linewidth=1.5,
                       label=f'$R_b^{{max}}$ = {rb_max:.0f} bps')
            ax.legend(fontsize=10)

        ax.set_xlabel("Bit Rate (bps)", fontsize=12)
        ax.set_ylabel("BER", fontsize=12)
        ax.set_title("BER vs Bit Rate", fontsize=14)
        ax.set_yscale('symlog', linthresh=1e-3)
        ax.set_ylim(bottom=0)
        ax.grid(True, which='both', alpha=0.3)
        plt.tight_layout()
        plt.show()
    else:
        print("No hi ha resultats vàlids per fer el plot.")

    return results

"""
def experiment2_BERvsBitrate(tx_signal_sequence, rx_signals, bits_per_sample_rx, bit_rates):
    
    Calcula el BER de cada senyal RX respecte el senyal TX de referència.
    Compara directament els bits binaritzats, sense assumir cap modulació.
    Args:
        tx_signal_sequence: array de bits TX (referència)
        rx_signals        : llista de noms de senyals RX (fitxers a fitxers/BW_RX/)
        bits_per_sample_tx: mostres per bit del TX
        bits_per_sample_rx: llista de mostres per bit dels RX (una per cada senyal)
        bit_rates         : llista de taxes de bits en bits/segon (una per cada senyal)
    Returns:
        dict amb { nom_senyal: { 'ber', 'total_errors', 'total_bits', 'bit_rate' } }
    
    print(f"\n--- EXPERIMENT 2: BER vs BITRATE ---")

    results = {}

    # --- Preprocessa tots els RX ---
    for s in rx_signals:
        signal_path = "fitxers/BW_RX/" + s
        convertir_senyal_osciloscopi(signal_path + ".csv", signal_path + "_neta.csv", channels=1)

    # --- Processa cada RX ---
    for s, b, bit_rate in zip(rx_signals, bits_per_sample_rx, bit_rates):
        senyal_csv = "fitxers/BW_RX/" + s + "_neta.csv"
        try:
            rx_signal = np.loadtxt(senyal_csv, delimiter=',')
        except Exception as e:
            print(f"Error llegint {senyal_csv}: {e}")
            results[s] = None
            continue

        rx_bits = _decode_bits_from_signal(rx_signal, b, mode=None)

        if rx_bits is None or len(rx_bits) == 0:
            print(f"[WARNING] No s'han pogut decodificar bits de {s}")
            results[s] = None
            continue

        ber, errors, total = _calculate_ber(tx_signal_sequence, rx_bits)

        results[s] = {
            'ber':          ber,
            'total_errors': errors,
            'total_bits':   total,
            'bit_rate':     bit_rate,
        }
        print(f"  {s:30s}  BER = {ber:.6f}  ({errors}/{total} errors)  @{bit_rate} bps")

    # --- Plot BER vs Bitrate ---
    valid = {s: v for s, v in results.items() if v is not None}

    if valid:
        sorted_items = sorted(valid.items(), key=lambda x: x[1]['bit_rate'])
        x = [v['bit_rate']     for _, v in sorted_items]
        y = [v['ber']          for _, v in sorted_items]
        labels = [s            for s, _ in sorted_items]

        fig, ax = plt.subplots(figsize=(9, 5))

        ax.plot(x, y, marker='o', linewidth=2, color='steelblue', markersize=7)

        for xi, yi, label in zip(x, y, labels):
            ax.annotate(label, (xi, yi),
                        textcoords="offset points", xytext=(6, 6),
                        fontsize=8, color='dimgray')

        ax.set_xlabel("Bit Rate (bps)", fontsize=12)
        ax.set_ylabel("BER", fontsize=12)
        ax.set_title("BER vs Bit Rate", fontsize=14)
        ax.set_yscale('log')          # log scale makes BER curves much easier to read
        ax.grid(True, which='both', alpha=0.3)
        plt.tight_layout()
        plt.show()
    else:
        print("No hi ha resultats vàlids per fer el plot.")

    return results
"""