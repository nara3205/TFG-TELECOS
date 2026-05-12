import numpy as np
import pandas as pd
import subprocess

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
    
    # Repeticions cadena de bits amb separació (8 zeros)
    raw_bits = ""
    for _ in range(num_repeats):
        raw_bits += frame_bits + '00000000'

    if mode == "MANCHESTER":
        tx_bits_str = "".join(["10" if b == '1' else "01" for b in raw_bits])
    else:
        tx_bits_str = raw_bits

    # Padding fins a 4096 -> script csv2arb ho demana
    tx_signal = np.zeros(4096)
    signal_data = np.array([1 if b == '1' else 0.0 for b in tx_bits_str])
    
    if len(signal_data) > 4096:
        print(f"El senyal ({len(signal_data)}) excedeix les 4096 mostres")
    
    tx_signal[:min(len(signal_data), 4096)] = signal_data[:4096]

    # Guardar a CSV i cridar script
    nom_csv = "fitxers/senyal.csv"
    np.savetxt(nom_csv, tx_signal, delimiter=',', fmt='%.2f')
    print(f"Senyal guardat a {nom_csv}")
    executar_comanda(f"python3 utils/csv2arb.py {nom_csv}")

def decodificar(senyal_csv=None, mode=None, bits_per_sample=6.1):
    print("\n--- MODE DECODIFICACIÓ (RX) ---")

    # LLegim el CSV de l'oscil·loscopi ("netejat" per l'script convert_signal.py per quedar-nos 2a columna)  
    if senyal_csv is None:
        senyal_csv = input("Introdueix el nom del CSV del senyal rebut (ex: 'senyal_osv2_neta.csv'): ").strip()  
    try:
        rx_signal = np.loadtxt(senyal_csv, delimiter=',')
    except Exception as e:
        print(f"No s'ha pogut llegir el CSV: {e}")
        return

    if mode is None:
        mode = input("En quin mode s'ha envFFat? (OOK / MANCHESTER): ").strip().upper()
    
    # Binarització
    rx_bits_raw = binaritzar(rx_signal)
    
    # Creem el patró original de bits -> start_sequence
    if mode == "MANCHESTER":
        base_pattern = np.array([1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1])
    else:
        base_pattern = np.array([1,0,1,0,1,0,1,0])
    # Repetim patró ->  cada bit del patró es repeteix bits_per_sample cops
    pattern = np.repeat(base_pattern, bits_per_sample)

    # Correlació bipolar -> passem senyal a -1 i 1
    rx_bipolar = 2 * rx_bits_raw - 1
    pattern_bipolar = 2 * pattern - 1    
    correlation = np.correlate(rx_bipolar, pattern_bipolar, mode='valid')
    threshold_corr = len(pattern) * 0.85
    potential_starts = np.where(correlation >= threshold_corr)[0]

    # Mirem que hi haguin sepaarcions
    found_indices = []
    if len(potential_starts) > 0:
        found_indices.append(potential_starts[0])
        for idx in potential_starts[1:]:
            # Separem almenys la meitat d'un frame (bits_per_sample)
            if idx > found_indices[-1] + (bits_per_sample * 20): # a modificar el 20!
                found_indices.append(idx)
    print(f"S'han trobat {len(found_indices)} frames a {bits_per_sample} samples/bit.")

    for start_idx in found_indices:
        try:
            # Funció aux per llegir un bit real saltant N mostres
            def get_bit_at(bit_pos):
                sample_idx = start_idx + int((bit_pos + 0.5) * bits_per_sample)

                # Si surt de rang → retornem None (o 0 si prefereixes)
                if sample_idx >= len(rx_bits_raw):
                    return None

                return rx_bits_raw[sample_idx]
            
            bits_dec = ""
            
            if mode == "MANCHESTER":
                # 1. Llegim els primers 16 bits de dades (que són 32 polsos en Manchester)
                # 8 bits de start + 8 bits de longitud
                bits_capçalera = ""
                for b in range(16):
                    # En Manchester mirem parelles de polsos (b*2 i b*2 + 1)
                    p1 = get_bit_at(b * 2)
                    p2 = get_bit_at(b * 2 + 1)
                    bits_capçalera += "1" if (p1 == 1 and p2 == 0) else "0"

                # 2. Extraiem la longitud real del preàmbul
                msg_len = int(bits_capçalera[8:16], 2)
                total_bits_dades = 16 + (msg_len * 8)

                # 3. Llegim la resta del missatge bit a bit
                bits_dec = bits_capçalera
                for b in range(16, total_bits_dades):
                    p1 = get_bit_at(b * 2)
                    p2 = get_bit_at(b * 2 + 1)
                    bits_dec += "1" if (p1 == 1 and p2 == 0) else "0"
            else:
                # 1. Llegim primer els 16 bits inicials (8 de sincronisme + 8 de longitud)
                bits_capçalera = ""
                for b in range(16):
                    bit = get_bit_at(b)
                    if bit is None:
                        break
                    bits_capçalera += str(bit)

                # 2. Extraiem la longitud real (bits del 8 al 15)
                msg_len = int(bits_capçalera[8:16], 2)
                total_bits_a_llegir = 16 + (msg_len * 8)

                # 3. Ara sí, llegim exactament el que toca
                bits_dec = bits_capçalera
                for b in range(16, total_bits_a_llegir):
                    bit = get_bit_at(b)
                    if bit is None:
                        break
                    bits_dec += str(bit)

            # --- DECODIFICACIÓ  ---
            # El start sequence ja saltat amb la correlació 
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

def decodificar_v2(senyal_csv=None, mode=None,  bits_per_sample=6.1):
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
        
        valor = rx_bits_raw[flancs[i]+ int(bits_per_sample/2)] 
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

            # Convertir a text o valor
            if len(msg_bits) > 0:
                resultat = int(msg_bits, 2)
                
                print(f"\n--> Frame a bit {start_idx}")
                print(f"Bits Totals: {bits_dec}")
                print(f"Missatge: '{resultat}'")
                
            return resultat

        except Exception as e:
            print(f"Error en frame {start_idx}: {e}")

def binaritzar(rx_signal):
    v_max, v_min = np.max(rx_signal), np.min(rx_signal)
    threshold = (v_max + v_min) / 2
    return (rx_signal > threshold).astype(int)

def convertir_senyal_osciloscopi(senyal_csv=None, senyal_neta_csv=None, channels=1):

    if channels == 1:
        df = pd.read_csv(senyal_csv, usecols=[1], header=None)

        df_net = pd.to_numeric(df.iloc[:, 0], errors='coerce').dropna()

        # GUARDAR TAL QUAL (float)
        df_net.to_csv(senyal_neta_csv, index=False, header=False, float_format='%.10f')

    elif channels == 2:
        df = pd.read_csv(senyal_csv, usecols=[1, 2], header=None)

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

        # Calculem el llindar adaptatiu (binarització)
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
        file_ch1 = base + "_rx.csv"
        file_ch2 = base + "_tx.csv"

        ch1_bits.to_csv(file_ch1, index=False, header=False)
        ch2_bits.to_csv(file_ch2, index=False, header=False)

        print(f"Guardat: {file_ch1} i {file_ch2}")

    else:
        raise ValueError("channels ha de ser 1 o 2")

def main():
    print("========================================")
    print("   SCRIPT PROVA    ")
    print("========================================")
    opcio = input("(C)odificar o (D)ecodificar?: ").strip().lower()

    if opcio == 'c':
        codificar()
    elif opcio == 'd':
        decodificar()
    else:
        print("Opció no vàlida.")

if __name__ == "__main__":
    main()
