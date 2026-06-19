import numpy as np

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

    rx_bits_aligned = rx_bits_hard[best_offset: best_offset + n_tx]
    analog_norm_aligned = analog_norm[best_offset: best_offset + n_tx]

    error_signal = analog_norm_aligned - tx_bits

    bit_errors = int(np.sum(tx_bits != rx_bits_aligned))

    power_signal = np.mean(tx_bits ** 2)

    power_noise = np.mean(error_signal ** 2)

    snr_linear = power_signal / power_noise
    snr_db = 10 * np.log10(snr_linear)

    return snr_db, snr_linear, bit_errors, n_tx


def calculate_rigorous_optical_snr(tx_bits, analog_samples):
    """
    Calculates the true effective SNR and Optical Q-factor
    accounting for signal-dependent asymmetric noise (Shot + Floor).
    """
    tx_bits = np.array(tx_bits, dtype=int)
    analog_samples = np.array(analog_samples, dtype=float)
    n_tx = len(tx_bits)

    # 1. Synchronization / Alignment via hard thresholding
    v_thresh = np.mean(analog_samples)
    rx_bits_hard = (analog_samples > v_thresh).astype(int)

    tx_bipolar = 2 * tx_bits - 1
    rx_bipolar = 2 * rx_bits_hard - 1
    correlation = np.correlate(rx_bipolar, tx_bipolar, mode="valid")
    best_offset = int(np.argmax(correlation))

    analog_aligned = analog_samples[best_offset: best_offset + n_tx]
    rx_bits_aligned = rx_bits_hard[best_offset: best_offset + n_tx]

    # 2. Extract conditional voltage distributions
    samples_high = analog_aligned[tx_bits == 1]
    samples_low = analog_aligned[tx_bits == 0]

    if len(samples_high) == 0 or len(samples_low) == 0:
        return -99.0, 0.0, 0.0, n_tx

    # 3. Calculate distinct means and standard deviations
    mu_high = np.mean(samples_high)
    mu_low = np.mean(samples_low)

    sigma_high = np.std(samples_high)
    sigma_low = np.std(samples_low)

    # 4. Calculate the Rigorous Optical Q-factor
    denominator = sigma_high + sigma_low
    if denominator == 0:
        return 99.0, 999999.0, 999.0, n_tx

    Q_factor = (mu_high - mu_low) / denominator

    # 5. Effective SNR derived from physical capacity
    snr_linear = Q_factor ** 2
    snr_db = 10 * np.log10(snr_linear)

    bit_errors = int(np.sum(tx_bits != rx_bits_aligned))

    return snr_db, snr_linear, Q_factor, bit_errors


def calculate_isi_robust_snr(tx_bits, analog_samples):
    """
    Calculates the true stochastic SNR and Inner Eye Opening by isolating
    Inter-Symbol Interference (ISI) using 3-bit pattern context.
    """
    tx_bits = np.array(tx_bits, dtype=int)
    analog_samples = np.array(analog_samples, dtype=float)
    n_tx = len(tx_bits)

    # 1. Basic Synchronization (Find the best offset using cross-correlation)
    v_thresh = np.mean(analog_samples)
    rx_bits_hard = (analog_samples > v_thresh).astype(int)
    tx_bipolar = 2 * tx_bits - 1
    rx_bipolar = 2 * rx_bits_hard - 1
    correlation = np.correlate(rx_bipolar, tx_bipolar, mode="valid")
    best_offset = int(np.argmax(correlation))

    analog_aligned = analog_samples[best_offset: best_offset + n_tx]

    # 2. Group samples by 3-bit context: [bit_{k-1}, bit_{k}, bit_{k+1}]
    pattern_groups = {(b1, b2, b3): [] for b1 in (0, 1) for b2 in (0, 1) for b3 in (0, 1)}

    for k in range(1, n_tx - 1):
        pattern = (tx_bits[k - 1], tx_bits[k], tx_bits[k + 1])
        sample_value = analog_aligned[k]
        pattern_groups[pattern].append(sample_value)

    # 3. Analyze each pattern's deterministic mean and stochastic variance
    pattern_means = {}
    pattern_vars = []

    means_for_ones = []
    means_for_zeros = []

    for pattern, samples in pattern_groups.items():
        if len(samples) < 5:  # Ensure statistical relevance
            continue

        p_mean = np.mean(samples)
        p_var = np.var(samples)

        pattern_means[pattern] = p_mean
        pattern_vars.append(p_var)

        if pattern[1] == 1:
            means_for_ones.append(p_mean)
        else:
            means_for_zeros.append(p_mean)

    if not means_for_ones or not means_for_zeros:
        return -99.0, 0.0, 0.0, False

    # 4. Calculate True Stochastic Noise (Strips out deterministic ISI)
    true_noise_power = np.mean(pattern_vars)

    # 5. Calculate Inner Eye Opening
    lowest_one = np.min(means_for_ones)
    highest_zero = np.max(means_for_zeros)
    v_eye = lowest_one - highest_zero

    is_eye_open = v_eye > 0

    # 6. Compute Inner Eye SNR
    if is_eye_open:
        power_signal = v_eye ** 2
        snr_linear = power_signal / true_noise_power
        snr_db = 10 * np.log10(snr_linear)
    else:
        snr_linear = 0.0
        snr_db = -99.0  # Eye is closed due to severe ISI distortion

    return snr_db, snr_linear, v_eye, is_eye_open



if __name__ == '__main__':

    # Original
    calculate_snr()
    # Limitacions
    #
    #     En dividir tot el vector pel rang [v_min, v_max],
    #     qualsevol pic aïllat de soroll (outlier) altera artificialment l'escala de totes les mostres.
    #     La SNR (elèctrica) s'ha de mesurar utilitzant les amplituds de voltatge reals.
    #
    #     np.mean(tx_bits  2) calcula la potència de la seqüència digital ideal (valor constant d'aprox 0.5).
    #     Això ignora l'atenuació real que pateix el senyal òptic en el canal segons distància.
    #
    #     El càlcul del senyal d'error assumeix que el nivell alt és exactament 1.0 i el baix és 0.0,
    #     però el TIA entrega nivells de tensió mitjans concrets.

    # Mètode 2
    calculate_rigorous_optical_snr()
    # Directament de la teoria de comunicacions òptiques IM/DD, on el soroll és asimètric.
    # El shot noise és dependent del senyal (el LED encès genera més que el LED apagat¡).
    # -> Utilitzar el Factor-Q òptic i calcular l'efectiu SNR=Q^2 amb voltatges reals.
    #
    # Aquesta funció funciona a bitrates baixos. Col·lapsa a altes freqüències:
    # quan el senyal es distorsiona pel filtre passa-baix de l'OPT101,
    # les rampes de pujada i baixada s'interpreten erròniament com a soroll aleatori (np.std), fent caure la SNR.

    # Mètode 3
    calculate_isi_robust_snr()
    # Solució pel tema bitrate alt: eye diagram, utilitzar un context de 3 bits para separar el senyal en 8 patrons diferents.
    # Es mesura la variància real dins de cada trajectòria de l'ull,
    # aïllant la distorsió determinista (ISI causada pel TIA).
    # La potència del senyal es calcula com el quadrat de l'obertura interior de l'ull.
    #
    # Limitacions: obligatòri transmetre seqüències de dades llargues i estadísticament equilibrades (codis PRBS)
    # per garantir que els 8 patrons possibles apareguin suficients vegades.
    # No modela el clipping no lineal que passa en el TIA a distàncies curtes.