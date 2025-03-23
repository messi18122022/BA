import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --------------------- Konfiguration ---------------------
# Ordner, in denen die Messdateien hinterlegt sind
ORDNER_MIT_SÄULE = "Van-Deemter/Kationentauscher/mit_saule"
ORDNER_OHNE_SÄULE = "Van-Deemter/Kationentauscher/ohne_saule"

# Ordner, in dem die Plots gespeichert werden sollen
OUTPUT_DIR = "Van-Deemter"  # Dieser Ordner wird sowohl für den van Deemter Plot als auch für die H-Werte-Rechnung genutzt

# Standard-Flussraten-Vektor (Standard: 2.5, 2.4, 2.3, ..., 0.1)
default_flow_rates = np.arange(2.5, 0.0, -0.1)
# ----------------------------------------------------------

def finde_peak_fenster(zeit, signal, schwellen_prozent=0.05):
    if len(zeit) == 0 or len(signal) == 0:
        return None, None
    i_peak = np.argmax(signal)
    baseline = np.min(signal)
    peak_value = signal[i_peak]
    if peak_value == baseline:
        return None, None
    threshold = baseline + schwellen_prozent * (peak_value - baseline)
    left_index = i_peak
    while left_index > 0 and signal[left_index] > threshold:
        left_index -= 1
    right_index = i_peak
    while right_index < len(signal) - 1 and signal[right_index] > threshold:
        right_index += 1
    return left_index, right_index

def momentanalyse(zeit, signal):
    m0 = np.sum(signal)
    if m0 == 0:
        return np.nan, np.nan
    t_mean = np.sum(zeit * signal) / m0
    varianz = np.sum(((zeit - t_mean) ** 2) * signal) / m0
    stdev = np.sqrt(varianz)
    return t_mean, stdev

def berechne_halbwertsbreite(zeit, signal):
    i_peak = np.argmax(signal)
    peak_value = signal[i_peak]
    baseline = np.min(signal)
    half_max = baseline + (peak_value - baseline) / 2.0
    left_index = i_peak
    while left_index > 0 and signal[left_index] > half_max:
        left_index -= 1
    if left_index == 0:
        t_left = zeit[0]
    else:
        t1, t2 = zeit[left_index], zeit[left_index + 1]
        s1, s2 = signal[left_index], signal[left_index + 1]
        t_left = t1 + (half_max - s1) * (t2 - t1) / (s2 - s1) if s2 != s1 else t1
    right_index = i_peak
    while right_index < len(signal) - 1 and signal[right_index] > half_max:
        right_index += 1
    if right_index == len(signal) - 1:
        t_right = zeit[-1]
    else:
        t1, t2 = zeit[right_index - 1], zeit[right_index]
        s1, s2 = signal[right_index - 1], signal[right_index]
        t_right = t1 + (half_max - s1) * (t2 - t1) / (s2 - s1) if s2 != s1 else t2
    return t_left, t_right

def process_files(filepaths, beschriftung):
    tR_list = []
    fwhm_list = []
    for datei in filepaths:
        try:
            df = pd.read_csv(datei, delimiter=',')
            if df.shape[1] < 2:
                print(f"Warnung: Die Datei '{datei}' hat nicht genügend Spalten.")
                continue
            zeit = df.iloc[:, 0].values
            signal = df.iloc[:, 1].values
            left_idx, right_idx = finde_peak_fenster(zeit, signal, schwellen_prozent=0.05)
            if left_idx is None or right_idx is None:
                print(f"Warnung: Kein Peak in '{datei}' gefunden.")
                continue
            t_slice = zeit[left_idx:right_idx+1]
            s_slice = signal[left_idx:right_idx+1]
            t_mean, _ = momentanalyse(t_slice, s_slice)
            t_left, t_right = berechne_halbwertsbreite(t_slice, s_slice)
            fwhm = t_right - t_left
            tR_list.append(t_mean)
            fwhm_list.append(fwhm)
            # Erzeuge Plot: Gesamtes Chromatogramm und Zoom auf den Peak
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            fig.suptitle(f"{os.path.basename(datei)}\nRetentionszeit: {t_mean:.2f} min, FWHM: {fwhm:.2f} min")
            axes[0].plot(zeit, signal, label="Signal", color="blue")
            axes[0].axvspan(zeit[left_idx], zeit[right_idx], color="orange", alpha=0.2, label="Peak-Fenster")
            axes[0].axvline(t_mean, color="red", linestyle="--", label="Retentionszeit")
            axes[0].axvspan(t_left, t_right, color="red", alpha=0.1, label="FWHM")
            axes[0].set_title("Gesamtes Chromatogramm")
            axes[0].set_xlabel("Zeit (min)")
            axes[0].set_ylabel("Signal (µS/cm)")
            axes[0].legend()
            axes[1].plot(zeit, signal, label="Signal", color="blue")
            axes[1].axvspan(zeit[left_idx], zeit[right_idx], color="orange", alpha=0.2)
            axes[1].axvline(t_mean, color="red", linestyle="--", label="Retentionszeit")
            axes[1].axvspan(t_left, t_right, color="red", alpha=0.1, label="FWHM")
            x_left = zeit[left_idx] - 0.1 * (zeit[right_idx] - zeit[left_idx])
            x_right = zeit[right_idx] + 0.1 * (zeit[right_idx] - zeit[left_idx])
            axes[1].set_xlim(x_left, x_right)
            peak_min = np.min(s_slice)
            peak_max = np.max(s_slice)
            if peak_min != peak_max:
                y_margin = 0.1 * (peak_max - peak_min)
                axes[1].set_ylim(peak_min - y_margin, peak_max + y_margin)
            axes[1].set_title("Zoom auf Peak")
            axes[1].set_xlabel("Zeit (min)")
            axes[1].set_ylabel("Signal (µS/cm)")
            axes[1].legend()
            plt.tight_layout(rect=[0, 0.03, 1, 0.92])
            base_name = os.path.splitext(os.path.basename(datei))[0]
            output_path = os.path.join(os.path.dirname(datei), base_name + ".pdf")
            plt.savefig(output_path)
            plt.close()
        except Exception as e:
            print(f"Fehler beim Verarbeiten der Datei '{datei}': {e}")
    print(f"Alle {beschriftung} Plots wurden als PDF gespeichert.")
    return tR_list, fwhm_list

def erstelle_hwerte_pdf(tR_total_list, fwhm_total_list, tR_extra_list, fwhm_extra_list,
                         tR_column_list, W_column_list, N_list, HETP_list, output_dir):
    """
    Erzeugt ein PDF, in dem der Rechenweg für die H-Werte (Säulenparameter) für jede Messung
    detailliert aufgeführt wird.
    """
    # Erstelle Datenzeilen für die Tabelle
    daten = []
    for i in range(len(tR_total_list)):
        # Formatierung der Werte (auf 3 Nachkommastellen)
        messung = i + 1
        tR_total = f"{tR_total_list[i]:.3f}"
        fwhm_total = f"{fwhm_total_list[i]:.3f}"
        tR_extra = f"{tR_extra_list[i]:.3f}"
        fwhm_extra = f"{fwhm_extra_list[i]:.3f}"
        tR_column = f"{tR_column_list[i]:.3f}"
        W_column = f"{W_column_list[i]:.3f}"
        N_val = f"{N_list[i]:.3f}"
        HETP = f"{HETP_list[i]:.3f}"
        daten.append([messung, tR_total, fwhm_total, tR_extra, fwhm_extra, tR_column, W_column, N_val, HETP])
    
    # Spaltenüberschriften
    spalten = ["Messung", "tR_total (min)", "FWHM_total (min)", "tR_extra (min)",
               "FWHM_extra (min)", "tR_column (min)", "W_column (min)", "N", "HETP (mm)"]
    
    # Erstelle eine neue Figur und füge die Tabelle ein
    fig, ax = plt.subplots(figsize=(12, 0.5 * (len(daten)+2)))
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=daten, colLabels=spalten, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    plt.title("Nachvollziehbare Berechnungen der H-Werte\nRechenweg: tR_column = tR_total - tR_extra,  W_column = sqrt(fwhm_total² - fwhm_extra²),  N = (tR_column / W_column)²,  HETP = 150 mm / N", fontsize=12)
    
    output_file = os.path.join(output_dir, "Hwerte_Berechnungen.pdf")
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    print("Das PDF mit den H-Werte-Berechnungen wurde erfolgreich erstellt und gespeichert.")

def main():
    # Dateien aus den angegebenen Ordnern laden
    files_with_column = sorted(
        glob.glob(os.path.join(ORDNER_MIT_SÄULE, "*.csv")) + 
        glob.glob(os.path.join(ORDNER_MIT_SÄULE, "*.txt"))
    )
    if not files_with_column:
        print("Es wurden keine Dateien für Messungen mit Säule gefunden.")
        return
    tR_total_list, fwhm_total_list = process_files(files_with_column, "mit Säule")

    files_without_column = sorted(
        glob.glob(os.path.join(ORDNER_OHNE_SÄULE, "*.csv")) + 
        glob.glob(os.path.join(ORDNER_OHNE_SÄULE, "*.txt"))
    )
    if not files_without_column:
        print("Es wurden keine Dateien für Messungen ohne Säule gefunden.")
        return
    tR_extra_list, fwhm_extra_list = process_files(files_without_column, "ohne Säule")

    if len(tR_total_list) != len(tR_extra_list):
        print("Die Anzahl der Messungen mit und ohne Säule stimmt nicht überein.")
        return

    # Berechnung der Säulenparameter und Bodenzahl
    tR_column_list = []
    W_column_list = []
    N_list = []
    HETP_list = []  # HETP = Säulenlänge (150 mm) / N
    for i in range(len(tR_total_list)):
        tR_column = tR_total_list[i] - tR_extra_list[i]
        try:
            W_column = np.sqrt(fwhm_total_list[i]**2 - fwhm_extra_list[i]**2)
        except Exception as e:
            print(f"Fehler bei der Berechnung der korrigierten Peakbreite für Messung {i+1}: {e}")
            continue
        if W_column <= 0:
            print(f"Warnung: Für Messung {i+1} ergibt sich eine nicht sinnvolle Peakbreite (W_column={W_column}).")
            continue
        N = (tR_column / W_column) ** 2
        HETP = 150 / N
        tR_column_list.append(tR_column)
        W_column_list.append(W_column)
        N_list.append(N)
        HETP_list.append(HETP)

    n_measurements = len(HETP_list)
    if n_measurements == 0:
        print("Keine gültigen Messungen vorhanden für die Flussratenzuordnung.")
        return

    # Bestimmung des Flussraten-Vektors
    if len(default_flow_rates) == n_measurements:
        flow_rates = default_flow_rates
    else:
        flow_rates = np.linspace(default_flow_rates[0], default_flow_rates[-1], n_measurements)

    # van Deemter Plot: HETP vs. Flussrate
    plt.figure(figsize=(8, 6))
    plt.plot(flow_rates, HETP_list, marker="o", linestyle="-")
    plt.xlabel("Flussrate (mL/min)")
    plt.ylabel("HETP (mm)")
    plt.title("van Deemter Plot\n(HETP vs. Flussrate)")
    plt.grid(True)
    plt.tight_layout()
    van_deemter_output_file = os.path.join(OUTPUT_DIR, "van_Deemter_Plot.pdf")
    plt.savefig(van_deemter_output_file)
    plt.close()
    print("Der van Deemter Plot wurde erfolgreich erstellt und gespeichert.")

    # Erzeuge das zweite PDF mit detaillierten H-Werte-Berechnungen
    erstelle_hwerte_pdf(tR_total_list, fwhm_total_list, tR_extra_list, fwhm_extra_list,
                         tR_column_list, W_column_list, N_list, HETP_list, OUTPUT_DIR)

if __name__ == '__main__':
    main()