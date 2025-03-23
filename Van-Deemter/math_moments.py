import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

def finde_peak_fenster(zeit, signal, schwellen_prozent=0.05):
    """
    Bestimmt den Bereich um das globale Maximum (Peak),
    indem von dort nach links und rechts gesucht wird, bis das Signal
    unter einen definierten Schwellenwert fällt.
    """
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
    """
    Berechnet den ersten Moment (gewichteter Mittelwert, Retentionszeit)
    und den zweiten Moment (Standardabweichung) direkt aus den Rohdaten.
    """
    m0 = np.sum(signal)
    if m0 == 0:
        return np.nan, np.nan
    t_mean = np.sum(zeit * signal) / m0
    varianz = np.sum(((zeit - t_mean) ** 2) * signal) / m0
    stdev = np.sqrt(varianz)
    return t_mean, stdev

def berechne_halbwertsbreite(zeit, signal):
    """
    Ermittelt die Halbwertsbreite (FWHM) des Peaks.
    Es wird der Punkt gesucht, an dem das Signal 50 % seines
    (Peak - Baseline)-Wertes erreicht – links und rechts vom Peak.
    Mittels linearer Interpolation wird ein genauerer Wert ermittelt.
    """
    i_peak = np.argmax(signal)
    peak_value = signal[i_peak]
    baseline = np.min(signal)
    half_max = baseline + (peak_value - baseline) / 2.0

    # Links vom Peak
    left_index = i_peak
    while left_index > 0 and signal[left_index] > half_max:
        left_index -= 1
    if left_index == 0:
        t_left = zeit[0]
    else:
        t1, t2 = zeit[left_index], zeit[left_index + 1]
        s1, s2 = signal[left_index], signal[left_index + 1]
        if s2 != s1:
            t_left = t1 + (half_max - s1) * (t2 - t1) / (s2 - s1)
        else:
            t_left = t1

    # Rechts vom Peak
    right_index = i_peak
    while right_index < len(signal) - 1 and signal[right_index] > half_max:
        right_index += 1
    if right_index == len(signal) - 1:
        t_right = zeit[-1]
    else:
        t1, t2 = zeit[right_index - 1], zeit[right_index]
        s1, s2 = signal[right_index - 1], signal[right_index]
        if s2 != s1:
            t_right = t1 + (half_max - s1) * (t2 - t1) / (s2 - s1)
        else:
            t_right = t2

    return t_left, t_right

def process_files(filepaths, dialog_title):
    """
    Liest die Dateien ein, berechnet für jede Datei:
      - t_R (mittels Momentanalyse)
      - FWHM (über die Halbwertsbreiten-Funktion)
    und speichert den entsprechenden Plot (Gesamt und Zoom) als PDF
    im gleichen Ordner wie die Eingabedatei.
    Gibt zwei Listen zurück: eine für die Retentionszeiten und eine für die FWHM.
    """
    tR_list = []
    fwhm_list = []

    for datei in filepaths:
        try:
            df = pd.read_csv(datei, delimiter=',')
            if df.shape[1] < 2:
                messagebox.showwarning("Fehler", f"Die Datei '{datei}' hat nicht genügend Spalten.")
                continue

            # Zeit und Signal extrahieren
            zeit = df.iloc[:, 0].values
            signal = df.iloc[:, 1].values

            # Peak-Fenster bestimmen
            left_idx, right_idx = finde_peak_fenster(zeit, signal, schwellen_prozent=0.05)
            if left_idx is None or right_idx is None:
                messagebox.showwarning("Warnung", f"Kein Peak in '{datei}' gefunden.")
                continue

            # Auswahl des relevanten Peak-Bereichs
            t_slice = zeit[left_idx:right_idx+1]
            s_slice = signal[left_idx:right_idx+1]

            # Berechnung der Retentionszeit (erster Moment)
            t_mean, _ = momentanalyse(t_slice, s_slice)
            # Berechnung der Halbwertsbreite (FWHM)
            t_left, t_right = berechne_halbwertsbreite(t_slice, s_slice)
            fwhm = t_right - t_left

            tR_list.append(t_mean)
            fwhm_list.append(fwhm)

            # Plot erstellen: Gesamt und Zoom
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            fig.suptitle(f"{os.path.basename(datei)}\n"
                         f"Retentionszeit: {t_mean:.2f} min, FWHM: {fwhm:.2f} min")

            # Linker Plot: Gesamtes Chromatogramm
            axes[0].plot(zeit, signal, label="Signal", color="blue")
            axes[0].axvspan(zeit[left_idx], zeit[right_idx], color="orange", alpha=0.2, label="Peak-Fenster")
            axes[0].axvline(t_mean, color="red", linestyle="--", label="Retentionszeit")
            axes[0].axvspan(t_left, t_right, color="red", alpha=0.1, label="FWHM")
            axes[0].set_title("Gesamtes Chromatogramm")
            axes[0].set_xlabel("Zeit (min)")
            axes[0].set_ylabel("Signal (µS/cm)")
            axes[0].legend()

            # Rechter Plot: Zoom auf den Peak
            axes[1].plot(zeit, signal, label="Signal", color="blue")
            axes[1].axvspan(zeit[left_idx], zeit[right_idx], color="orange", alpha=0.2)
            axes[1].axvline(t_mean, color="red", linestyle="--", label="Retentionszeit")
            axes[1].axvspan(t_left, t_right, color="red", alpha=0.1, label="FWHM")
            x_left = zeit[left_idx] - 0.1 * (zeit[right_idx] - zeit[left_idx])
            x_right = zeit[right_idx] + 0.1 * (zeit[right_idx] - zeit[left_idx])
            axes[1].set_xlim(x_left, x_right)

            # Anpassung der Y-Achse im Zoom-Plot
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
            
            # Speichern des Plots im gleichen Ordner wie die Datei
            base_name = os.path.splitext(os.path.basename(datei))[0]
            output_path = os.path.join(os.path.dirname(datei), base_name + ".pdf")
            plt.savefig(output_path)
            plt.close()

        except Exception as e:
            messagebox.showerror("Fehler", f"Fehler beim Verarbeiten der Datei '{datei}':\n{e}")

    messagebox.showinfo("Fertig", f"Alle {dialog_title} Plots wurden als PDF gespeichert.")
    return tR_list, fwhm_list

def main():
    # Tkinter-Hauptfenster ausblenden
    root = tk.Tk()
    root.withdraw()

    # 1) Dialog: Messungen mit Säule
    files_with_column = filedialog.askopenfilenames(
        title="Wählen Sie die TXT-/CSV-Dateien für Messungen MIT Säule",
        filetypes=[("Textdateien", "*.txt"), ("CSV-Dateien", "*.csv"), ("Alle Dateien", "*.*")]
    )
    if not files_with_column:
        messagebox.showinfo("Keine Auswahl", "Es wurden keine Dateien für Messungen mit Säule ausgewählt.")
        return

    tR_total_list, fwhm_total_list = process_files(files_with_column, "mit Säule")

    # 2) Dialog: Messungen ohne Säule (extrakolumnar)
    files_without_column = filedialog.askopenfilenames(
        title="Wählen Sie die TXT-/CSV-Dateien für Messungen OHNE Säule",
        filetypes=[("Textdateien", "*.txt"), ("CSV-Dateien", "*.csv"), ("Alle Dateien", "*.*")]
    )
    if not files_without_column:
        messagebox.showinfo("Keine Auswahl", "Es wurden keine Dateien für Messungen ohne Säule ausgewählt.")
        return

    tR_extra_list, fwhm_extra_list = process_files(files_without_column, "ohne Säule")

    # Überprüfen, ob beide Gruppen gleich viele Messungen haben
    if len(tR_total_list) != len(tR_extra_list):
        messagebox.showerror("Fehler", "Die Anzahl der Messungen mit und ohne Säule stimmt nicht überein.")
        return

    # Berechnung der Säulenparameter und Bodenzahl für jede Messung
    tR_column_list = []
    W_column_list = []
    N_list = []
    HETP_list = []  # HETP = Säulenlänge / N, Säulenlänge = 150 mm

    for i in range(len(tR_total_list)):
        # Korrigierte Retentionszeit
        tR_column = tR_total_list[i] - tR_extra_list[i]
        # Korrigierte Peakbreite: W_column = sqrt(W_total^2 - W_extra^2)
        # Hier: fwhm_total_list und fwhm_extra_list
        try:
            W_column = np.sqrt(fwhm_total_list[i]**2 - fwhm_extra_list[i]**2)
        except Exception as e:
            messagebox.showerror("Fehler", f"Fehler bei der Berechnung der korrigierten Peakbreite für Messung {i+1}: {e}")
            continue

        # Vermeidung von Division durch 0 oder negative Werte
        if W_column <= 0:
            messagebox.showwarning("Warnung", f"Für Messung {i+1} ergibt sich eine nicht sinnvolle Peakbreite (W_column={W_column}).")
            continue

        # Berechnung der Bodenzahl
        N = (tR_column / W_column) ** 2
        # Berechnung der HETP (Bodenhöhe): H = Säulenlänge / N
        HETP = 150 / N

        tR_column_list.append(tR_column)
        W_column_list.append(W_column)
        N_list.append(N)
        HETP_list.append(HETP)

    # 3) Dialog: Zielordner für den van Deemter Plot
    van_deemter_output_dir = filedialog.askdirectory(
        title="Zielordner für den van Deemter Plot wählen"
    )
    if not van_deemter_output_dir:
        messagebox.showinfo("Keine Auswahl", "Es wurde kein Ausgabeverzeichnis für den van Deemter Plot ausgewählt.")
        return

    # Erstellen des van Deemter Plots
    # (Hier verwenden wir als x-Achse einfach die Messungsnummer, da z.B. der Flussratenparameter nicht vorliegt.)
    x_values = np.arange(1, len(HETP_list) + 1)
    plt.figure(figsize=(8, 6))
    plt.plot(x_values, HETP_list, marker="o", linestyle="-")
    plt.xlabel("Messung (Index)")
    plt.ylabel("HETP (mm)")  # HETP = Säulenlänge / N
    plt.title("van Deemter Plot\n(HETP vs. Messung)")
    plt.grid(True)
    plt.tight_layout()

    # Speichern des van Deemter Plots als PDF
    van_deemter_output_file = os.path.join(van_deemter_output_dir, "van_Deemter_Plot.pdf")
    plt.savefig(van_deemter_output_file)
    plt.close()

    messagebox.showinfo("Fertig", "Alle Plots wurden erfolgreich erstellt und gespeichert.")

if __name__ == '__main__':
    main()