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
    und den zweiten Moment (Standardabweichung).
    Diese Methode setzt jedoch symmetrische Peaks voraus.
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
    Dazu wird der Punkt gesucht, an dem das Signal 50% seines 
    (peak-baseline)-Werts erreicht – links und rechts vom Peak.
    Lineare Interpolation sorgt für genauere Zeitwerte.
    """
    i_peak = np.argmax(signal)
    peak_value = signal[i_peak]
    baseline = np.min(signal)
    
    # Halbwert des Peaks (50 % zwischen Baseline und Peak)
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

def main():
    # Tkinter-Hauptfenster ausblenden
    root = tk.Tk()
    root.withdraw()
    
    # 1) Dateien auswählen
    dateipfade = filedialog.askopenfilenames(
        title="Wählen Sie Ihre TXT-/CSV-Dateien aus",
        filetypes=[("Textdateien", "*.txt"), ("CSV-Dateien", "*.csv"), ("Alle Dateien", "*.*")]
    )
    if not dateipfade:
        messagebox.showinfo("Keine Auswahl", "Es wurden keine Dateien ausgewählt.")
        return
    
    # 2) Zielordner für PDF-Ausgabe wählen
    output_dir = filedialog.askdirectory(title="Zielordner für PDF-Dateien wählen")
    if not output_dir:
        messagebox.showinfo("Keine Auswahl", "Es wurde kein Ausgabeverzeichnis ausgewählt.")
        return
    
    for datei in dateipfade:
        try:
            # Daten einlesen
            df = pd.read_csv(datei, delimiter=',')
            if df.shape[1] < 2:
                messagebox.showwarning("Fehler", f"Die Datei '{datei}' hat nicht genügend Spalten.")
                continue
            
            # Zeit und Signal extrahieren
            zeit = df.iloc[:, 0].values
            signal = df.iloc[:, 1].values
            
            # 3) Peak-Fenster suchen
            left_idx, right_idx = finde_peak_fenster(zeit, signal, schwellen_prozent=0.05)
            if left_idx is None or right_idx is None:
                messagebox.showwarning("Warnung", f"Kein Peak in '{datei}' gefunden.")
                continue
            
            # Teilbereiche für Analyse ausschneiden
            t_slice = zeit[left_idx:right_idx+1]
            s_slice = signal[left_idx:right_idx+1]
            
            # 4) Berechne Retentionszeit mittels Momentanalyse (optional)
            t_mean, _ = momentanalyse(t_slice, s_slice)
            
            # 5) Berechne Halbwertsbreite (FWHM) für asymmetrische Peaks
            t_left, t_right = berechne_halbwertsbreite(t_slice, s_slice)
            fwhm = t_right - t_left
            
            # 6) Subplots erstellen: links Gesamt, rechts Zoom
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            fig.suptitle(f"{os.path.basename(datei)}\n"
                         f"Retentionszeit: {t_mean:.2f} min, FWHM: {fwhm:.2f} min")
            
            # Linker Plot: gesamtes Chromatogramm
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
            
            # Y-Achse anpassen
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
            
            # 7) Speichern als PDF
            base_name = os.path.splitext(os.path.basename(datei))[0]
            output_file = os.path.join(output_dir, base_name + ".pdf")
            plt.savefig(output_file)
            plt.close()
            
        except Exception as e:
            messagebox.showerror("Fehler", f"Fehler beim Verarbeiten der Datei '{datei}':\n{e}")
    
    messagebox.showinfo("Fertig", "Alle Plots wurden erfolgreich als PDF-Dateien gespeichert.")

if __name__ == '__main__':
    main()