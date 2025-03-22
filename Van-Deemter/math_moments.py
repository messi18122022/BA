import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

def finde_peak_fenster(zeit, signal, schwellen_prozent=0.05):
    """
    Findet den Bereich um den höchsten Peak (Maximum des Signals),
    indem nach links und rechts bis unter einen definierten Schwellenwert
    gegangen wird.
    
    Parameter:
    -----------
    zeit : array-like
        Zeitwerte.
    signal : array-like
        Signalwerte.
    schwellen_prozent : float
        Prozentsatz (zwischen 0 und 1), der für den Abstand zwischen
        Baseline und Peak verwendet wird, um die linke und rechte
        Grenze des Peaks zu bestimmen.
    
    Rückgabe:
    -----------
    left_index, right_index : int
        Index-Grenzen des Peak-Fensters.
    """
    if len(zeit) == 0 or len(signal) == 0:
        return None, None
    
    # Index des globalen Maximums
    i_peak = np.argmax(signal)
    
    # Baseline hier sehr einfach als Minimum des Signals
    baseline = np.min(signal)
    peak_value = signal[i_peak]
    
    # Falls peak_value == baseline, wäre kein echter Peak vorhanden
    if peak_value == baseline:
        return None, None
    
    # Schwellenwert definieren (z.B. 5 % über der Baseline bis zum Peak)
    threshold = baseline + schwellen_prozent * (peak_value - baseline)
    
    # Suche von i_peak nach links, bis das Signal unter threshold fällt
    left_index = i_peak
    while left_index > 0 and signal[left_index] > threshold:
        left_index -= 1
    
    # Suche von i_peak nach rechts, bis das Signal unter threshold fällt
    right_index = i_peak
    while right_index < len(signal) - 1 and signal[right_index] > threshold:
        right_index += 1
    
    return left_index, right_index

def momentanalyse(zeit, signal):
    """
    Führt die Momentanalyse (erster und zweiter Moment) durch:
      - Retentionszeit (gewichteter Mittelwert)
      - Peakbreite (Standardabweichung)
    """
    m0 = np.sum(signal)
    if m0 == 0:
        return np.nan, np.nan
    
    t_mean = np.sum(zeit * signal) / m0
    varianz = np.sum(((zeit - t_mean) ** 2) * signal) / m0
    stdev = np.sqrt(varianz)
    
    return t_mean, stdev

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
            # Daten einlesen (Trennzeichen: Komma)
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
            
            # Teilbereiche für Momentanalyse ausschneiden
            t_slice = zeit[left_idx:right_idx+1]
            s_slice = signal[left_idx:right_idx+1]
            
            # 4) Momentanalyse nur im Peak-Fenster
            rt, pb = momentanalyse(t_slice, s_slice)
            
            # 5) Subplots erstellen: links Gesamt, rechts Zoom
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            fig.suptitle(f"{os.path.basename(datei)}\n"
                         f"Retentionszeit: {rt:.2f} min, Peakbreite: {pb:.2f} min")
            
            # -- Linker Plot: gesamtes Chromatogramm --
            axes[0].plot(zeit, signal, label="Signal", color="blue")
            axes[0].axvspan(zeit[left_idx], zeit[right_idx], color="orange", alpha=0.2, label="Peak-Fenster")
            axes[0].axvline(rt, color="red", linestyle="--", label="Retentionszeit")
            axes[0].axvspan(rt - pb, rt + pb, color="red", alpha=0.1, label="Peakbreite")
            
            axes[0].set_title("Gesamtes Chromatogramm")
            axes[0].set_xlabel("Zeit (min)")
            axes[0].set_ylabel("Signal (µS/cm)")
            axes[0].legend()
            
            # -- Rechter Plot: Zoom auf den Peak --
            axes[1].plot(zeit, signal, label="Signal", color="blue")
            axes[1].axvspan(zeit[left_idx], zeit[right_idx], color="orange", alpha=0.2)
            axes[1].axvline(rt, color="red", linestyle="--", label="Retentionszeit")
            axes[1].axvspan(rt - pb, rt + pb, color="red", alpha=0.1)
            
            # X-Limits: etwas Puffer um den Peak herum
            x_left = zeit[left_idx] - 0.1 * (zeit[right_idx] - zeit[left_idx])
            x_right = zeit[right_idx] + 0.1 * (zeit[right_idx] - zeit[left_idx])
            axes[1].set_xlim(x_left, x_right)
            
            # Y-Limits: ebenfalls etwas Puffer um den Peak
            peak_min = np.min(s_slice)
            peak_max = np.max(s_slice)
            if peak_min != peak_max:
                y_margin = 0.1 * (peak_max - peak_min)
                axes[1].set_ylim(peak_min - y_margin, peak_max + y_margin)
            
            axes[1].set_title("Zoom auf Peak")
            axes[1].set_xlabel("Zeit (min)")
            axes[1].set_ylabel("Signal (µS/cm)")
            axes[1].legend()
            
            plt.tight_layout(rect=[0, 0.03, 1, 0.92])  # etwas Platz für suptitle
            
            # 6) Speichern als PDF
            base_name = os.path.splitext(os.path.basename(datei))[0]
            output_file = os.path.join(output_dir, base_name + ".pdf")
            plt.savefig(output_file)
            plt.close()
            
        except Exception as e:
            messagebox.showerror("Fehler", f"Fehler beim Verarbeiten der Datei '{datei}':\n{e}")
    
    messagebox.showinfo("Fertig", "Alle Plots wurden erfolgreich als PDF-Dateien gespeichert.")

if __name__ == '__main__':
    main()
