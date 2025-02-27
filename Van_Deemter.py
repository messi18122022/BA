import numpy as np
import tkinter as tk
from tkinter import filedialog
import os

def calculate_peak_parameters(time, intensity):
    """
    Berechnet aus den übergebenen Zeit- und Intensitätsdaten:
    - die Retentionszeit (Zeit, an der der Maximalwert erreicht wird)
    - die Halbwertsbreite (FWHM), also die Breite des Peaks bei halber Höhe.
    """
    max_idx = np.argmax(intensity)
    peak_time = time[max_idx]
    peak_intensity = intensity[max_idx]
    half_max = peak_intensity / 2.0

    # Suche links vom Maximum
    left_idx = max_idx
    while left_idx > 0 and intensity[left_idx] > half_max:
        left_idx -= 1
    if left_idx == 0:
        t_left = time[0]
    else:
        t1, t2 = time[left_idx], time[left_idx+1]
        i1, i2 = intensity[left_idx], intensity[left_idx+1]
        t_left = t1 + (half_max - i1) * (t2 - t1) / (i2 - i1)

    # Suche rechts vom Maximum
    right_idx = max_idx
    while right_idx < len(intensity) - 1 and intensity[right_idx] > half_max:
        right_idx += 1
    if right_idx == len(intensity) - 1:
        t_right = time[-1]
    else:
        t1, t2 = time[right_idx-1], time[right_idx]
        i1, i2 = intensity[right_idx-1], intensity[right_idx]
        t_right = t1 + (half_max - i1) * (t2 - t1) / (i2 - i1)

    fwhm = t_right - t_left
    return {"Retentionszeit": peak_time, "Halbwertsbreite": fwhm}

def process_file(file_path):
    """
    Liest eine Datei ein, die einen mehrzeiligen Header (hier 7 Zeilen) besitzt
    und anschließend Messdaten enthält (zwei Spalten, getrennt durch Kommas):
      - Spalte 1: Zeit in Minuten
      - Spalte 2: Messwert (z. B. Leitfähigkeit in µS/cm)
    Wird der Fehler 'utf-8' codec auftaucht, wird mit dem Encoding 'latin1' geladen.
    """
    try:
        # Überspringe den 7-zeiligen Header und verwende das Encoding 'latin1'
        data = np.genfromtxt(file_path, delimiter=',', skip_header=7, encoding='latin1')
    except Exception as e:
        print(f"Fehler beim Laden der Datei {file_path}: {e}")
        return None

    if data.ndim == 1 or data.shape[1] < 2:
        print(f"Datei {file_path} hat nicht das erwartete Format (mindestens 2 Spalten).")
        return None

    # Annahme: erste Spalte = Zeit (in Minuten), zweite Spalte = Intensität/Messwert
    time = data[:, 0]
    intensity = data[:, 1]

    return calculate_peak_parameters(time, intensity)

def select_and_process_files():
    """
    Öffnet ein Dialogfenster zur Dateiauswahl, verarbeitet jede Datei
    und gibt ein Dictionary zurück, in dem der Dateiname als Key und
    die berechneten Parameter (Retentionszeit und Halbwertsbreite) als Value gespeichert sind.
    """
    root = tk.Tk()
    root.withdraw()
    
    file_paths = filedialog.askopenfilenames(
        title="Bitte die Dateien auswählen",
        filetypes=[("Text/CSV Dateien", "*.txt *.csv"), ("Alle Dateien", "*.*")]
    )
    
    results = {}
    for file_path in file_paths:
        params = process_file(file_path)
        if params is not None:
            file_name = os.path.basename(file_path)
            results[file_name] = params

    return results

if __name__ == "__main__":
    ergebnisse = select_and_process_files()
    print("Berechnete Parameter für die Injektionen:")
    for file, params in ergebnisse.items():
        print(f"{file}: Retentionszeit = {params['Retentionszeit']:.4f} min, Halbwertsbreite = {params['Halbwertsbreite']:.4f} min")
