#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dieses Skript ermöglicht die Dateiauswahl per Dialog.
Für jede ausgewählte Datei wird der höchste Peak (Peakmaximum) im Chromatogramm gesucht.
Anschließend wird:
  • Das Signal mittels eines Savitzky–Golay-Filters geglättet,
  • Die Retentionszeit (Zeit des Peakmaximums) bestimmt und
  • Die Halbwertsbreite (FWHM) direkt aus dem geglätteten Signal mittels scipy.signal.peak_widths berechnet.
Pro Datei wird eine Seite im PDF-Report erstellt, die zwei Plots enthält:
  1. Das komplette Chromatogramm (mit Originaldaten, geglätteter Kurve und Markierung der FWHM-Grenzen).
  2. Einen stark vergrößerten Ausschnitt des Peakbereichs.
Die berechneten Metriken (Retentionszeit und FWHM) werden als Text eingeblendet.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.signal import peak_widths, savgol_filter
import tkinter as tk
from tkinter import filedialog
import os
import sys

# Datei einlesen (latin-1, um Encoding-Probleme zu vermeiden)
def read_data(filename):
    with open(filename, 'r', encoding='latin-1') as f:
        lines = f.readlines()
    data_start = None
    for i, line in enumerate(lines):
        # Es wird angenommen, dass eine Zeile, die mit "min" beginnt, den Header darstellt.
        if line.strip().startswith("min"):
            data_start = i + 1
            break
    if data_start is None:
        raise ValueError(f"Die Datei {filename} enthält keinen erkennbaren Header 'min,µS/cm'.")
    data = np.genfromtxt(lines[data_start:], delimiter=",")
    if data.ndim == 1 or data.shape[1] < 2:
        raise ValueError(f"Die Datei {filename} enthält keine gültigen 2-spaltigen Daten.")
    return data[:,0], data[:,1]  # t, y

# Hauptfunktion: Verarbeitung der Dateien und Erstellung des PDF-Reports
def process_files(file_list, pdf_filename="Report3.pdf"):
    with PdfPages(pdf_filename) as pdf:
        for file in file_list:
            try:
                t, y = read_data(file)
            except Exception as e:
                print(f"Fehler beim Lesen von {file}: {e}")
                continue

            # Glätten des Signals (Savitzky-Golay-Filter)
            # Das Fenster (window_length) muss ungerade sein und kleiner als die Datenlänge
            window_length = 11 if len(y) >= 11 else (len(y) // 2) * 2 + 1
            y_smooth = savgol_filter(y, window_length=window_length, polyorder=3)

            # Finde das Peakmaximum im geglätteten Signal
            peak_idx = np.argmax(y_smooth)
            retention_time = t[peak_idx]
            peak_val = y_smooth[peak_idx]

            # Berechne die Halbwertsbreite (FWHM) aus dem geglätteten Signal
            # peak_widths liefert die Breite in "Indices" – da t gleichmäßig verteilt ist,
            # multiplizieren wir mit dem Zeitintervall (t[1]-t[0]), um Minuten zu erhalten.
            widths, h_eval, left_ips, right_ips = peak_widths(y_smooth, [peak_idx], rel_height=0.5)
            fwhm = widths[0] * (t[1]-t[0])
            # Die Positionen der FWHM-Grenzen (in t):
            left_bound = t[int(np.floor(left_ips[0]))]  # Näherungsweise
            right_bound = t[int(np.ceil(right_ips[0]))]

            # Für die Visualisierung: Erzeuge einen stark vergrößerten Ausschnitt um den Peak
            zoom_margin = fwhm  # z.B. so, dass ca. eine FWHM- Breite links und rechts angezeigt wird
            zoom_left = max(retention_time - zoom_margin, t[0])
            zoom_right = min(retention_time + zoom_margin, t[-1])
            mask_zoom = (t >= zoom_left) & (t <= zoom_right)
            t_zoom = t[mask_zoom]
            y_zoom = y[mask_zoom]

            # Erzeuge feine Vektoren für die Darstellung der geglätteten Kurve
            t_fine_full = np.linspace(t[0], t[-1], 500)
            y_smooth_full = savgol_filter(y, window_length=window_length, polyorder=3)
            t_fine_zoom = np.linspace(zoom_left, zoom_right, 500)
            y_smooth_zoom = savgol_filter(y[mask_zoom], window_length=window_length if len(y[mask_zoom])>=11 else (len(y[mask_zoom])//2)*2+1, polyorder=3)

            # Erzeuge eine Figur mit 2 Subplots (vertikal)
            fig, axs = plt.subplots(2, 1, figsize=(8, 10))
            fig.suptitle(f"Datei: {os.path.basename(file)}", fontsize=14)

            # Plot 1: Komplettes Chromatogramm
            axs[0].plot(t, y, 'ko', markersize=3, label="Messdaten")
            axs[0].plot(t, y_smooth, 'b-', linewidth=1, label="Geglättet")
            # Markiere die FWHM-Grenzen mit vertikalen Linien
            axs[0].axvline(x=retention_time - fwhm/2, color='r', linestyle='--', label="FWHM-Grenzen")
            axs[0].axvline(x=retention_time + fwhm/2, color='r', linestyle='--')
            axs[0].set_xlabel("Zeit [min]")
            axs[0].set_ylabel("Leitfähigkeit [µS/cm]")
            axs[0].legend()
            axs[0].set_title("Komplettes Chromatogramm")

            # Plot 2: Zoom auf den Peakbereich
            axs[1].plot(t_zoom, y_zoom, 'ko', markersize=3, label="Messdaten (Zoom)")
            axs[1].plot(t_zoom, y_smooth_zoom, 'b-', linewidth=1, label="Geglättet (Zoom)")
            axs[1].axvline(x=retention_time - fwhm/2, color='r', linestyle='--', label="FWHM-Grenzen")
            axs[1].axvline(x=retention_time + fwhm/2, color='r', linestyle='--')
            axs[1].set_xlabel("Zeit [min]")
            axs[1].set_ylabel("Leitfähigkeit [µS/cm]")
            axs[1].legend()
            axs[1].set_title("Zoom auf den Peak")

            # Füge eine Textbox mit den berechneten Metriken hinzu.
            textstr = (
                f"Peak-Maximum (Retentionszeit): {retention_time:.4f} min\n"
                f"Halbwertsbreite (FWHM): {fwhm:.4f} min"
            )
            fig.text(0.15, 0.03, textstr, fontsize=10,
                     bbox=dict(facecolor='white', alpha=0.5))

            pdf.savefig(fig)
            plt.close(fig)
            print(f"Datei {file} ausgewertet: Retentionszeit = {retention_time:.4f} min, FWHM = {fwhm:.4f} min.")
    print(f"\nReport wurde erstellt: {pdf_filename}")

if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()  # Hauptfenster ausblenden
    file_paths = filedialog.askopenfilenames(
        title="Wähle die Dateien aus",
        filetypes=[("Textdateien", "*.txt"), ("Alle Dateien", "*.*")]
    )
    file_list = list(file_paths)
    if not file_list:
        print("Keine Dateien ausgewählt.")
        sys.exit(1)
    process_files(file_list)
