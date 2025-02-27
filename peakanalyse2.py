#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dieses Skript ermöglicht die Dateiauswahl per Dialog.
Für jede ausgewählte Datei wird geprüft, ob ein Peak mit Maximum > 28.0 µS/cm vorhanden ist.
Wird ein solcher Peak gefunden, wird:
  • Das Signal mit einem Savitzky–Golay-Filter geglättet.
  • Die Halbwertsbreite (FWHM) direkt aus dem geglätteten Signal mittels scipy.signal.peak_widths berechnet.
  • Die Basispeakbreite ermittelt, indem die t-Positionen (linear interpoliert) bestimmt werden,
    bei denen der geglättete Messwert den Wert 28.0 µS/cm erreicht.
  • Zur Visualisierung wird zusätzlich ein Gauss‑Fit (nur für den Plot) durchgeführt.
Pro Datei wird eine Seite im PDF-Report erstellt, die zwei Plots enthält:
  1. Das komplette Chromatogramm (mit Messdaten, geglätteter Kurve und Gauss‑Fit).
  2. Einen stärker vergrößerten Ausschnitt des Peakbereichs.
Die berechneten Metriken (FWHM und Basispeakbreite) werden als Text eingeblendet.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.optimize import curve_fit
from scipy.signal import peak_widths, savgol_filter
import tkinter as tk
from tkinter import filedialog
import os
import sys

# Gauss-Modell zur Visualisierung (Fit dient nur der Darstellung)
def gaussian(t, offset, amplitude, t0, sigma):
    return offset + amplitude * np.exp(-((t - t0)**2)/(2*sigma**2))

# Datei einlesen (latin-1, um Encoding-Probleme zu vermeiden)
def read_data(filename):
    with open(filename, 'r', encoding='latin-1') as f:
        lines = f.readlines()
    data_start = None
    for i, line in enumerate(lines):
        if line.strip().startswith("min"):
            data_start = i + 1
            break
    if data_start is None:
        raise ValueError(f"Die Datei {filename} enthält keinen erkennbaren Header 'min,µS/cm'.")
    data = np.genfromtxt(lines[data_start:], delimiter=",")
    if data.ndim == 1 or data.shape[1] < 2:
        raise ValueError(f"Die Datei {filename} enthält keine gültigen 2-spaltigen Daten.")
    return data[:,0], data[:,1]  # t, y

# Berechnet die Schnittpunkte (links und rechts) des geglätteten Signals mit einem gegebenen Schwellenwert
def compute_base_crossings(t, y_smooth, threshold):
    peak_idx = np.argmax(y_smooth)
    # Linke Seite: Suche vom Peakindex rückwärts, bis y unter threshold fällt.
    i = peak_idx
    while i > 0 and y_smooth[i] >= threshold:
        i -= 1
    if i == peak_idx:
        left_cross = t[peak_idx]
    else:
        # Lineare Interpolation zwischen t[i] und t[i+1]:
        left_cross = t[i] + (threshold - y_smooth[i])/(y_smooth[i+1]-y_smooth[i])*(t[i+1]-t[i])
    
    # Rechte Seite: Suche vom Peakindex vorwärts, bis y unter threshold fällt.
    j = peak_idx
    while j < len(y_smooth)-1 and y_smooth[j] >= threshold:
        j += 1
    if j == peak_idx:
        right_cross = t[peak_idx]
    else:
        right_cross = t[j-1] + (y_smooth[j-1]-threshold)/(y_smooth[j-1]-y_smooth[j])*(t[j]-t[j-1])
    
    return left_cross, right_cross

# Hauptfunktion: Verarbeitung der Dateien und Erstellung des PDF-Reports
def process_files(file_list, pdf_filename="Report2.pdf"):
    with PdfPages(pdf_filename) as pdf:
        for file in file_list:
            try:
                t, y = read_data(file)
            except Exception as e:
                print(f"Fehler beim Lesen von {file}: {e}")
                continue

            if np.max(y) <= 28.0:
                print(f"Datei {file}: Kein Peak mit Signal > 28.0 µS/cm gefunden (max = {np.max(y):.2f}).")
                continue

            # Glätte das Signal mit einem Savitzky-Golay-Filter
            # Wähle ein Fenster (muss ungerade sein) – hier z.B. 11, falls genügend Datenpunkte vorhanden sind
            window_length = 11 if len(y) >= 11 else len(y) // 2 * 2 + 1
            y_smooth = savgol_filter(y, window_length=window_length, polyorder=3)

            peak_idx = np.argmax(y_smooth)
            peak_val = y_smooth[peak_idx]

            # Berechne FWHM aus dem geglätteten Signal mittels peak_widths (rel_height=0.5)
            widths, h_eval, left_ips, right_ips = peak_widths(y_smooth, [peak_idx], rel_height=0.5)
            # Da t üblicherweise gleichmäßig verteilt ist, multiplizieren wir mit (t[1]-t[0])
            fwhm = widths[0] * (t[1]-t[0])

            # Berechne Basispeakbreite (bei 28.0 µS/cm) aus dem geglätteten Signal
            left_cross, right_cross = compute_base_crossings(t, y_smooth, threshold=28.0)
            base_width = right_cross - left_cross

            # Zur Visualisierung: Definiere einen Fit-Bereich um die Basiskreuzungen (mit etwas Rand)
            margin = 0.1 * (t[-1]-t[0])
            fit_left = max(left_cross - margin, t[0])
            fit_right = min(right_cross + margin, t[-1])
            mask_fit = (t >= fit_left) & (t <= fit_right)
            t_fit = t[mask_fit]
            y_fit = y[mask_fit]

            # Startwerte für den Gauss-Fit (nur zur Visualisierung)
            offset_guess = np.min(y_fit)
            amplitude_guess = np.max(y_fit) - offset_guess
            t0_guess = t[peak_idx]
            sigma_guess = (fit_right - fit_left)/4
            p0 = [offset_guess, amplitude_guess, t0_guess, sigma_guess]

            try:
                popt, _ = curve_fit(gaussian, t_fit, y_fit, p0=p0)
            except RuntimeError as err:
                print(f"Fit konnte für {file} nicht berechnet werden: {err}")
                popt = p0

            # Erzeuge feine Vektoren für den Gauss-Fit zur Darstellung
            t_fine_full = np.linspace(t[0], t[-1], 500)
            y_fit_full = gaussian(t_fine_full, *popt)

            # Erzeuge einen stärker vergrößerten Ausschnitt (Zoom) um den Peak
            zoom_factor = 3  # 50 % der Breite zwischen den Basiskreuzungen
            center = t0_guess
            half_width = (right_cross - left_cross) * zoom_factor / 2
            zoom_left = center - half_width
            zoom_right = center + half_width
            mask_zoom = (t >= zoom_left) & (t <= zoom_right)
            t_zoom = t[mask_zoom]
            y_zoom = y[mask_zoom]
            t_fine_zoom = np.linspace(zoom_left, zoom_right, 500)
            y_fit_zoom = gaussian(t_fine_zoom, *popt)

            # Erzeuge eine Figur mit 2 Subplots (vertikal)
            fig, axs = plt.subplots(2, 1, figsize=(8, 10))
            fig.suptitle(f"Datei: {os.path.basename(file)}", fontsize=14)

            # Plot 1: Komplettes Chromatogramm (Originaldaten, geglättete Kurve und Gauss-Fit)
            axs[0].plot(t, y, 'ko', markersize=3, label="Messdaten")
            axs[0].plot(t, y_smooth, 'b-', linewidth=1, label="Geglättet")
            axs[0].plot(t_fine_full, y_fit_full, 'r-', linewidth=2, label="Gauss-Fit")
            axs[0].set_xlabel("Zeit [min]")
            axs[0].set_ylabel("Leitfähigkeit [µS/cm]")
            axs[0].legend()
            axs[0].set_title("Komplettes Chromatogramm")

            # Plot 2: Stärkerer Zoom auf den Peakbereich
            axs[1].plot(t_zoom, y_zoom, 'ko', markersize=3, label="Messdaten (Zoom)")
            axs[1].plot(t_fine_zoom, y_fit_zoom, 'r-', linewidth=2, label="Gauss-Fit (Zoom)")
            axs[1].set_xlabel("Zeit [min]")
            axs[1].set_ylabel("Leitfähigkeit [µS/cm]")
            axs[1].legend()
            axs[1].set_title("Stärkerer Zoom auf den Peak")

            # Füge eine Textbox mit den berechneten Metriken hinzu.
            textstr = (
                f"Peak-Maximum (geglättet): {peak_val:.2f} µS/cm\n"
                f"Halbwertsbreite (FWHM): {fwhm:.4f} min\n"
                f"Basispeakbreite (bei 28.0 µS/cm): {base_width:.4f} min"
            )
            fig.text(0.15, 0.03, textstr, fontsize=10,
                     bbox=dict(facecolor='white', alpha=0.5))

            pdf.savefig(fig)
            plt.close(fig)
            print(f"Datei {file} ausgewertet: FWHM = {fwhm:.4f} min, Basispeakbreite = {base_width:.4f} min.")
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
