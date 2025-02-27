#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dieses Skript ermöglicht die Auswahl von Dateien über ein Dialogfenster.
Für jede ausgewählte Datei wird geprüft, ob ein Peak vorhanden ist,
dessen Maximum > 28.0 µS/cm beträgt.
Wird ein solcher Peak gefunden, so wird der gesamte Peak (von der Basislinie bis zur Spitze)
analysiert, ein Gauss‑Fit (mit Offset) durchgeführt und folgende Metriken berechnet:
    • Halbwertsbreite (FWHM)
    • Basispeakbreite (Abstand zwischen den Schnittpunkten des Fits mit 28.0 µS/cm)
Pro Datei wird eine Seite im PDF‑Report erzeugt, die zwei Plots enthält:
    1. Komplette Darstellung des Chromatogramms mit Fit
    2. Vergrößerte (stärker gezoomte) Ansicht des Peakbereichs
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.optimize import curve_fit
import tkinter as tk
from tkinter import filedialog
import os
import sys

# Gauss-Modell mit Offset
def gaussian(x, offset, amplitude, t0, sigma):
    return offset + amplitude * np.exp(-((x - t0)**2)/(2*sigma**2))

# Funktion zum Einlesen der Datei.
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
    return data[:,0], data[:,1]

# Bestimmt den Peakbereich, indem vom Maximum aus in beide Richtungen gesucht wird,
# bis der Messwert nahe der Basis (baseline + tol * Amplitude) liegt.
def find_peak_bounds(x, y, baseline, tol=0.05):
    imax = np.argmax(y)
    amplitude = np.max(y) - baseline
    # Suche links vom Maximum:
    left = imax
    while left > 0 and y[left] > baseline + tol * amplitude:
        left -= 1
    # Suche rechts vom Maximum:
    right = imax
    while right < len(y) - 1 and y[right] > baseline + tol * amplitude:
        right += 1
    return left, right

# Berechnet die Halbwertsbreite (FWHM) und Basispeakbreite (bei level_base)
def compute_widths(offset, amplitude, t0, sigma, level_base=28.0):
    fwhm = 2 * sigma * np.sqrt(2 * np.log(2))
    if not (offset < level_base < offset + amplitude):
        base_width = np.nan
    else:
        ratio = (level_base - offset) / amplitude
        base_width = 2 * sigma * np.sqrt(-2 * np.log(ratio))
    return fwhm, base_width

# Hauptfunktion: Verarbeitung der ausgewählten Dateien und Erzeugung des PDF-Reports.
def process_files(file_list, pdf_filename="Report.pdf"):
    with PdfPages(pdf_filename) as pdf:
        for file in file_list:
            try:
                x, y = read_data(file)
            except Exception as e:
                print(f"Fehler beim Lesen von {file}: {e}")
                continue

            # Nur Peaks mit Maximum > 28.0 µS/cm werden analysiert.
            if np.max(y) <= 28.0:
                print(f"Datei {file}: Kein Peak mit Signal > 28.0 µS/cm gefunden (max = {np.max(y):.2f}).")
                continue

            # Basislinie als Minimum der Messwerte
            baseline = np.min(y)
            # Bestimme den Peakbereich (von Basis bis Maximum) mit einer Toleranz.
            left_idx, right_idx = find_peak_bounds(x, y, baseline, tol=0.05)
            x_fit = x[left_idx:right_idx+1]
            y_fit = y[left_idx:right_idx+1]

            # Erste Schätzungen für den Fit:
            offset_guess = np.min(y_fit)
            amplitude_guess = np.max(y_fit) - offset_guess
            t0_guess = x_fit[np.argmax(y_fit)]
            sigma_guess = (x_fit[-1] - x_fit[0]) / 4
            p0 = [offset_guess, amplitude_guess, t0_guess, sigma_guess]

            try:
                popt, pcov = curve_fit(gaussian, x_fit, y_fit, p0=p0)
            except RuntimeError as err:
                print(f"Fit konnte für {file} nicht berechnet werden: {err}")
                continue

            offset, amplitude, t0, sigma = popt
            fwhm, base_width = compute_widths(offset, amplitude, t0, sigma, level_base=28.0)

            # Erzeuge feine x-Vektoren für den Fit:
            # Vollständiger Bereich:
            x_fine_full = np.linspace(x[0], x[-1], 500)
            y_fit_full = gaussian(x_fine_full, *popt)
            
            # Bestimme einen stärker gezoomten Bereich rund um das Peakmaximum:
            # Hier definieren wir, wie viel vom ursprünglich ermittelten Peakbereich (x_fit) angezeigt werden soll.
            zoom_factor = 0.1  # 50% des ursprünglichen Peakbereichs
            peak_width = x_fit[-1] - x_fit[0]
            half_zoom = (peak_width * zoom_factor) / 2
            zoom_left = t0 - half_zoom
            zoom_right = t0 + half_zoom

            # Filtere die Messdaten im Zoom-Bereich:
            mask_zoom = (x_fit >= zoom_left) & (x_fit <= zoom_right)
            x_zoom = x_fit[mask_zoom]
            y_zoom = y_fit[mask_zoom]

            # Feiner Vektor für den Fit im Zoom-Bereich:
            x_fine_zoom = np.linspace(zoom_left, zoom_right, 500)
            y_fine_zoom = gaussian(x_fine_zoom, *popt)

            # Erzeuge eine Figur mit 2 Subplots (vertikal)
            fig, axs = plt.subplots(2, 1, figsize=(8, 10))
            fig.suptitle(f"Datei: {os.path.basename(file)}", fontsize=14)

            # Erster Plot: komplettes Chromatogramm mit Fit
            axs[0].plot(x, y, 'ko', markersize=3, label="Messdaten")
            axs[0].plot(x_fine_full, y_fit_full, 'r-', linewidth=2, label="Gauss-Fit")
            axs[0].set_xlabel("Zeit [min]")
            axs[0].set_ylabel("Leitfähigkeit [µS/cm]")
            axs[0].legend()
            axs[0].set_title("Komplettes Chromatogramm")

            # Zweiter Plot: starker Zoom auf den Peakbereich
            axs[1].plot(x_zoom, y_zoom, 'ko', markersize=3, label="Messdaten (Zoom)")
            axs[1].plot(x_fine_zoom, y_fine_zoom, 'r-', linewidth=2, label="Gauss-Fit (Zoom)")
            axs[1].set_xlabel("Zeit [min]")
            axs[1].set_ylabel("Leitfähigkeit [µS/cm]")
            axs[1].legend()
            axs[1].set_title("Stärkerer Zoom auf den Peak")

            # Füge eine Textbox mit den Kennzahlen hinzu (unten in der Figur)
            textstr = (
                f"Peak-Maximum: {offset + amplitude:.2f} µS/cm\n"
                f"Halbwertsbreite (FWHM): {fwhm:.4f} min\n"
                f"Basispeakbreite (bei 28.0 µS/cm): {base_width:.4f} min"
            )
            fig.text(0.15, 0.03, textstr, fontsize=10,
                     bbox=dict(facecolor='white', alpha=0.5))

            # Füge die aktuelle Figur als Seite dem PDF hinzu.
            pdf.savefig(fig)
            plt.close(fig)
            print(f"Datei {file} ausgewertet: FWHM = {fwhm:.4f} min, Basispeakbreite = {base_width:.4f} min.")

    print(f"\nReport wurde erstellt: {pdf_filename}")

# Hauptprogramm: Öffnet ein Dialogfenster zur Dateiauswahl und startet die Analyse.
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
