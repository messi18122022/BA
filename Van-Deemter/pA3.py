#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dieses Skript ermöglicht das Einlesen von Chromatogramm-Daten für van Deemter Analysen.
Zuerst werden die Messungen mit der Säule ausgewählt, anschließend die Messungen ohne Säule.
Die Reihenfolge in beiden Dialogfenstern muss übereinstimmen, damit die extrakolumnären Einflüsse
korrekt abgezogen werden können.

Berechnungsgrundlagen:
  • Für jedes Chromatogramm wird die Halbwertsbreite (FWHM) ermittelt.
  • Die Umrechnung in sigma erfolgt über: w₁/₂ = σ · √(8·ln2)
  • Es gilt: σ_obs² = σ_col² + σ_ext²,
    wobei σ_obs aus der Messung mit Säule und σ_ext aus der Messung ohne Säule berechnet wird.
  • Daraus folgt: σ_col = √(σ_obs² - σ_ext²) und w₁/₂ (Säule) = σ_col · √(8·ln2)
  
  Zusätzlich wird eine Gaussian-Kurve (rot) gefittet, um die Peakform zu beurteilen.
  Der Peak Gaussian Factor (PGF) wird berechnet nach:
  
      PGF = 1.83 · (w₁/₂ / w₁/10)
      
  Dabei sollte PGF idealerweise zwischen 0.8 und 1.15 liegen.
  
Pro Dateipaar wird eine Seite im PDF-Report erstellt, die ein 3×2-Raster enthält:
  - 1. Zeile:   Mit Säule (links: Komplett, rechts: Zoom)
  - 2. Zeile:   Ohne Säule (links: Komplett, rechts: Zoom)
  - 3. Zeile:   Enthält eine Textbox mit den berechneten Ergebnissen (FWHM, PGF etc.)
Auf der letzten Seite wird eine Tabelle dargestellt, in der nur noch die Dateinamen und die
FWHM (Säule) ausgegeben werden.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.signal import peak_widths, savgol_filter
from scipy.optimize import curve_fit
import tkinter as tk
from tkinter import filedialog
import os
import sys

def shorten_filename(filename):
    """
    Schneidet den Dateinamen ab: Sucht nach '_clt' und entfernt diesen Teil (inklusive des
    vorangehenden Unterstrichs) aus dem Namen.
    
    Beispiel:
      2025-02-26_S007_Cl_13_clt-dsk-n-1802_20250227-074721.txt 
      -> 2025-02-26_S007_Cl_13
    """
    base = os.path.basename(filename)
    index = base.find("_clt")
    if index != -1:
        return base[:index]
    return base

def read_data(filename):
    """
    Liest die Datei ein und extrahiert die Daten ab der Zeile,
    die mit 'min' beginnt. Es werden zwei Spalten erwartet: Zeit und Signal.
    """
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

def gaussian(x, A, mu, sigma):
    """Definition einer Gaußfunktion."""
    return A * np.exp(-((x - mu)**2) / (2 * sigma**2))

def process_pairs(files_with, files_without, pdf_filename="vanDeemter_Report.pdf"):
    """
    Verarbeitet die Dateipaare:
      - Liest die Messungen mit und ohne Säule ein.
      - Glättet die Signale mittels Savitzky-Golay-Filter.
      - Bestimmt das Peakmaximum, die FWHM und w₁/10.
      - Berechnet PGF und passt eine Gaussian-Kurve (rot) an.
      - Visualisiert die Ergebnisse in einem 3×2-Raster und speichert sie in einem PDF-Report.
    """
    if len(files_with) != len(files_without):
        print("Die Anzahl der Dateien mit Säule und ohne Säule stimmt nicht überein.")
        sys.exit(1)
    
    results = []  # Speicherung der Berechnungsergebnisse für die Tabelle
    const = np.sqrt(8 * np.log(2))  # Umrechnungsfaktor: w₁/₂ = σ * const

    with PdfPages(pdf_filename) as pdf:
        # Für jedes Dateipaar:
        for i, (file_with, file_without) in enumerate(zip(files_with, files_without), start=1):
            try:
                t_with, y_with = read_data(file_with)
                t_without, y_without = read_data(file_without)
            except Exception as e:
                print(f"Fehler beim Lesen der Dateien {file_with} oder {file_without}: {e}")
                continue
            
            # Fensterlängen für den Savitzky-Golay-Filter (muss ungerade sein)
            window_length_with = 11 if len(y_with) >= 11 else (len(y_with) // 2) * 2 + 1
            window_length_without = 11 if len(y_without) >= 11 else (len(y_without) // 2) * 2 + 1
            
            # Glätten der Signale
            y_with_smooth = savgol_filter(y_with, window_length=window_length_with, polyorder=3)
            y_without_smooth = savgol_filter(y_without, window_length=window_length_without, polyorder=3)
            
            # Bestimme das Peakmaximum und die Retentionszeit in beiden Signalen
            peak_idx_with = np.argmax(y_with_smooth)
            peak_idx_without = np.argmax(y_without_smooth)
            retention_time_with = t_with[peak_idx_with]
            retention_time_without = t_without[peak_idx_without]
            
            # Berechne FWHM (rel_height=0.5)
            widths_with, _, _, _ = peak_widths(y_with_smooth, [peak_idx_with], rel_height=0.5)
            widths_without, _, _, _ = peak_widths(y_without_smooth, [peak_idx_without], rel_height=0.5)
            fwhm_with = widths_with[0] * (t_with[1]-t_with[0])
            fwhm_without = widths_without[0] * (t_without[1]-t_without[0])
            
            # Berechne w₁/10 (rel_height=0.1)
            widths_with_10, _, _, _ = peak_widths(y_with_smooth, [peak_idx_with], rel_height=0.1)
            widths_without_10, _, _, _ = peak_widths(y_without_smooth, [peak_idx_without], rel_height=0.1)
            w1_10_with = widths_with_10[0] * (t_with[1]-t_with[0])
            w1_10_without = widths_without_10[0] * (t_without[1]-t_without[0])
            
            # Berechne PGF
            pgf_with = 1.83 * (fwhm_with / w1_10_with) if w1_10_with != 0 else np.nan
            pgf_without = 1.83 * (fwhm_without / w1_10_without) if w1_10_without != 0 else np.nan
            pgf_with_color = "green" if 0.8 <= pgf_with <= 1.15 else "red"
            pgf_without_color = "green" if 0.8 <= pgf_without <= 1.15 else "red"
            
            # Berechne σ-Werte und Säulen-FWHM
            sigma_obs = fwhm_with / const
            sigma_ext = fwhm_without / const
            if sigma_obs**2 >= sigma_ext**2:
                sigma_col = np.sqrt(sigma_obs**2 - sigma_ext**2)
            else:
                print(f"Warnung: Bei {shorten_filename(file_with)} ist σ_obs² kleiner als σ_ext². Setze σ_col auf 0.")
                sigma_col = 0
            fwhm_col = sigma_col * const
            
            # Gaussian-Fit für "mit Säule"
            mask_fit_with = (t_with >= retention_time_with - 2*fwhm_with) & (t_with <= retention_time_with + 2*fwhm_with)
            if np.sum(mask_fit_with) < 3:
                mask_fit_with = np.ones_like(t_with, dtype=bool)
            try:
                popt_with, _ = curve_fit(gaussian, t_with[mask_fit_with], y_with_smooth[mask_fit_with],
                                         p0=[y_with_smooth[peak_idx_with], retention_time_with, fwhm_with/(2*np.sqrt(2*np.log(2)))])
                gauss_fit_with = gaussian(t_with, *popt_with)
                gauss_fit_zoom_with = gaussian(t_with[mask_fit_with], *popt_with)
            except Exception as e:
                print("Gaussian fit failed for mit Säule:", e)
                gauss_fit_with = np.zeros_like(t_with)
                gauss_fit_zoom_with = np.zeros_like(t_with[mask_fit_with])
            
            # Gaussian-Fit für "ohne Säule"
            mask_fit_without = (t_without >= retention_time_without - 2*fwhm_without) & (t_without <= retention_time_without + 2*fwhm_without)
            if np.sum(mask_fit_without) < 3:
                mask_fit_without = np.ones_like(t_without, dtype=bool)
            try:
                popt_without, _ = curve_fit(gaussian, t_without[mask_fit_without], y_without_smooth[mask_fit_without],
                                            p0=[y_without_smooth[peak_idx_without], retention_time_without, fwhm_without/(2*np.sqrt(2*np.log(2)))])
                gauss_fit_without = gaussian(t_without, *popt_without)
                gauss_fit_zoom_without = gaussian(t_without[mask_fit_without], *popt_without)
            except Exception as e:
                print("Gaussian fit failed for ohne Säule:", e)
                gauss_fit_without = np.zeros_like(t_without)
                gauss_fit_zoom_without = np.zeros_like(t_without[mask_fit_without])
            
            # Ergebnisse speichern (nur Dateinamen und FWHM (Säule) für die Abschlusstabelle)
            results.append({
                "Datei mit Säule": shorten_filename(file_with),
                "Datei ohne Säule": shorten_filename(file_without),
                "FWHM (Säule) [min]": fwhm_col
            })
            
            # Zoom-Bereiche (wie bisher)
            zoom_margin_with = fwhm_with
            zoom_left_with = max(retention_time_with - zoom_margin_with, t_with[0])
            zoom_right_with = min(retention_time_with + zoom_margin_with, t_with[-1])
            mask_zoom_with = (t_with >= zoom_left_with) & (t_with <= zoom_right_with)
            
            zoom_margin_without = fwhm_without
            zoom_left_without = max(retention_time_without - zoom_margin_without, t_without[0])
            zoom_right_without = min(retention_time_without + zoom_margin_without, t_without[-1])
            mask_zoom_without = (t_without >= zoom_left_without) & (t_without <= zoom_right_without)
            
            # Erzeuge 6 Bereiche (3 Zeilen, 2 Spalten)
            fig, axs = plt.subplots(3, 2, figsize=(10, 12))
            fig.suptitle(f"Messung {i}: {shorten_filename(file_with)} vs. {shorten_filename(file_without)}", fontsize=14)
            
            # --- (0,0): Mit Säule - Komplettes Chromatogramm ---
            axs[0, 0].plot(t_with, y_with, 'ko', markersize=3, label="Messdaten")
            axs[0, 0].plot(t_with, y_with_smooth, 'b-', linewidth=1, label="Geglättet")
            axs[0, 0].plot(t_with, gauss_fit_with, 'r-', linewidth=1, label="Gaussian Fit")
            axs[0, 0].axvline(x=retention_time_with - fwhm_with/2, color='r', linestyle='--', label="FWHM-Grenzen")
            axs[0, 0].axvline(x=retention_time_with + fwhm_with/2, color='r', linestyle='--')
            axs[0, 0].set_xlabel("Zeit [min]")
            axs[0, 0].set_ylabel("Signal")
            axs[0, 0].legend(fontsize=8)
            axs[0, 0].set_title("Mit Säule - Komplett")
            
            # --- (0,1): Mit Säule - Zoom auf den Peak ---
            axs[0, 1].plot(t_with[mask_zoom_with], y_with[mask_zoom_with], 'ko', markersize=3, label="Messdaten (Zoom)")
            axs[0, 1].plot(t_with[mask_zoom_with], y_with_smooth[mask_zoom_with], 'b-', linewidth=1, label="Geglättet (Zoom)")
            axs[0, 1].plot(t_with[mask_zoom_with], gauss_fit_with[mask_zoom_with], 'r-', linewidth=1, label="Gaussian Fit")
            axs[0, 1].axvline(x=retention_time_with - fwhm_with/2, color='r', linestyle='--', label="FWHM-Grenzen")
            axs[0, 1].axvline(x=retention_time_with + fwhm_with/2, color='r', linestyle='--')
            axs[0, 1].set_xlabel("Zeit [min]")
            axs[0, 1].set_ylabel("Signal")
            axs[0, 1].legend(fontsize=8)
            axs[0, 1].set_title("Mit Säule - Zoom")
            
            # --- (1,0): Ohne Säule - Komplettes Chromatogramm ---
            axs[1, 0].plot(t_without, y_without, 'ko', markersize=3, label="Messdaten")
            axs[1, 0].plot(t_without, y_without_smooth, 'b-', linewidth=1, label="Geglättet")
            axs[1, 0].plot(t_without, gauss_fit_without, 'r-', linewidth=1, label="Gaussian Fit")
            axs[1, 0].axvline(x=retention_time_without - fwhm_without/2, color='r', linestyle='--', label="FWHM-Grenzen")
            axs[1, 0].axvline(x=retention_time_without + fwhm_without/2, color='r', linestyle='--')
            axs[1, 0].set_xlabel("Zeit [min]")
            axs[1, 0].set_ylabel("Signal")
            axs[1, 0].legend(fontsize=8)
            axs[1, 0].set_title("Ohne Säule - Komplett")
            
            # --- (1,1): Ohne Säule - Zoom auf den Peak ---
            axs[1, 1].plot(t_without[mask_zoom_without], y_without[mask_zoom_without], 'ko', markersize=3, label="Messdaten (Zoom)")
            axs[1, 1].plot(t_without[mask_zoom_without], y_without_smooth[mask_zoom_without], 'b-', linewidth=1, label="Geglättet (Zoom)")
            axs[1, 1].plot(t_without[mask_zoom_without], gauss_fit_without[mask_zoom_without], 'r-', linewidth=1, label="Gaussian Fit")
            axs[1, 1].axvline(x=retention_time_without - fwhm_without/2, color='r', linestyle='--', label="FWHM-Grenzen")
            axs[1, 1].axvline(x=retention_time_without + fwhm_without/2, color='r', linestyle='--')
            axs[1, 1].set_xlabel("Zeit [min]")
            axs[1, 1].set_ylabel("Signal")
            axs[1, 1].legend(fontsize=8)
            axs[1, 1].set_title("Ohne Säule - Zoom")
            
            # --- Zeile 3: Ergebnisbox ---
            axs[2, 0].axis('off')
            axs[2, 1].axis('off')
            # Basisinformationen
            textstr = (
                f"FWHM (mit Säule): {fwhm_with:.4f} min\n"
                f"FWHM (ohne Säule): {fwhm_without:.4f} min\n"
                f"FWHM (Säule): {fwhm_col:.4f} min"
            )
            axs[2, 0].text(0.05, 0.6, textstr,
                           transform=axs[2, 0].transAxes,
                           fontsize=10, va="center",
                           bbox=dict(facecolor='white', alpha=0.5))
            # PGF-Werte
            axs[2, 0].text(0.6, 0.6, f"PGF (mit Säule): {pgf_with:.2f}", 
                           transform=axs[2, 0].transAxes,
                           fontsize=10, va="center", color=pgf_with_color)
            axs[2, 0].text(0.6, 0.4, f"PGF (ohne Säule): {pgf_without:.2f}", 
                           transform=axs[2, 0].transAxes,
                           fontsize=10, va="center", color=pgf_without_color)
            axs[2, 0].text(0.6, 0.2, "Sollte zwischen 0.8 und 1.15 liegen", 
                           transform=axs[2, 0].transAxes,
                           fontsize=10, va="center", color='black')
            
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
            print(f"Messung {i} ausgewertet: {shorten_filename(file_with)} und {shorten_filename(file_without)}")
        
        # Letzte Seite: Tabelle mit den Dateinamen und FWHM (Säule)
        fig, ax = plt.subplots(figsize=(12, len(results)*0.5 + 2))
        ax.axis('tight')
        ax.axis('off')
        
        table_data = [["Datei mit Säule", "Datei ohne Säule", "FWHM (Säule) [min]"]]
        for res in results:
            table_data.append([
                res["Datei mit Säule"],
                res["Datei ohne Säule"],
                f"{res['FWHM (Säule) [min]']:.4f}"
            ])
        table = ax.table(cellText=table_data, loc="center", cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.5)
        ax.set_title("Übersicht der FWHM (Säule)", fontweight="bold")
        pdf.savefig(fig)
        plt.close(fig)
    
    print(f"\nReport wurde erstellt: {pdf_filename}")

if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()
    
    files_with = filedialog.askopenfilenames(
        title="Wähle die Dateien aus (Messungen mit Säule)",
        filetypes=[("Textdateien", "*.txt"), ("Alle Dateien", "*.*")]
    )
    files_with = list(files_with)
    if not files_with:
        print("Keine Dateien für Messungen mit Säule ausgewählt.")
        sys.exit(1)
    
    files_without = filedialog.askopenfilenames(
        title="Wähle die Dateien aus (Messungen ohne Säule) in derselben Reihenfolge",
        filetypes=[("Textdateien", "*.txt"), ("Alle Dateien", "*.*")]
    )
    files_without = list(files_without)
    if not files_without:
        print("Keine Dateien für Messungen ohne Säule ausgewählt.")
        sys.exit(1)
    
    process_pairs(files_with, files_without)
