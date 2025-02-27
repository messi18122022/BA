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
  
  Zusätzlich wird eine Gaussian-Kurve (rot) gefittet – mit Baselineabschätzung anhand der Messdaten.
  Für die PGF-Berechnung werden nun FWHM und w₁/10 direkt analog berechnet:
  
      PGF = 1.83 · (FWHM / w₁/10)
      
  Ein idealer Gauß-Peak sollte einen PGF im Bereich von ca. 0.8 bis 1.15 ergeben.
  
Pro Dateipaar wird eine Seite im PDF-Report erstellt, die ein 3×2-Raster enthält:
  - 1. Zeile:   Mit Säule (links: Komplett, rechts: Zoom)
  - 2. Zeile:   Ohne Säule (links: Komplett, rechts: Zoom)
  - 3. Zeile:   Enthält eine Textbox mit den Ergebnissen (als ein Block)
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
    base = os.path.basename(filename)
    index = base.find("_clt")
    if index != -1:
        return base[:index]
    return base

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

def gaussian_offset(x, A, mu, sigma, B):
    return B + A * np.exp(-((x - mu)**2) / (2 * sigma**2))

def process_pairs(files_with, files_without, pdf_filename="vanDeemter_Report.pdf"):
    if len(files_with) != len(files_without):
        print("Die Anzahl der Dateien mit Säule und ohne Säule stimmt nicht überein.")
        sys.exit(1)
    
    results = []
    const = np.sqrt(8 * np.log(2))  # w₁/₂ = σ * const

    with PdfPages(pdf_filename) as pdf:
        for i, (file_with, file_without) in enumerate(zip(files_with, files_without), start=1):
            try:
                t_with, y_with = read_data(file_with)
                t_without, y_without = read_data(file_without)
            except Exception as e:
                print(f"Fehler beim Lesen der Dateien {file_with} oder {file_without}: {e}")
                continue

            # Glätten der Signale
            window_length_with = 11 if len(y_with) >= 11 else (len(y_with) // 2) * 2 + 1
            window_length_without = 11 if len(y_without) >= 11 else (len(y_without) // 2) * 2 + 1
            y_with_smooth = savgol_filter(y_with, window_length=window_length_with, polyorder=3)
            y_without_smooth = savgol_filter(y_without, window_length=window_length_without, polyorder=3)

            # Peak und Retentionszeit
            peak_idx_with = np.argmax(y_with_smooth)
            peak_idx_without = np.argmax(y_without_smooth)
            retention_time_with = t_with[peak_idx_with]
            retention_time_without = t_without[peak_idx_without]

            # FWHM (rel_height=0.5)
            widths_with, _, _, _ = peak_widths(y_with_smooth, [peak_idx_with], rel_height=0.5)
            widths_without, _, _, _ = peak_widths(y_without_smooth, [peak_idx_without], rel_height=0.5)
            fwhm_with = widths_with[0] * (t_with[1]-t_with[0])
            fwhm_without = widths_without[0] * (t_without[1]-t_without[0])

            # w₁/10 analog zu FWHM, aber mit rel_height=0.9
            widths_with_10, _, _, _ = peak_widths(y_with_smooth, [peak_idx_with], rel_height=0.9)
            widths_without_10, _, _, _ = peak_widths(y_without_smooth, [peak_idx_without], rel_height=0.9)
            w1_10_with = widths_with_10[0] * (t_with[1]-t_with[0])
            w1_10_without = widths_without_10[0] * (t_without[1]-t_without[0])

            # PGF-Berechnung
            pgf_with = 1.83 * (fwhm_with / w1_10_with) if w1_10_with != 0 else np.nan
            pgf_without = 1.83 * (fwhm_without / w1_10_without) if w1_10_without != 0 else np.nan
            pgf_with_color = "green" if 0.8 <= pgf_with <= 1.15 else "red"
            pgf_without_color = "green" if 0.8 <= pgf_without <= 1.15 else "red"

            # σ-Werte und Säulen-FWHM
            sigma_obs = fwhm_with / const
            sigma_ext = fwhm_without / const
            if sigma_obs**2 >= sigma_ext**2:
                sigma_col = np.sqrt(sigma_obs**2 - sigma_ext**2)
            else:
                print(f"Warnung: Bei {shorten_filename(file_with)} ist σ_obs² kleiner als σ_ext². Setze σ_col auf 0.")
                sigma_col = 0
            fwhm_col = sigma_col * const

            # Gaussian-Fit mit Offset – enges Fit-Fenster (retention_time ± 1·FWHM)
            fit_window_factor = 1.0
            mask_fit_with = (t_with >= retention_time_with - fit_window_factor * fwhm_with) & (t_with <= retention_time_with + fit_window_factor * fwhm_with)
            mask_fit_without = (t_without >= retention_time_without - fit_window_factor * fwhm_without) & (t_without <= retention_time_without + fit_window_factor * fwhm_without)
            def estimate_baseline(x, y, mask):
                indices = np.where(mask)[0]
                if len(indices) < 2:
                    return np.min(y)
                edge_count = max(1, int(0.1 * len(indices)))
                return np.mean(np.concatenate((y[indices[:edge_count]], y[indices[-edge_count:]])))
            B0_with = estimate_baseline(t_with, y_with_smooth, mask_fit_with)
            B0_without = estimate_baseline(t_without, y_without_smooth, mask_fit_without)
            p0_with = [y_with_smooth[peak_idx_with] - B0_with, retention_time_with, fwhm_with/(2*np.sqrt(2*np.log(2))), B0_with]
            p0_without = [y_without_smooth[peak_idx_without] - B0_without, retention_time_without, fwhm_without/(2*np.sqrt(2*np.log(2))), B0_without]
            try:
                popt_with, _ = curve_fit(gaussian_offset, t_with[mask_fit_with], y_with_smooth[mask_fit_with], p0=p0_with)
                gauss_fit_with = gaussian_offset(t_with, *popt_with)
            except Exception as e:
                print("Gaussian fit failed for mit Säule:", e)
                gauss_fit_with = np.zeros_like(t_with)
            try:
                popt_without, _ = curve_fit(gaussian_offset, t_without[mask_fit_without], y_without_smooth[mask_fit_without], p0=p0_without)
                gauss_fit_without = gaussian_offset(t_without, *popt_without)
            except Exception as e:
                print("Gaussian fit failed for ohne Säule:", e)
                gauss_fit_without = np.zeros_like(t_without)

            results.append({
                "Datei mit Säule": shorten_filename(file_with),
                "Datei ohne Säule": shorten_filename(file_without),
                "FWHM (Säule) [min]": fwhm_col
            })

            # Zoom-Bereiche: Zoom-Faktor 1.5 * FWHM
            zoom_margin_with = 1.5 * fwhm_with
            zoom_left_with = max(retention_time_with - zoom_margin_with, t_with[0])
            zoom_right_with = min(retention_time_with + zoom_margin_with, t_with[-1])
            mask_zoom_with = (t_with >= zoom_left_with) & (t_with <= zoom_right_with)
            zoom_margin_without = 1.5 * fwhm_without
            zoom_left_without = max(retention_time_without - zoom_margin_without, t_without[0])
            zoom_right_without = min(retention_time_without + zoom_margin_without, t_without[-1])
            mask_zoom_without = (t_without >= zoom_left_without) & (t_without <= zoom_right_without)

            # Erzeuge 3×2-Raster: erste zwei Zeilen für Plots, dritte Zeile für Text
            fig, axs = plt.subplots(3, 2, figsize=(10, 12))
            fig.suptitle(f"Messung {i}: {shorten_filename(file_with)} vs. {shorten_filename(file_without)}", fontsize=14)
            
            # Mit Säule – Komplett
            axs[0, 0].plot(t_with, y_with, 'ko', markersize=3, label="Messdaten")
            axs[0, 0].plot(t_with, y_with_smooth, 'b-', linewidth=1, label="Geglättet")
            axs[0, 0].plot(t_with, gauss_fit_with, 'r-', linewidth=1, label="Gaussian Fit")
            axs[0, 0].axvline(x=retention_time_with - fwhm_with/2, color='gray', linestyle='--', linewidth=1.0, alpha=0.7, label="FWHM-Grenzen")
            axs[0, 0].axvline(x=retention_time_with + fwhm_with/2, color='gray', linestyle='--', linewidth=1.0, alpha=0.7)
            axs[0, 0].axvline(x=retention_time_with - w1_10_with/2, color='blue', linestyle=':', linewidth=1.0, alpha=0.7, label="w₁/10-Grenzen")
            axs[0, 0].axvline(x=retention_time_with + w1_10_with/2, color='blue', linestyle=':', linewidth=1.0, alpha=0.7)
            axs[0, 0].set_xlabel("Zeit [min]")
            axs[0, 0].set_ylabel("Signal")
            axs[0, 0].legend(fontsize=8)
            axs[0, 0].set_title("Mit Säule - Komplett")
            
            # Mit Säule – Zoom
            axs[0, 1].plot(t_with[mask_zoom_with], y_with[mask_zoom_with], 'ko', markersize=3, label="Messdaten (Zoom)")
            axs[0, 1].plot(t_with[mask_zoom_with], y_with_smooth[mask_zoom_with], 'b-', linewidth=1, label="Geglättet (Zoom)")
            axs[0, 1].plot(t_with[mask_zoom_with], gauss_fit_with[mask_zoom_with], 'r-', linewidth=1, label="Gaussian Fit")
            axs[0, 1].axvline(x=retention_time_with - fwhm_with/2, color='gray', linestyle='--', linewidth=1.0, alpha=0.7, label="FWHM-Grenzen")
            axs[0, 1].axvline(x=retention_time_with + fwhm_with/2, color='gray', linestyle='--', linewidth=1.0, alpha=0.7)
            axs[0, 1].axvline(x=retention_time_with - w1_10_with/2, color='blue', linestyle=':', linewidth=1.0, alpha=0.7, label="w₁/10-Grenzen")
            axs[0, 1].axvline(x=retention_time_with + w1_10_with/2, color='blue', linestyle=':', linewidth=1.0, alpha=0.7)
            axs[0, 1].set_xlabel("Zeit [min]")
            axs[0, 1].set_ylabel("Signal")
            axs[0, 1].legend(fontsize=8)
            axs[0, 1].set_title("Mit Säule - Zoom")
            
            # Ohne Säule – Komplett
            axs[1, 0].plot(t_without, y_without, 'ko', markersize=3, label="Messdaten")
            axs[1, 0].plot(t_without, y_without_smooth, 'b-', linewidth=1, label="Geglättet")
            axs[1, 0].plot(t_without, gauss_fit_without, 'r-', linewidth=1, label="Gaussian Fit")
            axs[1, 0].axvline(x=retention_time_without - fwhm_without/2, color='gray', linestyle='--', linewidth=1.0, alpha=0.7, label="FWHM-Grenzen")
            axs[1, 0].axvline(x=retention_time_without + fwhm_without/2, color='gray', linestyle='--', linewidth=1.0, alpha=0.7)
            axs[1, 0].axvline(x=retention_time_without - w1_10_without/2, color='blue', linestyle=':', linewidth=1.0, alpha=0.7, label="w₁/10-Grenzen")
            axs[1, 0].axvline(x=retention_time_without + w1_10_without/2, color='blue', linestyle=':', linewidth=1.0, alpha=0.7)
            axs[1, 0].set_xlabel("Zeit [min]")
            axs[1, 0].set_ylabel("Signal")
            axs[1, 0].legend(fontsize=8)
            axs[1, 0].set_title("Ohne Säule - Komplett")
            
            # Ohne Säule – Zoom
            axs[1, 1].plot(t_without[mask_zoom_without], y_without[mask_zoom_without], 'ko', markersize=3, label="Messdaten (Zoom)")
            axs[1, 1].plot(t_without[mask_zoom_without], y_without_smooth[mask_zoom_without], 'b-', linewidth=1, label="Geglättet (Zoom)")
            axs[1, 1].plot(t_without[mask_zoom_without], gauss_fit_without[mask_zoom_without], 'r-', linewidth=1, label="Gaussian Fit")
            axs[1, 1].axvline(x=retention_time_without - fwhm_without/2, color='gray', linestyle='--', linewidth=1.0, alpha=0.7, label="FWHM-Grenzen")
            axs[1, 1].axvline(x=retention_time_without + fwhm_without/2, color='gray', linestyle='--', linewidth=1.0, alpha=0.7)
            axs[1, 1].axvline(x=retention_time_without - w1_10_without/2, color='blue', linestyle=':', linewidth=1.0, alpha=0.7, label="w₁/10-Grenzen")
            axs[1, 1].axvline(x=retention_time_without + w1_10_without/2, color='blue', linestyle=':', linewidth=1.0, alpha=0.7)
            axs[1, 1].set_xlabel("Zeit [min]")
            axs[1, 1].set_ylabel("Signal")
            axs[1, 1].legend(fontsize=8)
            axs[1, 1].set_title("Ohne Säule - Zoom")
            
            # Ergebnisbox in der 3. Zeile – ein einziger Textblock
            axs[2, 0].axis('off')
            axs[2, 1].axis('off')
            result_text = (
                "Ergebnisse:\n"
                "Mit Säule: PGF = " + f"{pgf_with:.2f}" + "\n"
                "Ohne Säule: PGF = " + f"{pgf_without:.2f}" + "\n\n"
                "FWHM (Säule): " + f"{fwhm_col:.4f} min\n"
                "Akzeptanzkriterium: 0.8 < PGF < 1.15"
            )
            # Den Text in einem einzigen Textfeld, wobei die Farbe der PGF-Werte entsprechend gesetzt wird
            axs[2, 0].text(0.05, 0.5, result_text, transform=axs[2, 0].transAxes,
                           fontsize=10, va="center", bbox=dict(facecolor='white', alpha=0.5))
            
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
            print(f"Messung {i} ausgewertet: {shorten_filename(file_with)} und {shorten_filename(file_without)}")
        
        # Abschlusstabelle: nur Dateinamen und FWHM (Säule)
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
