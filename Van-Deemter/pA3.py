#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dieses Skript ermöglicht das Einlesen von Chromatogramm-Daten für van Deemter Analysen.
Zuerst werden die Messungen mit der Säule ausgewählt, anschließend die Messungen ohne Säule.
Die Reihenfolge in beiden Dialogfenstern muss übereinstimmen, damit die extrakolumnären Einflüsse
korrekt abgezogen werden können.

Berechnungsgrundlagen:
  • Für jedes Chromatogramm wird die Halbwertspeakbreite (FWHM) ermittelt.
  • Die Umrechnung in sigma erfolgt über: w₁/₂ = σ · √(8·ln2)
  • Es gilt: σ_obs² = σ_col² + σ_ext²,
    wobei σ_obs aus der Messung mit Säule und σ_ext aus der Messung ohne Säule berechnet wird.
  • Daraus folgt: σ_col = √(σ_obs² - σ_ext²) und w₁/₂ (Säule) = σ_col · √(8·ln2)
  
  Zusätzlich wird eine Gaussian-Kurve (rot) gefittet – mit Baselineabschätzung anhand der Messdaten.
  Für die PGF-Berechnung werden nun FWHM und w₁/10 direkt analog berechnet:
  
      PGF = 1.83 · (FWHM / w₁/10)
      
  Ein idealer Gauß-Peak sollte einen PGF im Bereich von ca. 0.8 bis 1.15 ergeben.

Zusätzlich wird ein van Deemter-Plot erstellt. Dazu wird für die Messung mit Säule:
  - N = 5.55 · (t_R)² / (w₁/₂)²
  - H = L / N
  (Säulenlänge L ist einstellbar, Standard: 100 mm)
  
Pro Dateipaar wird eine Seite im PDF-Report erstellt, die ein 3×2-Raster enthält:
  - 1. Zeile:   Mit Säule (links: Komplett, rechts: Zoom)
  - 2. Zeile:   Ohne Säule (links: Komplett, rechts: Zoom)
  - 3. Zeile:   Enthält eine Textbox mit den Ergebnissen inkl. der verwendeten Zahlen
Auf der ersten Seite wird der van Deemter-Plot (H vs. Flussrate) abgelegt,
auf den folgenden Seiten die Einzelplots pro Messung,
und zum Schluss eine Tabelle mit den FWHM(Säule)-Werten.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.signal import peak_widths, savgol_filter
from scipy.optimize import curve_fit
import tkinter as tk
from tkinter import filedialog, simpledialog
import os
import sys

# Standardlänge der Säule in mm
L = 100.0

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

def process_pairs(files_with, files_without, flow_min, flow_max, delta, pdf_filename="vanDeemter_Report.pdf"):
    # 1) Prüfe Anzahl
    if len(files_with) != len(files_without):
        print("Die Anzahl der Dateien mit Säule und ohne Säule stimmt nicht überein.")
        sys.exit(1)

    # 2) Bestimme Anzahl der Flussratenpunkte
    n_points = int(round((flow_max - flow_min) / delta)) + 1
    if n_points != len(files_with):
        print(f"Die Anzahl der Messpunkte ({n_points}) entspricht nicht der Anzahl der Dateien ({len(files_with)}).")
        sys.exit(1)

    # Erzeuge die Flussraten-Liste
    flow_rates = np.linspace(flow_min, flow_max, n_points)

    # Listen für van Deemter
    vanDeemter_H = []
    measurement_figs = []
    results = []
    const = np.sqrt(8 * np.log(2))  # w₁/₂ = σ * const

    # 3) Schleife über die Dateipaare
    for j, (file_with, file_without) in enumerate(zip(files_with, files_without)):
        try:
            t_with, y_with = read_data(file_with)
            t_without, y_without = read_data(file_without)
        except Exception as e:
            print(f"Fehler beim Lesen der Dateien {file_with} oder {file_without}: {e}")
            continue

        # Signale glätten
        window_length_with = 11 if len(y_with) >= 11 else (len(y_with) // 2) * 2 + 1
        window_length_without = 11 if len(y_without) >= 11 else (len(y_without) // 2) * 2 + 1
        y_with_smooth = savgol_filter(y_with, window_length=window_length_with, polyorder=3)
        y_without_smooth = savgol_filter(y_without, window_length=window_length_without, polyorder=3)

        # Peak mit Säule
        peak_idx_with = np.argmax(y_with_smooth)
        retention_time_with = t_with[peak_idx_with]

        # FWHM (mit Säule)
        widths_with, _, _, _ = peak_widths(y_with_smooth, [peak_idx_with], rel_height=0.5)
        fwhm_with = widths_with[0] * (t_with[1]-t_with[0])

        # w₁/10 (mit Säule)
        widths_with_10, _, _, _ = peak_widths(y_with_smooth, [peak_idx_with], rel_height=0.9)
        w1_10_with = widths_with_10[0] * (t_with[1]-t_with[0])

        # Peak ohne Säule
        peak_idx_without = np.argmax(y_without_smooth)
        retention_time_without = t_without[peak_idx_without]

        # FWHM (ohne Säule)
        widths_without, _, _, _ = peak_widths(y_without_smooth, [peak_idx_without], rel_height=0.5)
        fwhm_without = widths_without[0] * (t_without[1]-t_without[0])

        # w₁/10 (ohne Säule)
        widths_without_10, _, _, _ = peak_widths(y_without_smooth, [peak_idx_without], rel_height=0.9)
        w1_10_without = widths_without_10[0] * (t_without[1]-t_without[0])

        # FWHM-Korrektur
        sigma_obs = fwhm_with / const
        sigma_ext = fwhm_without / const
        sigma_col = np.sqrt(sigma_obs**2 - sigma_ext**2) if sigma_obs**2 >= sigma_ext**2 else 0
        fwhm_col = sigma_col * const

        # PGF
        pgf_with = 1.83 * (fwhm_with / w1_10_with) if w1_10_with != 0 else np.nan
        pgf_without = 1.83 * (fwhm_without / w1_10_without) if w1_10_without != 0 else np.nan

        # van Deemter: N, H
        # N = 5.55 * (t_R)^2 / (fwhm_col)^2
        # H = L / N
        if fwhm_col == 0:
            N = np.nan
        else:
            N = 5.55 * (retention_time_with**2) / (fwhm_col**2)
        H = L / N if N != 0 else np.nan
        vanDeemter_H.append(H)

        results.append({
            "Datei mit Säule": shorten_filename(file_with),
            "Datei ohne Säule": shorten_filename(file_without),
            "FWHM (Säule) [min]": fwhm_col
        })

        # Zoom-Bereiche
        zoom_margin_with = 1.5 * fwhm_with
        zoom_left_with = max(retention_time_with - zoom_margin_with, t_with[0])
        zoom_right_with = min(retention_time_with + zoom_margin_with, t_with[-1])
        mask_zoom_with = (t_with >= zoom_left_with) & (t_with <= zoom_right_with)

        zoom_margin_without = 1.5 * fwhm_without
        zoom_left_without = max(retention_time_without - zoom_margin_without, t_without[0])
        zoom_right_without = min(retention_time_without + zoom_margin_without, t_without[-1])
        mask_zoom_without = (t_without >= zoom_left_without) & (t_without <= zoom_right_without)

        # Erzeuge die Plot-Seite
        fig, axs = plt.subplots(3, 2, figsize=(10, 12))
        fig.suptitle(f"Messung {j+1}: {shorten_filename(file_with)} vs. {shorten_filename(file_without)}", fontsize=14)

        # Gaussian-Fit (mit Säule)
        try:
            fit_window_factor = 1.0
            mask_fit_with = (t_with >= retention_time_with - fit_window_factor*fwhm_with) & (t_with <= retention_time_with + fit_window_factor*fwhm_with)
            def estimate_baseline(x, y, mask):
                idx = np.where(mask)[0]
                if len(idx) < 2:
                    return np.min(y)
                edge_count = max(1, int(0.1*len(idx)))
                return np.mean(np.concatenate((y[idx[:edge_count]], y[idx[-edge_count:]])))
            B0_with = estimate_baseline(t_with, y_with_smooth, mask_fit_with)
            p0_with = [y_with_smooth[peak_idx_with] - B0_with, retention_time_with, fwhm_with/(2*np.sqrt(2*np.log(2))), B0_with]
            popt_with, _ = curve_fit(gaussian_offset, t_with[mask_fit_with], y_with_smooth[mask_fit_with], p0=p0_with)
            gauss_fit_with = gaussian_offset(t_with, *popt_with)
        except:
            gauss_fit_with = np.zeros_like(t_with)

        # Gaussian-Fit (ohne Säule)
        try:
            mask_fit_without = (t_without >= retention_time_without - fit_window_factor*fwhm_without) & (t_without <= retention_time_without + fit_window_factor*fwhm_without)
            B0_without = estimate_baseline(t_without, y_without_smooth, mask_fit_without)
            p0_without = [y_without_smooth[peak_idx_without] - B0_without, retention_time_without, fwhm_without/(2*np.sqrt(2*np.log(2))), B0_without]
            popt_without, _ = curve_fit(gaussian_offset, t_without[mask_fit_without], y_without_smooth[mask_fit_without], p0=p0_without)
            gauss_fit_without = gaussian_offset(t_without, *popt_without)
        except:
            gauss_fit_without = np.zeros_like(t_without)

        # (0,0): Mit Säule - Komplett
        axs[0, 0].plot(t_with, y_with, 'ko', markersize=3, label="Messdaten")
        axs[0, 0].plot(t_with, y_with_smooth, 'b-', linewidth=1, label="Geglättet")
        axs[0, 0].plot(t_with, gauss_fit_with, 'r-', linewidth=1, label="Gaussian Fit")
        axs[0, 0].axvline(retention_time_with - fwhm_with/2, color='gray', linestyle='--', linewidth=1, alpha=0.7)
        axs[0, 0].axvline(retention_time_with + fwhm_with/2, color='gray', linestyle='--', linewidth=1, alpha=0.7)
        axs[0, 0].axvline(retention_time_with - w1_10_with/2, color='blue', linestyle=':', linewidth=1, alpha=0.7)
        axs[0, 0].axvline(retention_time_with + w1_10_with/2, color='blue', linestyle=':', linewidth=1, alpha=0.7)
        axs[0, 0].set_xlabel("Zeit [min]")
        axs[0, 0].set_ylabel("Signal")
        axs[0, 0].legend(fontsize=8)
        axs[0, 0].set_title("Mit Säule - Komplett")

        # (0,1): Mit Säule - Zoom
        axs[0, 1].plot(t_with[mask_zoom_with], y_with[mask_zoom_with], 'ko', markersize=3, label="Messdaten (Zoom)")
        axs[0, 1].plot(t_with[mask_zoom_with], y_with_smooth[mask_zoom_with], 'b-', linewidth=1, label="Geglättet (Zoom)")
        axs[0, 1].plot(t_with[mask_zoom_with], gauss_fit_with[mask_zoom_with], 'r-', linewidth=1, label="Gaussian Fit")
        axs[0, 1].axvline(retention_time_with - fwhm_with/2, color='gray', linestyle='--', linewidth=1, alpha=0.7)
        axs[0, 1].axvline(retention_time_with + fwhm_with/2, color='gray', linestyle='--', linewidth=1, alpha=0.7)
        axs[0, 1].axvline(retention_time_with - w1_10_with/2, color='blue', linestyle=':', linewidth=1, alpha=0.7)
        axs[0, 1].axvline(retention_time_with + w1_10_with/2, color='blue', linestyle=':', linewidth=1, alpha=0.7)
        axs[0, 1].set_xlabel("Zeit [min]")
        axs[0, 1].set_ylabel("Signal")
        axs[0, 1].legend(fontsize=8)
        axs[0, 1].set_title("Mit Säule - Zoom")

        # (1,0): Ohne Säule - Komplett
        axs[1, 0].plot(t_without, y_without, 'ko', markersize=3, label="Messdaten")
        axs[1, 0].plot(t_without, y_without_smooth, 'b-', linewidth=1, label="Geglättet")
        axs[1, 0].plot(t_without, gauss_fit_without, 'r-', linewidth=1, label="Gaussian Fit")
        axs[1, 0].axvline(retention_time_without - fwhm_without/2, color='gray', linestyle='--', linewidth=1, alpha=0.7)
        axs[1, 0].axvline(retention_time_without + fwhm_without/2, color='gray', linestyle='--', linewidth=1, alpha=0.7)
        axs[1, 0].axvline(retention_time_without - w1_10_without/2, color='blue', linestyle=':', linewidth=1, alpha=0.7)
        axs[1, 0].axvline(retention_time_without + w1_10_without/2, color='blue', linestyle=':', linewidth=1, alpha=0.7)
        axs[1, 0].set_xlabel("Zeit [min]")
        axs[1, 0].set_ylabel("Signal")
        axs[1, 0].legend(fontsize=8)
        axs[1, 0].set_title("Ohne Säule - Komplett")

        # (1,1): Ohne Säule - Zoom
        axs[1, 1].plot(t_without[mask_zoom_without], y_without[mask_zoom_without], 'ko', markersize=3, label="Messdaten (Zoom)")
        axs[1, 1].plot(t_without[mask_zoom_without], y_without_smooth[mask_zoom_without], 'b-', linewidth=1, label="Geglättet (Zoom)")
        axs[1, 1].plot(t_without[mask_zoom_without], gauss_fit_without[mask_zoom_without], 'r-', linewidth=1, label="Gaussian Fit")
        axs[1, 1].axvline(retention_time_without - fwhm_without/2, color='gray', linestyle='--', linewidth=1, alpha=0.7)
        axs[1, 1].axvline(retention_time_without + fwhm_without/2, color='gray', linestyle='--', linewidth=1, alpha=0.7)
        axs[1, 1].axvline(retention_time_without - w1_10_without/2, color='blue', linestyle=':', linewidth=1, alpha=0.7)
        axs[1, 1].axvline(retention_time_without + w1_10_without/2, color='blue', linestyle=':', linewidth=1, alpha=0.7)
        axs[1, 1].set_xlabel("Zeit [min]")
        axs[1, 1].set_ylabel("Signal")
        axs[1, 1].legend(fontsize=8)
        axs[1, 1].set_title("Ohne Säule - Zoom")

        # (2,0) und (2,1): Textbox
        axs[2, 0].axis('off')
        axs[2, 1].axis('off')
        result_text = (
            "Ergebnisse:\n"
            "Mit Säule:\n"
            f"  FWHM: {fwhm_with:.4f} min\n"
            f"  w₁/10: {w1_10_with:.4f} min\n"
            f"  PGF = 1.83 * ({fwhm_with:.4f} / {w1_10_with:.4f}) = {pgf_with:.2f}\n\n"
            "Ohne Säule:\n"
            f"  FWHM: {fwhm_without:.4f} min\n"
            f"  w₁/10: {w1_10_without:.4f} min\n"
            f"  PGF = 1.83 * ({fwhm_without:.4f} / {w1_10_without:.4f}) = {pgf_without:.2f}\n\n"
            f"FWHM (Säule): {fwhm_col:.4f} min\n"
            "Akzeptanzkriterium: 0.8 < PGF < 1.15"
        )
        axs[2, 0].text(0.05, 0.5, result_text, transform=axs[2, 0].transAxes,
                       fontsize=10, va="center", bbox=dict(facecolor='white', alpha=0.5))
        
        fig.tight_layout()
        measurement_figs.append(fig)
        plt.close(fig)

    # Erzeuge den van Deemter Plot (H vs. Flussrate)
    fig_vd, ax_vd = plt.subplots(figsize=(8,6))
    # Die Werte in vanDeemter_H haben wir in der Schleife oben bestimmt
    ax_vd.plot(flow_rates, vanDeemter_H, 'bo-', label="H (mm)")
    ax_vd.set_xlabel("Flussrate (mL/min)")
    ax_vd.set_ylabel("H (mm)")
    ax_vd.set_title("van Deemter Plot")
    ax_vd.legend()

    # Alles in PDF speichern
    with PdfPages(pdf_filename) as pdf:
        # 1. Seite: van Deemter Plot
        pdf.savefig(fig_vd)
        plt.close(fig_vd)
        # Folgeseiten: Messungsseiten
        for fig in measurement_figs:
            pdf.savefig(fig)
        # Letzte Seite: Tabelle
        fig_table, ax_table = plt.subplots(figsize=(12, len(results)*0.5 + 2))
        ax_table.axis('tight')
        ax_table.axis('off')
        table_data = [["Datei mit Säule", "Datei ohne Säule", "FWHM (Säule) [min]"]]
        for res in results:
            table_data.append([
                res["Datei mit Säule"],
                res["Datei ohne Säule"],
                f"{res['FWHM (Säule) [min]']:.4f}"
            ])
        table = ax_table.table(cellText=table_data, loc="center", cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.5)
        ax_table.set_title("Übersicht der FWHM (Säule)", fontweight="bold")
        pdf.savefig(fig_table)
        plt.close(fig_table)
    
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
    
    flow_min = float(simpledialog.askstring("Flussrate", "Minimale Flussrate (mL/min):", initialvalue="0.1"))
    flow_max = float(simpledialog.askstring("Flussrate", "Maximale Flussrate (mL/min):", initialvalue="2.0"))
    delta = float(simpledialog.askstring("Flussrate", "Schrittweite (mL/min):", initialvalue="0.1"))

    process_pairs(files_with, files_without, flow_min, flow_max, delta)
