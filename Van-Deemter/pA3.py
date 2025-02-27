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
  
Für jedes Dateipaar wird ein Plot mit zwei Subplots (Messung mit und ohne Säule) erzeugt.
Auf der letzten Seite des PDF-Reports wird eine Tabelle mit den relevanten Berechnungen erstellt.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.signal import peak_widths, savgol_filter
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

def process_pairs(files_with, files_without, pdf_filename="vanDeemter_Report.pdf"):
    """
    Verarbeitet die Dateipaare:
      - Liest die Messungen mit und ohne Säule ein.
      - Glättet die Signale mittels Savitzky-Golay-Filter.
      - Bestimmt das Peakmaximum und die FWHM.
      - Rechnet die FWHM in σ um und subtrahiert die extrakolumnären Einflüsse.
      - Visualisiert die Ergebnisse und speichert sie in einem PDF-Report.
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
            
            # Bestimme Fensterlänge für den Savitzky-Golay-Filter (muss ungerade sein)
            window_length_with = 11 if len(y_with) >= 11 else (len(y_with) // 2) * 2 + 1
            window_length_without = 11 if len(y_without) >= 11 else (len(y_without) // 2) * 2 + 1
            
            # Glätten der Signale
            y_with_smooth = savgol_filter(y_with, window_length=window_length_with, polyorder=3)
            y_without_smooth = savgol_filter(y_without, window_length=window_length_without, polyorder=3)
            
            # Bestimme das Peakmaximum in beiden geglätteten Signalen
            peak_idx_with = np.argmax(y_with_smooth)
            peak_idx_without = np.argmax(y_without_smooth)
            
            # Berechne die FWHM (Halbwertsbreite) beider Signale
            widths_with, _, _, _ = peak_widths(y_with_smooth, [peak_idx_with], rel_height=0.5)
            widths_without, _, _, _ = peak_widths(y_without_smooth, [peak_idx_without], rel_height=0.5)
            fwhm_with = widths_with[0] * (t_with[1]-t_with[0])
            fwhm_without = widths_without[0] * (t_without[1]-t_without[0])
            
            # Umrechnung in σ-Werte
            sigma_obs = fwhm_with / const
            sigma_ext = fwhm_without / const
            
            # Berechne σ der Säule (σ_col) via σ_obs² = σ_col² + σ_ext²
            if sigma_obs**2 >= sigma_ext**2:
                sigma_col = np.sqrt(sigma_obs**2 - sigma_ext**2)
            else:
                print(f"Warnung: Bei {shorten_filename(file_with)} ist σ_obs² kleiner als σ_ext². Setze σ_col auf 0.")
                sigma_col = 0
            
            # Umrechnung zurück in FWHM für die Säule
            fwhm_col = sigma_col * const
            
            # Ergebnisse speichern (verwende die gekürzten Dateinamen)
            results.append({
                "Datei mit Säule": shorten_filename(file_with),
                "Datei ohne Säule": shorten_filename(file_without),
                "FWHM (mit Säule) [min]": fwhm_with,
                "FWHM (ohne Säule) [min]": fwhm_without,
                "σ_obs": sigma_obs,
                "σ_ext": sigma_ext,
                "σ_col": sigma_col,
                "FWHM (Säule) [min]": fwhm_col
            })
            
            # Visualisierung: Zwei Subplots für das jeweilige Paar
            fig, axs = plt.subplots(2, 1, figsize=(8, 10))
            fig.suptitle(f"Messung {i}: {shorten_filename(file_with)} vs. {shorten_filename(file_without)}", fontsize=14)
            
            # Plot: Messung mit Säule
            axs[0].plot(t_with, y_with, 'ko', markersize=3, label="Messdaten (mit Säule)")
            axs[0].plot(t_with, y_with_smooth, 'b-', linewidth=1, label="Geglättet")
            axs[0].axvline(x=t_with[peak_idx_with] - fwhm_with/2, color='r', linestyle='--', label="FWHM-Grenzen")
            axs[0].axvline(x=t_with[peak_idx_with] + fwhm_with/2, color='r', linestyle='--')
            axs[0].set_xlabel("Zeit [min]")
            axs[0].set_ylabel("Signal")
            axs[0].legend()
            axs[0].set_title("Mit Säule")
            
            # Plot: Messung ohne Säule
            axs[1].plot(t_without, y_without, 'ko', markersize=3, label="Messdaten (ohne Säule)")
            axs[1].plot(t_without, y_without_smooth, 'b-', linewidth=1, label="Geglättet")
            axs[1].axvline(x=t_without[peak_idx_without] - fwhm_without/2, color='r', linestyle='--', label="FWHM-Grenzen")
            axs[1].axvline(x=t_without[peak_idx_without] + fwhm_without/2, color='r', linestyle='--')
            axs[1].set_xlabel("Zeit [min]")
            axs[1].set_ylabel("Signal")
            axs[1].legend()
            axs[1].set_title("Ohne Säule")
            
            # Textbox mit den berechneten Werten
            textstr = (
                f"FWHM (mit Säule): {fwhm_with:.4f} min\n"
                f"FWHM (ohne Säule): {fwhm_without:.4f} min\n"
                f"σ_obs: {sigma_obs:.4f}\n"
                f"σ_ext: {sigma_ext:.4f}\n"
                f"σ_col: {sigma_col:.4f}\n"
                f"FWHM (Säule): {fwhm_col:.4f} min"
            )
            fig.text(0.15, 0.02, textstr, fontsize=10,
                     bbox=dict(facecolor='white', alpha=0.5))
            
            pdf.savefig(fig)
            plt.close(fig)
            print(f"Messung {i} ausgewertet: {shorten_filename(file_with)} und {shorten_filename(file_without)}")
        
        # Letzte Seite: Tabelle mit allen Berechnungen
        fig, ax = plt.subplots(figsize=(12, len(results)*0.5 + 2))
        ax.axis('tight')
        ax.axis('off')
        # Tabellenkopf
        table_data = [["Datei mit Säule", "Datei ohne Säule", "FWHM (mit Säule) [min]",
                       "FWHM (ohne Säule) [min]", "σ_obs", "σ_ext", "σ_col", "FWHM (Säule) [min]"]]
        # Füge die Ergebnisse hinzu
        for res in results:
            table_data.append([
                res["Datei mit Säule"],
                res["Datei ohne Säule"],
                f"{res['FWHM (mit Säule) [min]']:.4f}",
                f"{res['FWHM (ohne Säule) [min]']:.4f}",
                f"{res['σ_obs']:.4f}",
                f"{res['σ_ext']:.4f}",
                f"{res['σ_col']:.4f}",
                f"{res['FWHM (Säule) [min]']:.4f}"
            ])
        table = ax.table(cellText=table_data, loc="center", cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        ax.set_title("Übersicht der Berechnungen", fontweight="bold")
        pdf.savefig(fig)
        plt.close(fig)
    
    print(f"\nReport wurde erstellt: {pdf_filename}")

if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()
    
    # Auswahl der Dateien: Messungen mit Säule
    files_with = filedialog.askopenfilenames(
        title="Wähle die Dateien aus (Messungen mit Säule)",
        filetypes=[("Textdateien", "*.txt"), ("Alle Dateien", "*.*")]
    )
    files_with = list(files_with)
    if not files_with:
        print("Keine Dateien für Messungen mit Säule ausgewählt.")
        sys.exit(1)
    
    # Auswahl der Dateien: Messungen ohne Säule (Reihenfolge muss übereinstimmen)
    files_without = filedialog.askopenfilenames(
        title="Wähle die Dateien aus (Messungen ohne Säule) in derselben Reihenfolge",
        filetypes=[("Textdateien", "*.txt"), ("Alle Dateien", "*.*")]
    )
    files_without = list(files_without)
    if not files_without:
        print("Keine Dateien für Messungen ohne Säule ausgewählt.")
        sys.exit(1)
    
    process_pairs(files_with, files_without)
