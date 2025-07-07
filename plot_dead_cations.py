#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# LaTeX aktivieren und Schriftart setzen
mpl.rc('text', usetex=True)
mpl.rc('font', family='serif')
# siunitx & amsmath einbinden
mpl.rcParams['text.latex.preamble'] = r'\usepackage{siunitx}\usepackage{amsmath}'

# Plot-Größe (16×8 cm)
FIG_SIZE = (16/2.54, 8/2.54)

# Pfad zur CSV-Datei
CSV_FILE = "Van-Deemter/Kationentauscher/Totvolumen_Säule.csv"

def main():
    # Datei prüfen und einlesen (Index enthält evtl. Summary-Zeilen)
    if not os.path.isfile(CSV_FILE):
        raise FileNotFoundError(f"CSV-Datei nicht gefunden: {CSV_FILE}")
    df = pd.read_csv(CSV_FILE, index_col=0)
    
    # Nur numerische Messreihen verwenden (Summary-Zeilen wie "Mittelwert"/"Standardabweichung" ausschließen)
    df = df.loc[~df.index.isin(["Mittelwert", "Standardabweichung"])]
    df["flussrate (mL/min)"] = pd.to_numeric(df["flussrate (mL/min)"], errors="coerce")
    df["Totvolumen_Saeule (mL)"] = pd.to_numeric(df["Totvolumen_Saeule (mL)"], errors="coerce")
    df = df.dropna(subset=["flussrate (mL/min)", "Totvolumen_Saeule (mL)"])
    
    # Daten fürs Plot
    u = df["flussrate (mL/min)"].values
    V = df["Totvolumen_Saeule (mL)"].values
    
    # Mittelwert & Standardabweichung
    mean_V = np.mean(V)
    std_V  = np.std(V, ddof=1)
    
    # Plot erzeugen
    plt.figure(figsize=FIG_SIZE)
    plt.scatter(u, V, marker='o', zorder=2, label="Messwerte")
    # Mittelwert als Linie
    plt.axhline(mean_V, color='red', linestyle='--', zorder=1,
                label=rf"Mittelwert = {mean_V:.3f}\,\si{{\milli\liter}}")
    
    # Achsenbeschriftung mit SI-Einheiten
    plt.xlabel(r"Flussrate $\dot V$ / (\si{\milli\liter\per\minute})")
    plt.ylabel(r"Totvolumen $V^\mathrm{dead}_\mathrm{col}$ / (\si{\milli\liter})")
    
    plt.grid(True, zorder=0)
    plt.legend()
    plt.tight_layout()
    
    # Speichern und anzeigen
    out_pdf = "plot_totvolumen.pdf"
    plt.savefig(out_pdf, bbox_inches='tight')
    print(f"Plot gespeichert als: {out_pdf}")
    plt.show()

if __name__ == "__main__":
    main()