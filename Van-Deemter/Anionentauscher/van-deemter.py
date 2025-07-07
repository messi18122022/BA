#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams['text.latex.preamble'] = r'\usepackage{siunitx}\usepackage{amsmath}'

# Plot-Größe in Zentimetern
FIG_WIDTH_CM, FIG_HEIGHT_CM = 16, 8
# Umrechnung in Inches für Matplotlib
FIG_SIZE = (FIG_WIDTH_CM/2.54, FIG_HEIGHT_CM/2.54)

# Pfad zur CSV-Datei mit Flussraten, Retentionszeiten und Peakbreiten
CSV_FILE = "Van-Deemter/Anionentauscher/Ergebnisse_Retentionszeiten_Peakbreiten.csv"
# Säulenlänge L in mm
COLUMN_LENGTH = 100

def van_deemter(u, A, B, C):
    """Van Deemter Modell: H = A + B/u + C*u"""
    return A + B/u + C*u

def fit_and_plot(u, H, title, output_pdf):
    """Fit und Plot für gegebene H gegen Flussrate u."""
    popt, pcov = curve_fit(van_deemter, u, H)
    perr = np.sqrt(np.diag(pcov))
    u_fit = np.linspace(u.min(), u.max(), 200)
    H_fit = van_deemter(u_fit, *popt)
    # R²
    residuals = H - van_deemter(u, *popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((H - np.mean(H))**2)
    r2 = 1 - ss_res/ss_tot

    plt.figure(figsize=FIG_SIZE)
    plt.scatter(u, H, color='blue', label='Messdaten')
    plt.plot(u_fit, H_fit, 'r-', label='Fit')
    plt.xlabel(r"Flussrate $\Dot{V}$ / (\si{\milli\liter\per\minute})")
    plt.ylabel(r"HETP / (\si{\milli\meter})")
    # Titel entfernt
    plt.text(0.05, 0.95,
             f"A={popt[0]:.3f}±{perr[0]:.3f}\n"
             f"B={popt[1]:.3f}±{perr[1]:.3f}\n"
             f"C={popt[2]:.3f}±{perr[2]:.3f}\n"
             f"$R^2$={r2:.3f}",
             transform=plt.gca().transAxes,
             verticalalignment='top',
             bbox=dict(boxstyle="round", fc="white", alpha=0.5))
    plt.legend()
    plt.grid(True, zorder=0)
    plt.tight_layout()
    plt.savefig(output_pdf)
    plt.close()
    print(f"{title} gespeichert in: {output_pdf}")

def main():
    # Einlesen der Daten
    if not os.path.isfile(CSV_FILE):
        raise FileNotFoundError(f"CSV-Datei nicht gefunden: {CSV_FILE}")
    df = pd.read_csv(CSV_FILE)

    # Flussraten
    u = df["flussrate (mL/min)"].values

    # 1) Mit Säule
    tR_tot = df["retentionszeit_mit_säule (min)"].values
    sigma2_tot = df["peakbreite_mit_säule (min)"].values**2
    H_tot = COLUMN_LENGTH * sigma2_tot / (tR_tot**2)
    fit_and_plot(u, H_tot, "Van Deemter: mit Säule", 
                 os.path.join("Van-Deemter/Anionentauscher", "vd_mit_saeule.pdf"))

    # 2) Ohne Säule
    tR_ext = df["retentionszeit_ohne_säule (min)"].values
    sigma2_ext = df["peakbreite_ohne_säule (min)"].values**2
    H_ext = COLUMN_LENGTH * sigma2_ext / (tR_ext**2)
    fit_and_plot(u, H_ext, "Van Deemter: ohne Säule", 
                 os.path.join("Van-Deemter/Anionentauscher", "vd_ohne_saeule.pdf"))

    # 3) Korrigierte Werte
    tR_col = tR_tot - tR_ext
    sigma2_col = sigma2_tot - sigma2_ext
    H_col = COLUMN_LENGTH * sigma2_col / (tR_col**2)
    fit_and_plot(u, H_col, "Van Deemter: korrigiert", 
                 os.path.join("Van-Deemter/Anionentauscher", "vd_korrigiert.pdf"))

    # ---- Balkendiagramme: Retentionszeit und Peakbreite gestapelt ----
    flow_rates = u

    # Breite der Balken etwas kleiner als den Abstand der Flussraten
    width = (flow_rates[0] - flow_rates[1]) * 0.8

    # Retentionszeiten gestapelt
    ret_mit   = df["retentionszeit_mit_säule (min)"].values
    ret_ohne  = df["retentionszeit_ohne_säule (min)"].values
    plt.figure(figsize=FIG_SIZE)
    # Überlagerte Balken: Blau hinter Rot
    plt.bar(flow_rates, ret_mit, width=width, color='blue', label='mit Säule', zorder=2)
    plt.bar(flow_rates, ret_ohne, width=width, color='red', label='ohne Säule', zorder=2)
    plt.xlabel(r"Flussrate $\Dot{V}$ / (\si{\milli\liter\per\minute})")
    plt.ylabel(r"Retentionszeit $t_R$ / (\si{\minute})")
    plt.legend()
    plt.grid(True, zorder=0)
    plt.tight_layout()
    plt.savefig(os.path.join("Van-Deemter/Anionentauscher", "Balken_Retentionszeiten.pdf"))
    plt.close()

    # Peakbreiten gestapelt
    sigma_mit  = df["peakbreite_mit_säule (min)"].values
    sigma_ohne = df["peakbreite_ohne_säule (min)"].values
    plt.figure(figsize=FIG_SIZE)
    # Überlagerte Balken: Blau hinter Rot
    plt.bar(flow_rates, sigma_mit, width=width, color='blue', label='mit Säule', zorder=2)
    plt.bar(flow_rates, sigma_ohne, width=width, color='red', label='ohne Säule', zorder=2)
    plt.xlabel(r"Flussrate $\Dot{V}$ / (\si{\milli\liter\per\minute})")
    plt.ylabel(r"Peakbreite $\sigma$ / (\si{\minute})")
    plt.legend()
    plt.grid(True, zorder=0)
    plt.tight_layout()
    plt.savefig(os.path.join("Van-Deemter/Anionentauscher", "Balken_Peakbreiten.pdf"))
    plt.close()

if __name__ == "__main__":
    main()