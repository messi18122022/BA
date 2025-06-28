#!/usr/bin/env python3

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams['text.latex.preamble'] = r'\usepackage{siunitx}\usepackage{amsmath}'

# --------------------- Konfiguration ---------------------
ORDNER_MIT_SÄULE = "Van-Deemter/Anionentauscher/mit_saule"
ORDNER_OHNE_SÄULE = "Van-Deemter/Anionentauscher/ohne_saule"
OUTPUT_CSV = "Van-Deemter/Anionentauscher/Ergebnisse_Retentionszeiten_Peakbreiten.csv"

# Standard-Flussraten-Vektor (2.0, 1.9, ..., 0.1)
default_flow_rates = np.arange(2.0, 0.0, -0.1)

# ----- Manuelle Integrationsgrenzen -----
INTEGRATION_LIMITS_MIT = {
    "2025-03-07_S023_H2O_01.txt": (0.5, 0.7),
    "2025-03-07_S023_H2O_02.txt": (0.5, 0.75),
    "2025-03-07_S023_H2O_03.txt": (0.55, 0.80),
    "2025-03-07_S023_H2O_04.txt": (0.58, 0.82),
    "2025-03-07_S023_H2O_05.txt": (0.61, 0.86),
    "2025-03-07_S023_H2O_06.txt": (0.65, 0.92),
    "2025-03-07_S023_H2O_07.txt": (0.7, 1),
    "2025-03-07_S023_H2O_08.txt": (0.75, 1.05),
    "2025-03-07_S023_H2O_09.txt": (0.80, 1.13),
    "2025-03-07_S023_H2O_10.txt": (0.90, 1.225),
    "2025-03-07_S023_H2O_11.txt": (0.977, 1.33),
    "2025-03-07_S023_H2O_12.txt": (1.098, 1.49),
    "2025-03-07_S023_H2O_13.txt": (1.224, 1.668),
    "2025-03-07_S023_H2O_14.txt": (1.397, 1.89),
    "2025-03-07_S023_H2O_15.txt": (1.700, 2.215),
    "2025-03-07_S023_H2O_16.txt": (2.082, 2.644),
    "2025-03-07_S023_H2O_17.txt": (2.591, 3.274),
    "2025-03-07_S023_H2O_18.txt": (3.51, 4.36),
    "2025-03-07_S023_H2O_19.txt": (5.304, 6.451),
    "2025-03-07_S023_H2O_20.txt": (10.77, 13)
}

INTEGRATION_LIMITS_OHNE = {
    "2025-03-12_S025_H2O_01.txt": (0.113, 0.282),
    "2025-03-12_S025_H2O_02.txt": (0.123, 0.297),
    "2025-03-12_S025_H2O_03.txt": (0.128, 0.316),
    "2025-03-12_S025_H2O_04.txt": (0.147, 0.36),
    "2025-03-12_S025_H2O_05.txt": (0.181, 0.37),
    "2025-03-12_S025_H2O_06.txt": (0.186, 0.384),
    "2025-03-12_S025_H2O_07.txt": (0.2, 0.42),
    "2025-03-12_S025_H2O_08.txt": (0.205, 0.45),
    "2025-03-12_S025_H2O_09.txt": (0.223, 0.483),
    "2025-03-12_S025_H2O_10.txt": (0.24, 0.53),
    "2025-03-12_S025_H2O_11.txt": (0.262, 0.55),
    "2025-03-12_S025_H2O_12.txt": (0.3, 0.627),
    "2025-03-12_S025_H2O_13.txt": (0.34, 0.693),
    "2025-03-12_S025_H2O_14.txt": (0.393, 0.784),
    "2025-03-12_S025_H2O_15.txt": (0.466, 0.9),
    "2025-03-12_S025_H2O_16.txt": (0.57, 1.08),
    "2025-03-12_S025_H2O_17.txt": (0.726, 1.314),
    "2025-03-12_S025_H2O_18.txt": (0.99, 1.726),
    "2025-03-12_S025_H2O_19.txt": (1.5, 2.541),
    "2025-03-12_S025_H2O_20.txt": (3.15, 4.90)
}

def momentanalyse(zeit, signal):
    """Berechnet Retentionszeit (t_mean) und Standardabweichung sigma."""
    m0 = np.sum(signal)
    if m0 == 0:
        return np.nan, np.nan
    t_mean = np.sum(zeit * signal) / m0
    varianz = np.sum(((zeit - t_mean) ** 2) * signal) / m0
    return t_mean, np.sqrt(varianz)

def process_files(filepaths, beschriftung, integrationsgrenzen):
    tR_list = []
    sigma2_list = []
    for datei in filepaths:
        df = pd.read_csv(datei, delimiter=',')
        zeit = df.iloc[:, 0].values
        signal = df.iloc[:, 1].values
        signal_plot = -signal

        name = os.path.basename(datei)
        if name not in integrationsgrenzen:
            print(f"Warnung: Keine Integrationsgrenzen für {name}.")
            continue
        start, ende = integrationsgrenzen[name]
        i1 = np.argmin(np.abs(zeit - start))
        i2 = np.argmin(np.abs(zeit - ende))
        if i1 >= i2:
            print(f"Warnung: Ungültige Grenzen in {name}.")
            continue

        t_slice = zeit[i1:i2+1]
        s_slice = signal[i1:i2+1]
        s_slice_plot = -s_slice

        t_mean, sigma = momentanalyse(t_slice, s_slice)
        if np.isnan(t_mean):
            print(f"Warnung: Keine gültigen Werte in {name}.")
            continue

        tR_list.append(t_mean)
        sigma2_list.append(sigma**2)

        # Erzeuge Plot: Gesamtes Chromatogramm + Zoom
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 3))
        # Gesamt
        axes[0].plot(zeit, signal_plot, color="blue", label="Signal")
        axes[0].axvspan(zeit[i1], zeit[i2], color="orange", alpha=0.2, label="Peak-Fenster")
        axes[0].axvline(t_mean, color="red", linestyle="--", label="Retentionszeit")
        axes[0].axvspan(t_mean - sigma, t_mean + sigma, color="red", alpha=0.1, label=r"$t_R \pm \sigma$")
        axes[0].set_xlabel(r"Zeit $t$ / min")
        axes[0].set_ylabel(r"Leitfähigkeit $\sigma$ / \si{\micro\siemens\per\centi\meter}")
        axes[0].invert_yaxis()
        axes[0].legend(loc="upper right")

        # Zoom
        axes[1].plot(zeit, signal_plot, color="blue")
        axes[1].axvspan(zeit[i1], zeit[i2], color="orange", alpha=0.2)
        axes[1].axvline(t_mean, color="red", linestyle="--")
        axes[1].axvspan(t_mean - sigma, t_mean + sigma, color="red", alpha=0.1)
        axes[1].set_xlabel(r"Zeit $t$ / min")
        axes[1].set_ylabel(r"Leitfähigkeit $\sigma$ / \si{\micro\siemens\per\centi\meter}")
        x_left = zeit[i1] - 0.1*(zeit[i2]-zeit[i1])
        x_right = zeit[i2] + 0.1*(zeit[i2]-zeit[i1])
        axes[1].set_xlim(x_left, x_right)
        peak_min = np.min(s_slice_plot)
        peak_max = np.max(s_slice_plot)
        y_margin = 0.1*(peak_max - peak_min) if peak_max != peak_min else 0
        axes[1].set_ylim(peak_min - y_margin, peak_max + y_margin)
        axes[1].invert_yaxis()
        axes[1].text(
            0.95, 0.95,
            rf"$t_R = {t_mean:.3f}\,\mathrm{{min}}$",
            transform=axes[1].transAxes,
            ha='right',
            va='top',
            bbox=dict(boxstyle="round", fc="white", alpha=0.5)
        )

        plt.tight_layout(rect=[0, 0.03, 1, 0.92])
        output_pdf = os.path.join(os.path.dirname(datei), os.path.splitext(name)[0] + ".pdf")
        plt.savefig(output_pdf)
        plt.close()

    print(f"Alle {beschriftung} Plots wurden als PDF gespeichert.")
    return tR_list, sigma2_list

def main():
    files_mit = sorted(glob.glob(os.path.join(ORDNER_MIT_SÄULE, "*.csv")) +
                       glob.glob(os.path.join(ORDNER_MIT_SÄULE, "*.txt")))
    tR_mit, sigma2_mit = process_files(files_mit, "mit Säule", INTEGRATION_LIMITS_MIT)

    files_ohne = sorted(glob.glob(os.path.join(ORDNER_OHNE_SÄULE, "*.csv")) +
                        glob.glob(os.path.join(ORDNER_OHNE_SÄULE, "*.txt")))
    tR_ohne, sigma2_ohne = process_files(files_ohne, "ohne Säule", INTEGRATION_LIMITS_OHNE)

    # Flussraten passend zu tR_mit
    flow_rates = np.linspace(default_flow_rates[0], default_flow_rates[-1], len(tR_mit))

    # CSV schreiben
    df = pd.DataFrame({
        "flussrate (mL/min)": flow_rates,
        "retentionszeit_mit_säule (min)": tR_mit,
        "retentionszeit_ohne_säule (min)": tR_ohne,
        "peakbreite_mit_säule (min)": np.sqrt(sigma2_mit),
        "peakbreite_ohne_säule (min)": np.sqrt(sigma2_ohne),
    })
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Ergebnisse gespeichert in: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()