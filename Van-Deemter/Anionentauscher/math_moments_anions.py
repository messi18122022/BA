import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
from scipy.optimize import curve_fit

# --------------------- Konfiguration ---------------------
# Ordner, in denen die Messdateien hinterlegt sind
ORDNER_MIT_SÄULE = "Van-Deemter/Anionentauscher/mit_saule"
ORDNER_OHNE_SÄULE = "Van-Deemter/Anionentauscher/ohne_saule"

# Ordner, in dem die Plots gespeichert werden sollen
OUTPUT_DIR = "Van-Deemter/Anionentauscher"  # Wird für den van Deemter Plot und die H-Werte-Rechnung genutzt

# Standard-Flussraten-Vektor (Standard: 2.0, 1.9, 1.8, ..., 0.1)
default_flow_rates = np.arange(2.0, 0.0, -0.1)

# ----- Manuelle Integrationsgrenzen festlegen -----
# Die Schlüssel müssen exakt den Dateinamen entsprechen.
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
# ----------------------------------------------------------

def momentanalyse(zeit, signal):
    m0 = np.sum(signal)
    if m0 == 0:
        return np.nan, np.nan
    t_mean = np.sum(zeit * signal) / m0
    varianz = np.sum(((zeit - t_mean) ** 2) * signal) / m0
    stdev = np.sqrt(varianz)
    return t_mean, stdev

def process_files(filepaths, beschriftung, integrationsgrenzen):
    tR_list = []
    sigma2_list = []
    for datei in filepaths:
        try:
            df = pd.read_csv(datei, delimiter=',')
            if df.shape[1] < 2:
                print(f"Warnung: Die Datei '{datei}' hat nicht genügend Spalten.")
                continue
            zeit = df.iloc[:, 0].values
            signal = df.iloc[:, 1].values
            
            # Hole die Integrationsgrenzen anhand des Dateinamens
            dateiname = os.path.basename(datei)
            if dateiname not in integrationsgrenzen:
                print(f"Warnung: Für Datei '{dateiname}' wurden keine Integrationsgrenzen definiert. Datei wird übersprungen.")
                continue
            start_min, end_min = integrationsgrenzen[dateiname]
            
            left_idx = np.argmin(np.abs(zeit - start_min))
            right_idx = np.argmin(np.abs(zeit - end_min))
            if left_idx >= right_idx:
                print(f"Warnung: Für Datei '{dateiname}' sind die Integrationsgrenzen ungültig.")
                continue
            
            t_slice = zeit[left_idx:right_idx+1]
            s_slice = signal[left_idx:right_idx+1]
            t_mean, sigma = momentanalyse(t_slice, s_slice)
            if np.isnan(t_mean) or np.isnan(sigma):
                print(f"Warnung: Keine gültigen Werte in '{dateiname}' gefunden.")
                continue
            sigma2 = sigma**2
            tR_list.append(t_mean)
            sigma2_list.append(sigma2)
            
            # Erzeuge Plot: Gesamtes Chromatogramm und Zoom auf den Peak (Titel entfernt)
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            axes[0].plot(zeit, signal, color="blue", label="Signal")
            axes[0].axvspan(zeit[left_idx], zeit[right_idx], color="orange", alpha=0.2, label="Peak-Fenster")
            axes[0].axvline(t_mean, color="red", linestyle="--", label="Retentionszeit")
            axes[0].axvspan(t_mean - sigma, t_mean + sigma, color="red", alpha=0.1, label="t\_mean $\pm\sigma$")
            axes[1].plot(zeit, signal, color="blue", label="Signal")
            axes[1].axvspan(zeit[left_idx], zeit[right_idx], color="orange", alpha=0.2)
            axes[1].axvline(t_mean, color="red", linestyle="--", label="Retentionszeit")
            axes[1].axvspan(t_mean - sigma, t_mean + sigma, color="red", alpha=0.1, label="t\_mean $\pm\sigma$")
            # Dynamische X-Achsen-Berechnung für den Zoom
            x_left = zeit[left_idx] - 0.1 * (zeit[right_idx] - zeit[left_idx])
            x_right = zeit[right_idx] + 0.1 * (zeit[right_idx] - zeit[left_idx])
            axes[1].set_xlim(x_left, x_right)
            peak_min = np.min(s_slice)
            peak_max = np.max(s_slice)
            if peak_min != peak_max:
                y_margin = 0.1 * (peak_max - peak_min)
                axes[1].set_ylim(peak_min - y_margin, peak_max + y_margin)
            plt.tight_layout(rect=[0, 0.03, 1, 0.92])
            base_name = os.path.splitext(dateiname)[0]
            output_path = os.path.join(os.path.dirname(datei), base_name + ".pdf")
            plt.savefig(output_path)
            plt.close()
        except Exception as e:
            print(f"Fehler beim Verarbeiten der Datei '{datei}': {e}")
    print(f"Alle {beschriftung} Plots wurden als PDF gespeichert.")
    return tR_list, sigma2_list

def erstelle_hwerte_pdf(tR_total_list, sigma2_total_list, tR_extra_list, sigma2_extra_list,
                         tR_column_list, sigma_column_list, N_list, HETP_list, output_dir):
    """
    Erzeugt ein PDF, in dem der Rechenweg für die H-Werte (Säulenparameter) für jede Messung
    detailliert aufgeführt wird.
    """
    daten = []
    for i in range(len(tR_total_list)):
        messung = i + 1
        tR_total = f"{tR_total_list[i]:.3f}"
        sigma2_total = f"{sigma2_total_list[i]:.3f}"
        tR_extra = f"{tR_extra_list[i]:.3f}"
        sigma2_extra = f"{sigma2_extra_list[i]:.3f}"
        tR_column = f"{tR_column_list[i]:.3f}"
        sigma_column = f"{sigma_column_list[i]:.3f}"
        N_val = f"{N_list[i]:.3f}"
        HETP = f"{HETP_list[i]:.3f}"
        daten.append([messung, tR_total, sigma2_total, tR_extra, sigma2_extra, tR_column, sigma_column, N_val, HETP])
    
    spalten = ["Messung", "tR_total (min)", "σ²_total (min²)", "tR_extra (min)", "σ²_extra (min²)", 
               "tR_column (min)", "σ_column (min)", "N", "HETP (mm)"]
    
    fig, ax = plt.subplots(figsize=(12, 0.5 * (len(daten)+2)))
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=daten, colLabels=spalten, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    
    output_file = os.path.join(output_dir, "Hwerte_Berechnungen.pdf")
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    print("Das PDF mit den H-Werte-Berechnungen wurde erfolgreich erstellt und gespeichert.")

def main():
    # Dateien aus den angegebenen Ordnern laden
    files_with_column = sorted(
        glob.glob(os.path.join(ORDNER_MIT_SÄULE, "*.csv")) + 
        glob.glob(os.path.join(ORDNER_MIT_SÄULE, "*.txt"))
    )
    if not files_with_column:
        print("Es wurden keine Dateien für Messungen mit Säule gefunden.")
        return
    tR_total_list, sigma2_total_list = process_files(files_with_column, "mit Säule", INTEGRATION_LIMITS_MIT)

    files_without_column = sorted(
        glob.glob(os.path.join(ORDNER_OHNE_SÄULE, "*.csv")) + 
        glob.glob(os.path.join(ORDNER_OHNE_SÄULE, "*.txt"))
    )
    if not files_without_column:
        print("Es wurden keine Dateien für Messungen ohne Säule gefunden.")
        return
    tR_extra_list, sigma2_extra_list = process_files(files_without_column, "ohne Säule", INTEGRATION_LIMITS_OHNE)

    if len(tR_total_list) != len(tR_extra_list):
        print("Die Anzahl der Messungen mit und ohne Säule stimmt nicht überein.")
        return

    # Aggregierte Plots der Retentionszeiten gegen Flussrate
    if tR_total_list and tR_extra_list:
        # Flussraten-Vektor für Messungen mit Säule
        flow_rates_total = np.linspace(default_flow_rates[0], default_flow_rates[-1], len(tR_total_list))
        plt.figure(figsize=(4, 4))
        plt.scatter(flow_rates_total, tR_total_list, color='blue', label='Mit Säule')
        plt.xlabel("Flussrate (mL/min)")
        plt.ylabel("Retentionszeit $t_R$ (min)")
        plt.legend()
        aggregated_with_output = os.path.join(OUTPUT_DIR, 'Aggregated_Mit_Saule.pdf')
        plt.savefig(aggregated_with_output)
        plt.close()
        print("Der aggregierte Plot mit Messdaten (Retentionszeiten, mit Säule) wurde erfolgreich erstellt und gespeichert.")

        # Flussraten-Vektor für Messungen ohne Säule
        flow_rates_extra = np.linspace(default_flow_rates[0], default_flow_rates[-1], len(tR_extra_list))
        plt.figure(figsize=(4, 4))
        plt.scatter(flow_rates_extra, tR_extra_list, color='green', label='Ohne Säule')
        plt.xlabel("Flussrate (mL/min)")
        plt.ylabel("Retentionszeit $t_R$ (min)")
        plt.legend()
        aggregated_without_output = os.path.join(OUTPUT_DIR, 'Aggregated_Ohne_Saule.pdf')
        plt.savefig(aggregated_without_output)
        plt.close()
        print("Der aggregierte Plot mit Messdaten (Retentionszeiten, ohne Säule) wurde erfolgreich erstellt und gespeichert.")

    # Aggregierte Plots der Peakbreiten gegen Flussrate
    if tR_total_list and tR_extra_list:
        # Flussraten-Vektor für Messungen mit Säule
        flow_rates_total = np.linspace(default_flow_rates[0], default_flow_rates[-1], len(tR_total_list))
        sigma_total = np.sqrt(np.array(sigma2_total_list))
        plt.figure(figsize=(4, 4))
        plt.scatter(flow_rates_total, sigma_total, color='blue', label='Mit Säule')
        plt.xlabel("Flussrate (mL/min)")
        plt.ylabel("Peakbreite $\sigma$ (min)")
        plt.legend()
        aggregated_peak_total = os.path.join(OUTPUT_DIR, 'Aggregated_Peakbreite_Mit_Saule.pdf')
        plt.savefig(aggregated_peak_total)
        plt.close()
        print("Der aggregierte Plot der Peakbreiten (mit Säule) wurde erfolgreich erstellt und gespeichert.")

        # Flussraten-Vektor für Messungen ohne Säule
        flow_rates_extra = np.linspace(default_flow_rates[0], default_flow_rates[-1], len(tR_extra_list))
        sigma_extra = np.sqrt(np.array(sigma2_extra_list))
        plt.figure(figsize=(4, 4))
        plt.scatter(flow_rates_extra, sigma_extra, color='green', label='Ohne Säule')
        plt.xlabel("Flussrate (mL/min)")
        plt.ylabel("Peakbreite $\sigma$ (min)")
        plt.legend()
        aggregated_peak_extra = os.path.join(OUTPUT_DIR, 'Aggregated_Peakbreite_Ohne_Saule.pdf')
        plt.savefig(aggregated_peak_extra)
        plt.close()
        print("Der aggregierte Plot der Peakbreiten (ohne Säule) wurde erfolgreich erstellt und gespeichert.")

    # Neuer Block: van Deemter Plot ohne Systemkorrektur
    # Hier werden tR_total und sigma_total direkt genutzt, ohne systematische Korrektur durch tR_extra usw.
    if tR_total_list:
        flow_rates_total = np.linspace(default_flow_rates[0], default_flow_rates[-1], len(tR_total_list))
        sigma_total = np.sqrt(np.array(sigma2_total_list))
        # Berechne die Anzahl der theoretischen Böden und HETP
        N_total = (np.array(tR_total_list) / sigma_total)**2
        HETP_total = 100 / N_total
        # van Deemter Modell: HETP = A + B/Flussrate + C*Flussrate
        def van_deemter(u, A, B, C):
            return A + B/u + C*u
        try:
            popt, pcov = curve_fit(van_deemter, flow_rates_total, HETP_total)
            perr = np.sqrt(np.diag(pcov))
            u_fit = np.linspace(flow_rates_total.min(), flow_rates_total.max(), 200)
            HETP_fit = van_deemter(u_fit, *popt)
            residuals = HETP_total - van_deemter(flow_rates_total, *popt)
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((HETP_total - np.mean(HETP_total))**2)
            r_squared = 1 - (ss_res / ss_tot)
        except Exception as e:
            print(f"Fehler beim Fitten des van Deemter Plots ohne Systemkorrektur: {e}")
            popt = [np.nan, np.nan, np.nan]
            perr = [np.nan, np.nan, np.nan]
            u_fit = None
            HETP_fit = None
            r_squared = np.nan
        plt.figure(figsize=(8, 6))
        plt.scatter(flow_rates_total, HETP_total, color='purple', label='Messdaten ohne Systemkorrektur')
        if u_fit is not None:
            plt.plot(u_fit, HETP_fit, 'r-', label=r"Fit: HETP = A + B/Flussrate + C$\cdot$Flussrate")
        plt.xlabel("Flussrate (mL/min)")
        plt.ylabel("HETP (mm)")
        plt.grid(True)
        fit_text = (f"A = {popt[0]:.3f} ± {perr[0]:.3f}\n"
                    f"B = {popt[1]:.3f} ± {perr[1]:.3f}\n"
                    f"C = {popt[2]:.3f} ± {perr[2]:.3f}\n"
                    f"$R^2$ = {r_squared:.3f}")
        plt.text(0.05, 0.95, fit_text, transform=plt.gca().transAxes, fontsize=10,
                 verticalalignment='top', bbox=dict(boxstyle="round", fc="w"))
        plt.legend()
        plt.tight_layout()
        van_deemter_total_output_file = os.path.join(OUTPUT_DIR, "van_Deemter_Plot_ohne_Systemkorrektur.pdf")
        plt.savefig(van_deemter_total_output_file)
        plt.close()
        print("Der van Deemter Plot ohne Systemkorrektur wurde erfolgreich erstellt und gespeichert.")

    # ---- Berechnung der Säulen-Parameter ----
    tR_column_list = []
    sigma_column_list = []
    N_list = []
    HETP_list = []
    for i in range(len(tR_total_list)):
        tR_column = tR_total_list[i] - tR_extra_list[i]
        delta_sigma2 = sigma2_total_list[i] - sigma2_extra_list[i]
        if delta_sigma2 <= 0:
            print(f"Warnung: Für Messung {i+1} ist $\sigma^2_{{total}} - \sigma^2_{{extra}} = {delta_sigma2}$ <= 0. Überspringe Messung.")
            continue
        sigma_column = np.sqrt(delta_sigma2)
        if sigma_column <= 0:
            print(f"Warnung: Für Messung {i+1} ergibt sich ein ungültiger $\sigma_{{column}}$ (Wert: {sigma_column}).")
            continue
        N = (tR_column / sigma_column) ** 2
        HETP = 100 / N
        tR_column_list.append(tR_column)
        sigma_column_list.append(sigma_column)
        N_list.append(N)
        HETP_list.append(HETP)

    # Neuer Block: van Deemter Plot für Messungen ohne Säule
    if tR_extra_list:
        flow_rates_ohne = np.linspace(default_flow_rates[0], default_flow_rates[-1], len(tR_extra_list))
        sigma_ohne = np.sqrt(np.array(sigma2_extra_list))
        N_ohne = (np.array(tR_extra_list) / sigma_ohne)**2
        HETP_ohne = 100 / N_ohne
        def van_deemter(u, A, B, C):
            return A + B/u + C*u
        try:
            popt_ex, pcov_ex = curve_fit(van_deemter, flow_rates_ohne, HETP_ohne)
            perr_ex = np.sqrt(np.diag(pcov_ex))
            u_fit_ex = np.linspace(flow_rates_ohne.min(), flow_rates_ohne.max(), 200)
            HETP_fit_ex = van_deemter(u_fit_ex, *popt_ex)
            residuals_ex = HETP_ohne - van_deemter(flow_rates_ohne, *popt_ex)
            ss_res_ex = np.sum(residuals_ex**2)
            ss_tot_ex = np.sum((HETP_ohne - np.mean(HETP_ohne))**2)
            r_squared_ex = 1 - (ss_res_ex / ss_tot_ex)
        except Exception as e:
            print(f"Fehler beim Fitten des van Deemter Plots für Messungen ohne Säule: {e}")
            popt_ex = [np.nan, np.nan, np.nan]
            perr_ex = [np.nan, np.nan, np.nan]
            u_fit_ex = None
            HETP_fit_ex = None
            r_squared_ex = np.nan
        plt.figure(figsize=(8, 6))
        plt.scatter(flow_rates_ohne, HETP_ohne, color='orange', label='Messdaten ohne Säule')
        if u_fit_ex is not None:
            plt.plot(u_fit_ex, HETP_fit_ex, 'r-', label=r"Fit: HETP = A + B/Flussrate + C$\cdot$Flussrate")
        plt.xlabel("Flussrate (mL/min)")
        plt.ylabel("HETP (mm)")
        plt.grid(True)
        fit_text_ex = (f"A = {popt_ex[0]:.3f} ± {perr_ex[0]:.3f}\\n"
                       f"B = {popt_ex[1]:.3f} ± {perr_ex[1]:.3f}\\n"
                       f"C = {popt_ex[2]:.3f} ± {perr_ex[2]:.3f}\\n"
                       f"$R^2$ = {r_squared_ex:.3f}")
        plt.text(0.05, 0.95, fit_text_ex, transform=plt.gca().transAxes, fontsize=10,
                 verticalalignment='top', bbox=dict(boxstyle="round", fc="w"))
        plt.legend()
        plt.tight_layout()
        van_deemter_ohne_output_file = os.path.join(OUTPUT_DIR, "van_Deemter_Plot_fuer_Ohne_Saule.pdf")
        plt.savefig(van_deemter_ohne_output_file)
        plt.close()
        print("Der van Deemter Plot für Messungen ohne Säule wurde erfolgreich erstellt und gespeichert.")

    # Erstelle das PDF mit den H-Werte-Berechnungen
    erstelle_hwerte_pdf(
        tR_total_list, sigma2_total_list,
        tR_extra_list, sigma2_extra_list,
        tR_column_list, sigma_column_list,
        N_list, HETP_list,
        OUTPUT_DIR
    )

    # Erstelle Excel-Tabelle mit den berechneten Werten
    excel_data = {
        "Messung": list(range(1, len(tR_total_list) + 1)),
        "tR_total (min)": tR_total_list,
        "σ²_total (min²)": sigma2_total_list,
        "tR_extra (min)": tR_extra_list,
        "σ²_extra (min²)": sigma2_extra_list,
        "tR_column (min)": tR_column_list,
        "σ_column (min)": sigma_column_list,
        "N": N_list,
        "HETP (mm)": HETP_list
    }
    df_excel = pd.DataFrame(excel_data)
    excel_output_file = os.path.join(OUTPUT_DIR, "Hwerte_Berechnungen.xlsx")
    df_excel.to_excel(excel_output_file, index=False)
    print("Die Excel-Tabelle mit den H-Werte-Berechnungen wurde erfolgreich erstellt und gespeichert.")

if __name__ == '__main__':
    main()