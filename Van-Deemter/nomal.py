import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import glob

def compute_moments(time, signal):
    """
    Berechnet die ersten beiden Momente eines Peaks.
    
    Parameters:
    -----------
    time : ndarray
        Zeitachse.
    signal : ndarray
        Signal (z.B. Leitfähigkeit), das proportional zur Konzentration ist.
        
    Returns:
    --------
    mu0 : float
        Flächenintegral des Signals.
    mu1 : float
        Erster Moment (Mittelwert).
    mu2 : float
        Zweiter zentraler Moment (Varianz).
    """
    # Integrieren (Trapezregel)
    mu0 = np.trapz(signal, time)
    # Erster Moment
    mu1 = np.trapz(time * signal, time) / mu0
    # Zweiter zentraler Moment
    mu2 = np.trapz((time - mu1)**2 * signal, time) / mu0
    return mu0, mu1, mu2

def load_data(file_path):
    """
    Liest eine Messdatei ein.
    Annahme: Die Datei hat zwei Spalten (z.B. Zeit und Leitfähigkeit),
    getrennt durch Leerzeichen oder Tabulatoren.
    """
    df = pd.read_csv(file_path, sep='\s+', header=None, names=["time", "signal"])
    # Konvertiere die Spalten in numerische Werte:
    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    df["signal"] = pd.to_numeric(df["signal"], errors="coerce")
    return df["time"].values, df["signal"].values

def hetp_model(u0, A, B, C):
    return A + B/u0 + C*u0

def main():
    # Passe die Pfade zu Deinen Dateien an:
    files_no_column = sorted(glob.glob("Van-Deemter/Kationentauscher/ohne_saule/*.txt"))
    files_column    = sorted(glob.glob("Van-Deemter/Kationentauscher/mit_saule/*.txt"))
    
    mu0_no_col, mu1_no_col, mu2_no_col = [], [], []
    mu0_col, mu1_col, mu2_col = [], [], []
    flow_rates = []
    
    # Extrakolumnare Messungen einlesen
    for file in files_no_column:
        time, signal = load_data(file)
        m0, m1, m2 = compute_moments(time, signal)
        mu0_no_col.append(m0)
        mu1_no_col.append(m1)
        mu2_no_col.append(m2)
    
    # Säulenmessungen einlesen
    for file in files_column:
        time, signal = load_data(file)
        m0, m1, m2 = compute_moments(time, signal)
        mu0_col.append(m0)
        mu1_col.append(m1)
        mu2_col.append(m2)
        
        # Beispiel: Extrahiere u0 aus dem Dateinamen, z.B. "..._0.2.txt"
        try:
            u0 = float(file.split("_")[-1].replace(".txt", ""))
        except Exception:
            u0 = np.nan
        flow_rates.append(u0)
    
    df_no_col = pd.DataFrame({"mu0": mu0_no_col, "mu1": mu1_no_col, "mu2": mu2_no_col})
    df_col    = pd.DataFrame({"u0": flow_rates, "mu0": mu0_col, "mu1": mu1_col, "mu2": mu2_col})
    
    # Berechnung der extrakolumnaren Beiträge
    mu1_extra = df_no_col["mu1"].mean()
    mu2_extra = df_no_col["mu2"].mean()
    
    df_col["mu1_corr"] = df_col["mu1"] - mu1_extra
    df_col["mu2_corr"] = df_col["mu2"] - mu2_extra
    
    # Beispielhafte Berechnung von H_total (angepasst an Deine Situation)
    df_col["H_total"] = df_col["mu2_corr"] / (df_col["mu1_corr"]**2)
    
    df_col = df_col.dropna(subset=["u0", "H_total"])
    
    plt.figure()
    plt.scatter(df_col["u0"], df_col["H_total"], label="Daten")
    plt.xlabel("u0 (cm/s)")
    plt.ylabel("H_total (m)")
    plt.legend()
    plt.show()
    
    u0_vals = df_col["u0"].values
    H_total_vals = df_col["H_total"].values
    
    p0 = [1e-4, 1e-4, 1e-4]
    
    popt, pcov = curve_fit(hetp_model, u0_vals, H_total_vals, p0=p0)
    perr = np.sqrt(np.diag(pcov))
    
    print("Geschätzte Parameter:")
    print(f"A = {popt[0]:.3e} ± {perr[0]:.3e}")
    print(f"B = {popt[1]:.3e} ± {perr[1]:.3e}")
    print(f"C = {popt[2]:.3e} ± {perr[2]:.3e}")
    
    u0_fit = np.linspace(min(u0_vals), max(u0_vals), 100)
    H_fit = hetp_model(u0_fit, *popt)
    
    plt.figure()
    plt.scatter(u0_vals, H_total_vals, label="Daten")
    plt.plot(u0_fit, H_fit, "r-", label="Fit")
    plt.xlabel("u0 (cm/s)")
    plt.ylabel("H_total (m)")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()