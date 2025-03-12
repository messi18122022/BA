import tkinter as tk
from tkinter import filedialog
import re
import PyPDF2
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# Matplotlib so konfigurieren, dass LaTeX verwendet wird:
mpl.rc('text', usetex=True)
mpl.rc('font', family='serif')
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath} \usepackage{siunitx}'

def parse_pdf(file_path):
    data = {}
    with open(file_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    
    # Extrahiere Ident ohne den Zusatz " Ident"
    match_ident = re.search(r'(\d{4}-\d{2}-\d{2}_[^\s]+)(?=\s*Ident)', text)
    data['Ident'] = match_ident.group(1) if match_ident else None

    # Erwartete Zeilen:
    # Zeile 1: "Retentionszeit Halbwertsbreite"
    # Zeile 2: "min min"
    # Zeile 3: Zahlen, z.B. "1,065 0,063"
    pattern = r'Retentionszeit\s+Halbwertsbreite\s*\n\s*min\s+min\s*\n\s*([\d,]+)\s+([\d,]+)'
    match_rhw = re.search(pattern, text)
    if match_rhw:
        retention_str = match_rhw.group(1).replace(',', '.')
        half_str = match_rhw.group(2).replace(',', '.')
        retention_value = float(retention_str)
        # Subtrahiere 0.0002025 von der Halbwertsbreite (manuelle Korrektur)
        half_value = float(half_str) - 0.0002025
        data['Retentionszeit'] = retention_value
        data['Halbwertsbreite'] = half_value
    else:
        data['Retentionszeit'] = None
        data['Halbwertsbreite'] = None

    # Extrahiere die Flussrate (z.B. "Fluss 1,000 mL/min")
    match_fluss = re.search(r'Fluss\s+([0-9,]+)\s*mL/min', text)
    if match_fluss:
        flow_str = match_fluss.group(1).replace(',', '.')
        data['Flussrate'] = float(flow_str)
    else:
        data['Flussrate'] = None

    return data

def read_data():
    root = tk.Tk()
    root.withdraw()

    data_dict = {'mit_saeule': [], 'ohne_saeule': []}

    # PDFs MIT Säule
    mit_saeule_files = filedialog.askopenfilenames(
        title="Wählen Sie PDF-Dateien MIT Säule aus", 
        filetypes=[("PDF Dateien", "*.pdf")]
    )
    for file in mit_saeule_files:
        data = parse_pdf(file)
        data_dict['mit_saeule'].append(data)
    
    # PDFs OHNE Säule (optional)
    ohne_saeule_files = filedialog.askopenfilenames(
        title="Wählen Sie optional PDF-Dateien OHNE Säule aus", 
        filetypes=[("PDF Dateien", "*.pdf")]
    )
    for file in ohne_saeule_files:
        data = parse_pdf(file)
        data_dict['ohne_saeule'].append(data)
    
    print("Ausgelesene Daten:")
    print(data_dict)
    return data_dict

def van_deemter_model(u, A, B, C):
    return A + B/u + C*u

def van_deemter_analysis(data_dict, L=150):
    """
    Berechnet die Bodenhöhe H gemäß:
    
    \[
    H = \frac{L\,\Big((w_{1/2,\text{obs}})^2 - (w_{1/2,\text{ext}})^2\Big)}{5.55\,(t_{R,\text{obs}}-t_{R,\text{ext}})^2}
    \]
    
    Falls keine Messungen OHNE Säule vorliegen, werden \( t_{R,\text{ext}} \) und \( w_{1/2,\text{ext}} \) auf 0 gesetzt.
    
    L: Säulenlänge in \(\SI{}{\mm}\).
    """
    flow_rates = []
    H_values = []
    
    mit_list = data_dict['mit_saeule']
    ohne_list = data_dict['ohne_saeule']
    
    # Prüfe, ob Messungen ohne Säule vorliegen
    use_external = len(ohne_list) > 0

    for i, m in enumerate(mit_list):
        # Für Messungen OHNE Säule: Entweder den entsprechenden Wert verwenden oder 0 einsetzen
        if use_external:
            if i < len(ohne_list):
                o = ohne_list[i]
                t_ext = o['Retentionszeit'] if o['Retentionszeit'] is not None else 0
                w_ext = o['Halbwertsbreite'] if o['Halbwertsbreite'] is not None else 0
            else:
                t_ext = 0
                w_ext = 0
        else:
            t_ext = 0
            w_ext = 0

        if m['Retentionszeit'] is None or m['Halbwertsbreite'] is None or m['Flussrate'] is None:
            continue
        
        t_obs = m['Retentionszeit']
        w_obs = m['Halbwertsbreite']
        u = m['Flussrate']
        
        # Verhindere Division durch 0
        if (t_obs - t_ext) == 0:
            continue
        
        H = L * ((w_obs)**2 - (w_ext)**2) / (5.55 * (t_obs - t_ext)**2)
        flow_rates.append(u)
        H_values.append(H)
    
    print("Flussraten (u):", flow_rates)
    print("Bodenhöhen (H):", H_values)
    
    # Fit der Van-Deemter-Gleichung: H = A + B/u + C*u
    u_data = np.array(flow_rates)
    H_data = np.array(H_values)
    
    popt, pcov = curve_fit(van_deemter_model, u_data, H_data)
    perr = np.sqrt(np.diag(pcov))
    
    # Berechne R^2
    H_fit = van_deemter_model(u_data, *popt)
    residuals = H_data - H_fit
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((H_data - np.mean(H_data))**2)
    r_squared = 1 - (ss_res / ss_tot)
    
    print("Fitted coefficients:")
    print("A = {:.4f} ± {:.4f}".format(popt[0], perr[0]))
    print("B = {:.4f} ± {:.4f}".format(popt[1], perr[1]))
    print("C = {:.4f} ± {:.4f}".format(popt[2], perr[2]))
    print("Bestimmtheitsmaß $R^2$ =", r_squared)
    
    # Plot: Datenpunkte als Scatterplot und Fit-Kurve (als glatte Linie)
    plt.figure(figsize=(8,5))
    plt.scatter(u_data, H_data, label=r'Datenpunkte', color='blue')
    
    # Erstelle eine glatte Kurve für den Fit
    u_fit = np.linspace(np.min(u_data), np.max(u_data), 100)
    H_fit_line = van_deemter_model(u_fit, *popt)
    plt.plot(u_fit, H_fit_line, 'r-', label=r'Fit: $H = A + \frac{B}{u} + C\,u$')
    
    plt.xlabel(r'Flussrate $u$ / $\left( \SI{}{\milli \liter \per \minute} \right)$')
    plt.ylabel(r'Bodenh\"ohe $H$ / $\left( \SI{}{\milli \meter} \right)$')
    plt.legend()
    plt.grid(True)
    
    # Speichere den Plot als PDF
    plt.savefig("vD/van_deemter_plot.pdf", format="pdf")
    plt.show()

def main():
    data_dict = read_data()
    van_deemter_analysis(data_dict, L=150)

if __name__ == "__main__":
    main()
