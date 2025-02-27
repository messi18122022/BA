#!/usr/bin/env python3
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pdfplumber
from tkinter import Tk, filedialog

# Matplotlib LaTeX-Konfiguration: Alles wird in LaTeX gerendert und die Pakete siunitx und amsmath geladen.
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
})
plt.rcParams['text.latex.preamble'] = r'\usepackage{siunitx}\usepackage{amsmath}'

# Benutzerdefinierte Spaltenlänge (z.B. in mm) – bitte anpassen!
COLUMN_LENGTH = 100.0

def extract_data_from_pdf(pdf_path):
    """
    Extrahiert aus dem PDF-Report:
      - Den Fluss (z. B. "Fluss 0.100 mL/min")
      - Die Trennstufenzahl (N) aus der Zeile des Cl-Peaks.
    Es werden ausschließlich Daten des Cl-Peaks berücksichtigt.
    """
    with pdfplumber.open(pdf_path) as pdf:
        text = ""
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    
    # Suche nach dem Fluss (z. B. "Fluss 0.100 mL/min")
    flow_match = re.search(r'Fluss\s*([\d.,]+)\s*mL/min', text)
    if flow_match:
        flow_str = flow_match.group(1).replace(',', '.')
        flow = float(flow_str)
    else:
        flow = None

    # Suche nach der Zeile, die mit "Cl" beginnt und die Trennstufenzahl enthält
    cl_match = re.search(r'\bCl\b\s+([\d.,]+)\s+([\d.,]+)\s+(\d+)', text)
    if cl_match:
        n_str = cl_match.group(3).replace(',', '.')
        n = int(n_str)
    else:
        n = None

    return flow, n

def van_deemter(u, A, B, C):
    """
    Van-Deemter-Gleichung: H = A + B/u + C*u
    u: Flussrate (mL/min)
    """
    return A + B/u + C*u

def main():
    # Öffne ein Dialogfenster zur Dateiauswahl
    root = Tk()
    root.withdraw()  # Hauptfenster ausblenden
    root.attributes("-topmost", True)  # Dialogfenster in den Vordergrund bringen
    file_paths = filedialog.askopenfilenames(
        title="Wähle PDF-Dateien aus",
        filetypes=[("PDF Files", "*.pdf")])
    root.destroy()  # Schließe das Tkinter-Fenster
    
    if not file_paths:
        print("Keine Dateien ausgewählt.")
        return

    flows = []
    n_values = []
    
    for pdf_file in file_paths:
        flow, n = extract_data_from_pdf(pdf_file)
        if flow is not None and n is not None:
            flows.append(flow)
            n_values.append(n)
            print(pdf_file,n)
        else:
            print(f"{pdf_file}: Daten konnten nicht extrahiert werden.")

    if len(flows) == 0:
        print("Keine gültigen Daten gefunden.")
        return

    # Umwandeln in NumPy-Arrays
    flows = np.array(flows)
    n_values = np.array(n_values)
    
    # Berechnung des HETP (H) als Spaltenlänge geteilt durch die Trennstufenzahl
    H_values = COLUMN_LENGTH / n_values

    # Sortiere die Daten nach Flussrate
    sort_index = np.argsort(flows)
    flows = flows[sort_index]
    H_values = H_values[sort_index]
    
    # Führe den Fit der Van-Deemter-Gleichung durch: H = A + B/u + C*u
    popt, pcov = curve_fit(van_deemter, flows, H_values)
    A_fit, B_fit, C_fit = popt
    perr = np.sqrt(np.diag(pcov))  # Standardabweichungen der Parameter
    
    # Berechnung des R²-Werts
    residuals = H_values - van_deemter(flows, *popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((H_values - np.mean(H_values))**2)
    r_squared = 1 - (ss_res / ss_tot)
    
    print("Fittete Parameter:")
    print(f"A = {A_fit} ± {perr[0]}")
    print(f"B = {B_fit} ± {perr[1]}")
    print(f"C = {C_fit} ± {perr[2]}")
    print(f"R² = {r_squared}")
    
    # Erzeuge einen feinen Flussratenbereich für den Plot
    u_fit = np.linspace(np.min(flows), np.max(flows), 100)
    H_fit = van_deemter(u_fit, A_fit, B_fit, C_fit)
    
    # Erstelle den Plot (alle Texte im Plot nutzen LaTeX-Syntax)
    plt.figure(figsize=(8,5))
    plt.scatter(flows, H_values, color='red', label=r"Gemessene Daten")
    plt.plot(u_fit, H_fit, label=r"Van-Deemter Fit")
    
    # Verwende \si{}{} mit skalierenden Klammern
    plt.xlabel(r"Flussrate $\dot{V}$ / $\left(\si{\milli\liter\per\minute}\right)$", fontsize=12)
    plt.ylabel(r"HETP $H$ / $\left(\si{\milli\meter}\right)$", fontsize=12)
    # Legende mit komplett transparenter Legendenbox
    plt.legend(framealpha=1)
    plt.grid(True)
    
    # Speichere den Plot als PDF
    plt.savefig("van_deemter_plot.pdf", format="pdf")

if __name__ == '__main__':
    main()
