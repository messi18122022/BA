import numpy as np
import matplotlib.pyplot as plt
import glob
import re
from PyPDF2 import PdfReader
from tkinter import Tk
from tkinter.filedialog import askopenfilenames

# LaTeX aktivieren und Computer Modern als Schriftart verwenden
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams['font.serif'] = ['Computer Modern Roman']
plt.rcParams['text.latex.preamble'] = r'\usepackage[version=4]{mhchem}'

# Analyten: Verwende LaTeX-Syntax für schöne Formatierung
analyten = [r"$\ce{Cl-}$", r"$\ce{NO2-}$", r"$\ce{PO4^{3-}}$", r"$\ce{Br-}$", r"$\ce{NO3-}$", r"$\ce{SO4^{2-}}$"]
x = np.arange(len(analyten))
width = 0.35  # Breite der Balken


# Daten: Eigene Messwerte (Mittelwerte und Standardabweichungen) und Hersteller-Referenzwerte

# Wähle PDF-Berichte über Dialogfenster
root = Tk()
root.withdraw()
pdf_files = askopenfilenames(title="Wähle PDF-Berichte", filetypes=[("PDF files", "*.pdf")])
pdf_files = list(pdf_files)

ret_list, plates_list, asy_list, reso_list = [], [], [], []

for pdf_file in pdf_files:
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    lines = text.splitlines()
    # Finde Tabellenkopf
    for idx, line in enumerate(lines):
        if line.startswith("Komponentenname"):
            header_idx = idx
            break
    # Sammle die nächsten 7 nicht-leeren Zeilen (IP + 6 Analyte)
    data_lines = []
    i = header_idx + 2
    while len(data_lines) < 7 and i < len(lines):
        if lines[i].strip():
            data_lines.append(lines[i].strip())
        i += 1
    # Überspringe IP
    analyte_lines = data_lines[1:]
    def parse_val(val):
        val = val.replace(",", ".")
        return float(val) if re.match(r"^[0-9.]+$", val) else np.nan
    ret_row, plates_row, asy_row, reso_row = [], [], [], []
    for parts in (line.split() for line in analyte_lines):
        ret_row.append(parse_val(parts[1]))
        plates_row.append(parse_val(parts[2]))
        asy_row.append(parse_val(parts[3]))
        reso_row.append(parse_val(parts[4]))
    ret_list.append(ret_row)
    plates_list.append(plates_row)
    asy_list.append(asy_row)
    reso_list.append(reso_row)

# In numpy-Arrays umwandeln und Mittelwerte/Std-Abweichungen berechnen
ret_array = np.array(ret_list)
plates_array = np.array(plates_list)
asy_array = np.array(asy_list)
reso_array = np.array(reso_list)

ret_measured = np.nanmean(ret_array, axis=0)
ret_std = np.nanstd(ret_array, axis=0, ddof=1)
plates_measured = np.nanmean(plates_array, axis=0)
plates_std = np.nanstd(plates_array, axis=0, ddof=1)
asy_measured = np.nanmean(asy_array, axis=0)
asy_std = np.nanstd(asy_array, axis=0, ddof=1)
reso_measured = np.nanmean(reso_array, axis=0)
reso_std = np.nanstd(reso_array, axis=0, ddof=1)

# Hersteller-Referenzwerte
ret_ref = np.array([2.97, 3.87, 6.15, 8.59, 10.56, 15.14])
plates_ref = np.array([4470, 4214, 5863, 3475, 2224, 6478])
asy_ref = np.array([1.35, 1.34, 1.21, 1.49, 1.91, 1.01])
reso_ref = np.array([4.29, 8.18, 5.39, 2.66, 5.56, np.nan])

# Funktion zur Erstellung eines Balkendiagramms für einen Parameter:
def plot_bar(ax, measured, std, ref, ylabel, title):
    # Eigene Messwerte als Balken mit Errorbars
    bars_meas = ax.bar(x - width/2, measured, width, yerr=std, capsize=5, 
                       label="Messwert ($\\bar{x}$ ± $s$)", color='blue')
    # Hersteller-Referenzwerte als Balken (NaN-Werte werden nicht gezeichnet)
    bars_ref = ax.bar(x + width/2, ref, width, label="Referenz", color='red')
    
    ax.set_xticks(x)
    ax.set_xticklabels(analyten, rotation=45)
    ax.set_ylabel(ylabel)
    ax.legend()

# Plot für Retentionszeit
fig, ax = plt.subplots(figsize=(2.8,2.8))
plot_bar(ax, ret_measured, ret_std, ret_ref, "Retentionszeit $t_R$ / min", "Retentionszeit")
ax.set_ylim(0, 22)
plt.tight_layout()
fig.savefig("freigabe_säulen/Retentionszeit.pdf", format="pdf")
plt.close(fig)

# Plot für Theoretische Plattenzahl (Trennstufenzahl)
fig, ax = plt.subplots(figsize=(2.8,2.8))
plot_bar(ax, plates_measured, plates_std, plates_ref, "Bodenzahl $N$ / -", "Trennstufenzahl")
ax.set_ylim(0, 10500)
plt.tight_layout()
fig.savefig("freigabe_säulen/Trennstufenzahl.pdf", format="pdf")
plt.close(fig)

# Plot für Asymmetrie
fig, ax = plt.subplots(figsize=(2.8,2.8))
plot_bar(ax, asy_measured, asy_std, asy_ref, "Asymmetrie", "Asymmetrie")
ax.set_ylim(0, 2.9)
plt.tight_layout()
fig.savefig("freigabe_säulen/Asymmetrie.pdf", format="pdf")
plt.close(fig)

# Plot für Auflösung
fig, ax = plt.subplots(figsize=(2.8,2.8))
plot_bar(ax, reso_measured, reso_std, reso_ref, "Auflösung $R$ / - ", "Auflösung")
ax.set_ylim(0, 17)
plt.tight_layout()
fig.savefig("freigabe_säulen/Aufloesung.pdf", format="pdf")
plt.close(fig)