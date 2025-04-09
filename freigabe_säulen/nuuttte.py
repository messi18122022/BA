import numpy as np
import matplotlib.pyplot as plt

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

# Retentionszeit [min]
ret_measured = np.array([2.830, 3.664, 6.398, 7.912, 9.647, 15.203])
ret_std = np.array([0.004, 0.006, 0.006, 0.022, 0.030, 0.041])
ret_ref = np.array([2.97, 3.87, 6.15, 8.59, 10.56, 15.14])

# Theoretische Plattenzahl (Trennstufenzahl)
plates_measured = np.array([6895, 5921, 7474, 3948, 2508, 6738])
plates_std = np.array([86, 40, 44, 33, 21, 38])
plates_ref = np.array([4470, 4214, 5863, 3475, 2224, 6478])

# Asymmetrie
asy_measured = np.array([1.107, 1.195, 1.087, 1.505, 1.877, 1.017])
asy_std = np.array([0.010, 0.008, 0.008, 0.011, 0.020, 0.005])
asy_ref = np.array([1.35, 1.34, 1.21, 1.49, 1.91, 1.01])

# Auflösung
reso_measured = np.array([5.11, 11.24, 3.787, 2.722, 7.352, np.nan])  # Sulfat: ungültig = np.nan
reso_std = np.array([0.02, 0.042, 0.024, 0.008, 0.031, 0])
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
ax.set_ylim(0, 2.7)
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