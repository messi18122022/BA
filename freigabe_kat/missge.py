import numpy as np
import matplotlib.pyplot as plt

# LaTeX aktivieren und Computer Modern als Schriftart verwenden
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams['font.serif'] = ['Computer Modern Roman']
plt.rcParams['text.latex.preamble'] = r'\usepackage[version=4]{mhchem}'

# Analyten: Verwende LaTeX-Syntax für schöne Formatierung
analyten = [r"$\ce{Li+}$", r"$\ce{Na+}$", r"$\ce{NH4^{+}}$", r"$\ce{K+}$", r"$\ce{Ca^2+}$", r"$\ce{Mg^{2+}}$"]
x = np.arange(len(analyten))
width = 0.35  # Breite der Balken

# Daten: Eigene Messwerte (Mittelwerte und Standardabweichungen) und Hersteller-Referenzwerte

# Retentionszeit [min]
ret_measured = np.array([4.28, 6.42, 7.46, 11.352, 14.735, 20.195])
ret_std      = np.array([0.0, 0.0, 0.0, 0.0041, 0.005, 0.0105])
ret_ref      = np.array([4.64, 6.98, 8.11, 12.43, 15.53, 21.04])

# Theoretische Plattenzahl
plates_measured = np.array([8098, 9194.5, 10350, 8013.5, 4622, 4070])
plates_std      = np.array([28, 18, 17, 21, 21, 26])
plates_ref      = np.array([6434, 8129, 9321, 7794, 4720, 3907])

# Asymmetrie
asy_measured = np.array([0.943, 1.01, 1.117, 1.457, 1.553, 1.992])
asy_std      = np.array([0.005, 0.006, 0.005, 0.005, 0.010, 0.016])
asy_ref      = np.array([1.03, 1.08, 1.14, 1.42, 1.54, 2.14])

# Auflösung
reso_measured = np.array([9.362, 3.702, 9.720, 4.928, 5.117, np.nan])
reso_std      = np.array([0.012, 0.004, 0.006, 0.008, 0.016, np.nan])
reso_ref      = np.array([8.62, 3.51, 9.62, 4.22, 4.90, np.nan])

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
fig.savefig("freigabe_kat/Retentionszeit.pdf", format="pdf")
plt.close(fig)

# Plot für Theoretische Plattenzahl (Trennstufenzahl)
fig, ax = plt.subplots(figsize=(2.8,2.8))
plot_bar(ax, plates_measured, plates_std, plates_ref, "Bodenzahl $N$ / -", "Trennstufenzahl")
ax.set_ylim(0, 15000)
plt.tight_layout()
fig.savefig("freigabe_kat/Trennstufenzahl.pdf", format="pdf")
plt.close(fig)

# Plot für Asymmetrie
fig, ax = plt.subplots(figsize=(2.8,2.8))
plot_bar(ax, asy_measured, asy_std, asy_ref, "Asymmetrie", "Asymmetrie")
ax.set_ylim(0, 2.7)
plt.tight_layout()
fig.savefig("freigabe_kat/Asymmetrie.pdf", format="pdf")
plt.close(fig)

# Plot für Auflösung
fig, ax = plt.subplots(figsize=(2.8,2.8))
plot_bar(ax, reso_measured, reso_std, reso_ref, "Auflösung $R$ / - ", "Auflösung")
ax.set_ylim(0, 17)
plt.tight_layout()
fig.savefig("freigabe_kat/Aufloesung.pdf", format="pdf")
plt.close(fig)