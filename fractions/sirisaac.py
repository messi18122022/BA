import matplotlib.pyplot as plt
import matplotlib
# LaTeX-Unterstützung aktivieren
matplotlib.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'font.serif': ['Computer Modern'],
})

# Daten für Spülzeiten (t_rinse) und Konzentration der Fraktion (c_fraction)
t_rinse = [15, 25, 36, 48, 58, 70, 82, 93, 104, 115, 125, 135, 148, 161, 171, 1259]
c_fraction = [964920.00, 3661.98, 686.66, 431.34, 334.26, 265.96, 185.14, 150.06, 125.07, 101.12, 83.00, 88.56, 92.64, 87.06, 91.93, 16.38]

# Figur mit LaTeX-Einstellungen und spezifischer Größe (16 cm x 6.7 cm)
plt.figure(figsize=(16/2.54, 6.7/2.54))
plt.plot(t_rinse, c_fraction, marker='o', linestyle='None')
plt.xlabel(r'kumulierte Spülzeit $t_{\mathrm{rinse}}\,(\mathrm{min})$')
plt.ylabel(r'$c^{\mathrm{NO_3^-}}_{\mathrm{fraction}}\,(\mathrm{mg}\,\mathrm{L}^{-1})$')
plt.grid(True)
plt.ylim(0, 3700)
plt.tight_layout()
plt.show()
