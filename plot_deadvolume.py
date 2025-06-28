#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
# LaTeX aktivieren und Schriftart setzen
import numpy as np
import matplotlib as mpl
mpl.rc('text', usetex=True)
mpl.rc('font', family='serif')
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}\usepackage{siunitx}'

# Daten: Volumenstrom (mL/min) vs. Totvolumen (mL)
flowrate = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
            1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
dead_volume = [0.786, 0.771, 0.773, 0.765, 0.768, 0.764, 0.737, 0.742,
               0.746, 0.747, 0.745, 0.733, 0.744, 0.756, 0.750, 0.734,
               0.759, 0.817, 0.788, 0.805]

plt.figure(figsize=(16/2.54, 8/2.54))
plt.plot(flowrate, dead_volume, marker='o', linestyle='None', label='Messwerte')
# Mittelwert berechnen
mean_dead = np.mean(dead_volume)
# Mittelwert als rote gestrichelte Linie hinzufuegen
plt.axhline(mean_dead, color='red', linestyle='--', label=rf"Mittelwert = ${mean_dead:.3f}\,\mathrm{{mL}}$")
plt.legend()
plt.xticks([0.1, 0.5, 1.0, 1.5, 2.0])
plt.xlabel(r'Flussrate $\dot{V}$ / \si{\milli\liter\per\minute}')
plt.ylabel(r'Totvolumen $V^\mathrm{dead}_\mathrm{col}$ / \si{\milli\liter}')
plt.grid(True)
plt.tight_layout()
plt.savefig('plot_deadvolume.pdf', format='pdf', bbox_inches='tight')
plt.show()