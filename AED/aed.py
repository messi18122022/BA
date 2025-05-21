import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# --- EINSTELLUNGEN ---
qm_true = 1.5  # maximale Adsorptionskapazität
K_true = 0.8   # Gleichgewichtskonstante
K0 = 1e3  # Vorfaktor für K(E)
E_min = 10000  # minimale Adsorptionsenergie in J/mol
E_max = 50000  # maximale Adsorptionsenergie in J/mol
E_points = 200  # Anzahl der Energiepunkte
max_iter = 500
skalieren_mit_qm = True

np.random.seed(0)

# Langmuir-Isotherm-Modell
def langmuir(C, qm, K):
    return (qm * K * C) / (1 + K * C)

C_data = np.linspace(0.01, 5, 50)  # Konzentrationen
q_data = langmuir(C_data, qm_true, K_true)

# Fit ausführen
popt, pcov = curve_fit(langmuir, C_data, q_data)
qm_fit, K_fit = popt

print(f"Gefittete Parameter: qm = {qm_fit:.4f}, K = {K_fit:.4f}")

# Plot
C_fit = np.linspace(0.01, 5, 200)
q_fit = langmuir(C_fit, qm_fit, K_fit)

plt.figure(figsize=(8, 5))
plt.plot(C_data, q_data, 'o', label='synthetische Daten')
plt.plot(C_fit, q_fit, '-', label='Langmuir-Fit')
plt.xlabel('Konzentration C')
plt.ylabel('Beladung q')
plt.title('Langmuir-Fit an perfekte Daten')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("AED/langmuir_fit.pdf")
plt.show()

# --- AED-Berechnung mit EM-Verfahren ---

# Konstanten
R = 8.314  # J/mol·K
T = 298.15  # Temperatur in Kelvin

# Energie-Bereich (in J/mol)
E_vals = np.linspace(E_min, E_max, E_points)  # z.B. 10–50 kJ/mol
K_vals = K0 * np.exp(-E_vals / (R * T))  # K(E)

# Anfangsverteilung (gleich verteilt)
f_E = np.ones_like(E_vals)
f_E /= np.sum(f_E)

# EM-Iterationen
for iteration in range(max_iter):
    q_calc = []
    for C in C_data:
        theta = (K_vals * C) / (1 + K_vals * C)
        q_C = np.sum(f_E * theta)
        q_calc.append(q_C)
    q_calc = np.array(q_calc)

    ratio = q_data / (q_calc + 1e-12)  # Vermeide Division durch 0

    for j in range(len(E_vals)):
        num = 0
        for i, C in enumerate(C_data):
            theta = (K_vals[j] * C) / (1 + K_vals[j] * C)
            num += ratio[i] * theta
        f_E[j] *= num
    f_E /= np.sum(f_E)  # Normalisierung

if skalieren_mit_qm:
    f_E *= qm_true

# Plot der Adsorptionsenergieverteilung
plt.figure(figsize=(8, 5))
plt.plot(E_vals / 1000, f_E, label='AED (f(E))')  # in kJ/mol
plt.xlabel('Adsorptionsenergie E [kJ/mol]')
plt.ylabel('f(E)')
plt.title('Adsorptionsenergieverteilung (AED) via EM')
plt.grid(True)
plt.tight_layout()
plt.savefig("AED/aed_em_result.pdf")
plt.show()

# Vergleich: gemessene vs. rekonstruierte Isotherme
plt.figure(figsize=(8, 5))
plt.plot(C_data, q_data, 'o', label='Messdaten')
plt.plot(C_data, q_calc, '-', label='EM-Rekonstruktion')
plt.xlabel('Konzentration C')
plt.ylabel('Beladung q')
plt.title('Vergleich: Messdaten vs. EM-Rekonstruktion')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("AED/em_reconstruction.pdf")
plt.show()
