import numpy as np  # Importiere die Bibliothek NumPy, die leistungsfähige Funktionen für mathematische Operationen und die Arbeit mit Arrays bereitstellt
import matplotlib.pyplot as plt  # Importiere Matplotlib zum Erstellen von Diagrammen und Visualisierungen
from scipy.optimize import curve_fit  # Importiere curve_fit aus SciPy, um Funktionen an Daten anzupassen (Curve Fitting)

# --- EINSTELLUNGEN ---
# qm_true: Die maximale Adsorptionskapazität. Das bedeutet, wie viele Moleküle maximal an der Oberfläche haften können.
qm_true = 1.5  # Setze den Wert von qm_true auf 1.5 (beliebige Einheiten)

# K_true: Die Gleichgewichtskonstante für den Adsorptionsprozess. Sie beschreibt, wie stark die Moleküle an der Oberfläche haften.
K_true = 0.8   # Setze den Wert von K_true auf 0.8 (beliebige Einheiten)

K0 = 1e3  # Setze den Wert von K0 auf 1000 (1e3), Basis-Gleichgewichtskonstante für Energiebezug

# --- Konstanten ---
R = 8.314  # Universelle Gaskonstante in Joule pro Mol Kelvin (J/(mol·K))
T = 298.15  # Temperatur in Kelvin (Raumtemperatur, ca. 25°C)

# Erzeuge synthetische Konzentrationsdaten von 0.01 bis 5 in 50 Schritten
# Dies simuliert experimentelle Messwerte, allerdings ohne Rauschen, um das Modell zu testen.
C_data = np.linspace(0.01, 5, 50)  # Erstelle ein Array mit 50 Konzentrationswerten zwischen 0.01 und 5

# Berechne die Beladung q für die synthetischen Konzentrationswerte mit den "wahren" Parametern qm_true und K_true
def langmuir(C, qm, K):
    """
    Berechnet die Beladung q nach dem Langmuir-Modell.
    Formel: q = (qm * K * C) / (1 + K * C)
    qm ist die maximale Kapazität,
    K ist die Gleichgewichtskonstante,
    C ist die Konzentration.
    """
    # Berechnung der Beladung q für gegebene Konzentration C
    return (qm * K * C) / (1 + K * C)

q_data = langmuir(C_data, qm_true, K_true)  # Berechne q für jede Konzentration C_data

# --- Hinweis aus der Literatur ---
# Laut Gritti & Guiochon (2005) sollte der Energiebereich (E_min bis E_max) aus dem Konzentrationsbereich gewählt werden.
# Dazu werden K_min = 1 / C_max und K_max = 1 / C_min gesetzt.
# Dann ergibt sich E = -RT * ln(K / K0), sodass:
#   E_min = -RT * ln(K_max / K0)
#   E_max = -RT * ln(K_min / K0)
# Diese Beziehung stellt sicher, dass die betrachteten Energien genau den Bereich abdecken,
# den die experimentellen Konzentrationen informativ beeinflussen.
# Zudem ist laut der Literatur die absolute Wahl von K0 nicht kritisch für die Form von f(E),
# solange sie konstant bleibt – es beeinflusst nur die absolute Lage der Energieachse.

C_min = np.min(C_data)  # minimale Konzentration im Datensatz
C_max = np.max(C_data)  # maximale Konzentration im Datensatz

K_min = 1 / C_max  # kleinste Gleichgewichtskonstante entspricht der größten Konzentration
K_max = 1 / C_min  # größte Gleichgewichtskonstante entspricht der kleinsten Konzentration

# Berechnung der minimalen und maximalen Adsorptionsenergie in Joule pro Mol
E_min = -R * T * np.log(K_max / K0)  # minimale Energie, entspricht stärkster Bindung
E_max = -R * T * np.log(K_min / K0)  # maximale Energie, entspricht schwächster Bindung

# E_points: Die Anzahl der diskreten Punkte, die wir verwenden, um den Energie-Bereich in unserem Modell abzubilden.
E_points = 200  # Wir teilen den Bereich zwischen E_min und E_max in 200 gleich große Abschnitte auf

# max_iter: Maximale Anzahl der Iterationen für das EM-Verfahren, das zur Schätzung der Energieverteilung verwendet wird.
max_iter = 500  # Wir wiederholen den Schätzprozess bis zu 500 Mal, um eine gute Lösung zu finden

# skalieren_mit_qm: Ein Wahrheitswert, der angibt, ob die gefundene Verteilung mit der maximalen Adsorptionskapazität skaliert werden soll.
skalieren_mit_qm = True  # Wenn True, wird die Verteilung so angepasst, dass sie die tatsächliche Kapazität widerspiegelt

# Setze den Zufallszahlengenerator auf einen festen Wert, damit die Ergebnisse immer gleich sind (Reproduzierbarkeit).
np.random.seed(0)

# --- Definition des Langmuir-Isotherm-Modells ---
# Funktion zur Berechnung der Beladung q in Abhängigkeit von Konzentration C, qm und K
def langmuir(C, qm, K):
    """
    Berechnet die Beladung q nach dem Langmuir-Modell.
    Formel: q = (qm * K * C) / (1 + K * C)
    qm ist die maximale Kapazität,
    K ist die Gleichgewichtskonstante,
    C ist die Konzentration.
    """
    # Berechnung der Beladung q für gegebene Konzentration C
    return (qm * K * C) / (1 + K * C)

# --- Fit der Langmuir-Isotherme an die synthetischen Daten ---
# curve_fit versucht, die Funktion langmuir so anzupassen, dass sie die Datenpunkte (C_data, q_data) bestmöglich beschreibt
popt, pcov = curve_fit(langmuir, C_data, q_data)  # popt sind die geschätzten Parameter, pcov die Kovarianzmatrix
qm_fit, K_fit = popt  # Extrahiere die geschätzten Parameter qm und K aus popt

# Ausgabe der gefitteten Parameter zur Kontrolle, mit 4 Dezimalstellen Genauigkeit
print(f"Gefittete Parameter: qm = {qm_fit:.4f}, K = {K_fit:.4f}")

# Erzeuge feinere Konzentrationswerte für die Visualisierung des Fits (200 Punkte zwischen 0.01 und 5)
C_fit = np.linspace(0.01, 5, 200)

# Berechne die Beladung q mit den gefitteten Parametern für die Visualisierung
q_fit = langmuir(C_fit, qm_fit, K_fit)

# --- Plot der synthetischen Daten und des Langmuir-Fits ---
plt.figure(figsize=(8, 5))  # Erzeuge eine neue Abbildung mit der Größe 8x5 Zoll
plt.plot(C_data, q_data, 'o', label='synthetische Daten')  # Zeichne die Originaldaten als Punkte
plt.plot(C_fit, q_fit, '-', label='Langmuir-Fit')  # Zeichne die gefittete Kurve als Linie
plt.xlabel('Konzentration C')  # Beschrifte die x-Achse mit "Konzentration C"
plt.ylabel('Beladung q')       # Beschrifte die y-Achse mit "Beladung q"
plt.title('Langmuir-Fit an perfekte Daten')  # Titel des Plots
plt.legend()  # Zeige eine Legende mit den Beschriftungen der Linien/Punkte an
plt.grid(True)  # Aktiviere Gitterlinien im Plot für bessere Lesbarkeit
plt.tight_layout()  # Optimiere das Layout, damit keine Elemente abgeschnitten werden
plt.savefig("AED/langmuir_fit.pdf")  # Speichere den Plot als PDF-Datei im Ordner "AED"

# --- AED-Berechnung mit EM-Verfahren ---
# Ziel: Wir wollen die Verteilung der Adsorptionsenergien f(E) herausfinden, basierend auf den Messdaten q_data.
# Das EM-Verfahren (Expectation-Maximization) hilft uns, diese Verteilung schrittweise zu schätzen.

# --- Berechnung des Adsorptionsenergie-Bereichs und der Gleichgewichtskonstanten K(E) ---
# Erzeuge ein Array von Adsorptionsenergien zwischen E_min und E_max, aufgeteilt in E_points Punkte
E_vals = np.linspace(E_min, E_max, E_points)  # Array mit 200 Werten zwischen 5704 und 21430 J/mol

# Berechne die Gleichgewichtskonstante K für jede Energie E mit einer Arrhenius-ähnlichen Beziehung:
# K(E) = K0 * exp(-E / (R * T))
# Diese Formel sagt: Je höher die Energie E, desto kleiner ist K(E), also die Affinität.
K_vals = K0 * np.exp(-E_vals / (R * T))  # Array mit K-Werten für jede Energie

# --- Initialisierung der Verteilung f(E) ---
# Wir starten mit einer Gleichverteilung, da wir keine Vorinformation über die wahre Verteilung haben.
f_E = np.ones_like(E_vals)  # Erstelle ein Array mit Einsen, gleiche Länge wie E_vals

# Normalisiere f_E so, dass die Summe aller Werte 1 ergibt. Damit interpretieren wir f_E als Wahrscheinlichkeitsverteilung.
f_E /= np.sum(f_E)  # Division jedes Elements durch die Summe aller Elemente

# --- EM-Algorithmus zur Schätzung der Verteilung f(E) ---
# Wir wiederholen den Prozess max_iter-mal, um die Verteilung schrittweise zu verbessern.
for iteration in range(max_iter):
    q_calc = []  # Leere Liste, um die berechneten Beladungen für jede Konzentration zu speichern

    # Berechne für jede Konzentration C die erwartete Beladung q_C basierend auf der aktuellen Verteilung f_E
    for C in C_data:
        # Berechne die Besetzungsfraktion theta für jede Energie E: theta = (K(E)*C) / (1 + K(E)*C)
        # Theta gibt an, wie viel von der Oberfläche bei Energie E besetzt ist.
        theta = (K_vals * C) / (1 + K_vals * C)
        # Die Gesamtbeladung q_C ist das gewichtete Mittel von theta über alle Energien, gewichtet mit f_E
        q_C = np.sum(f_E * theta)
        q_calc.append(q_C)  # Speichere die berechnete Beladung

    q_calc = np.array(q_calc)  # Wandle die Liste in ein NumPy-Array um, um weitere Berechnungen zu ermöglichen

    # Berechne das Verhältnis der gemessenen Beladungen q_data zu den berechneten Beladungen q_calc
    # Dieses Verhältnis zeigt, wie gut die aktuelle Verteilung f_E die Messdaten beschreibt.
    # Werte >1 zeigen Unterabschätzung, Werte <1 Überabschätzung.
    ratio = q_data / (q_calc + 1e-12)  # 1e-12 wird hinzugefügt, um Division durch Null zu vermeiden

    # Aktualisiere die Verteilung f_E für jede Energie E
    for j in range(len(E_vals)):
        num = 0  # Numerator für die Aktualisierung von f_E[j]
        for i, C in enumerate(C_data):
            # Berechne erneut theta für die j-te Energie und i-te Konzentration
            theta = (K_vals[j] * C) / (1 + K_vals[j] * C)
            # Summe über alle Konzentrationen von ratio[i] * theta
            num += ratio[i] * theta
        # Multipliziere f_E[j] mit dieser Summe, um die Verteilung anzupassen
        f_E[j] *= num

    # Normalisiere die Verteilung f_E so, dass die Summe 1 bleibt (damit es wieder eine Wahrscheinlichkeitsverteilung ist)
    f_E /= np.sum(f_E)

# Optional: Skaliere die Verteilung mit der maximalen Adsorptionskapazität qm_true,
# um die absolute Menge der adsorbierten Moleküle darzustellen.
if skalieren_mit_qm:
    f_E *= qm_true  # Multipliziere jedes Element von f_E mit qm_true

# --- Plot der Adsorptionsenergieverteilung ---
plt.figure(figsize=(8, 5))  # Neue Abbildung mit Größe 8x5 Zoll
# Trage die Energie in kJ/mol auf der x-Achse auf (E_vals in J/mol geteilt durch 1000)
plt.plot(E_vals / 1000, f_E, label='AED (f(E))')  # Zeichne die Verteilung f(E)
plt.xlabel('Adsorptionsenergie E [kJ/mol]')  # Beschrifte die x-Achse
plt.ylabel('f(E)')  # Beschrifte die y-Achse
plt.title('Adsorptionsenergieverteilung (AED) via EM')  # Titel des Plots
plt.grid(True)  # Aktiviere Gitterlinien
plt.tight_layout()  # Optimiere Layout
plt.savefig("AED/aed_em_result.pdf")  # Speichere den Plot als PDF-Datei
plt.show()  # Zeige den Plot an

# --- Vergleich: gemessene vs. rekonstruierte Isotherme ---
plt.figure(figsize=(8, 5))  # Neue Abbildung für den Vergleich
plt.plot(C_data, q_data, 'o', label='Messdaten')  # Zeichne die Original-Messdaten als Punkte
plt.plot(C_data, q_calc, '-', label='EM-Rekonstruktion')  # Zeichne die rekonstruierte Beladung als Linie
plt.xlabel('Konzentration C')  # Beschrifte die x-Achse
plt.ylabel('Beladung q')       # Beschrifte die y-Achse
plt.title('Vergleich: Messdaten vs. EM-Rekonstruktion')  # Titel des Plots
plt.legend()  # Zeige Legende
plt.grid(True)  # Aktiviere Gitterlinien
plt.tight_layout()  # Optimiere Layout
plt.savefig("AED/em_reconstruction.pdf")  # Speichere den Plot als PDF-Datei
plt.show()  # Zeige den Plot an
