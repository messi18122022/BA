import tkinter as tk
from tkinter import filedialog
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def main():
    # Tkinter initialisieren und Hauptfenster verstecken
    root = tk.Tk()
    root.withdraw()

    # Dialog zum Auswählen mehrerer TXT-Dateien öffnen
    file_paths = filedialog.askopenfilenames(
        title="TXT Dateien auswählen", 
        filetypes=[("TXT Dateien", "*.txt")]
    )

    if not file_paths:
        print("Keine Dateien ausgewählt.")
        return

    num_files = len(file_paths)
    
    # Erstellen eines Farbverlaufs von Blau nach Rot
    cmap = LinearSegmentedColormap.from_list("bluetored", ["blue", "red"], N=num_files)

    plt.figure(figsize=(10, 6))
    
    # Jede Datei einlesen und plotten
    for idx, file in enumerate(file_paths):
        # Daten aus der TXT-Datei einlesen, dabei die Kopfzeile überspringen
        data = np.loadtxt(file, delimiter=",", skiprows=1)
        x = data[:, 0]  # Multiplizierte Spalte
        y = data[:, 1]  # Originale Spalte

        # Plotten der Daten mit einer Farbe aus dem erstellten Farbverlauf
        plt.plot(x, y, color=cmap(idx), label=f"Datei {idx+1}")

    plt.xlabel("Retentionsvolumen V_R / mL")
    plt.ylabel("Leitfähigkeit σ / μS/cm")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
