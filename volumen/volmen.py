import tkinter as tk
from tkinter import filedialog, simpledialog
import os

def dateien_verarbeiten():
    """
    Öffnet Dialogfenster zur Auswahl von Dateien, Multiplikatoren und Speicherort,
    und führt die Multiplikation durch.
    """

    # 1. Dateien auswählen
    root = tk.Tk()
    root.withdraw()  # Hauptfenster von tkinter ausblenden
    dateipfade = filedialog.askopenfilenames(title="Wähle die Textdateien aus")

    # 2. Multiplikatoren abfragen
    multiplikatoren = {}
    for dateipfad in dateipfade:
        dateiname = os.path.basename(dateipfad)
        multiplikator = simpledialog.askfloat(title="Multiplikator",
                                             prompt=f"Gib den Multiplikator für {dateiname} ein:")
        if multiplikator is not None:  # Überprüfen, ob der Benutzer nicht auf Abbrechen geklickt hat
            multiplikatoren[dateipfad] = multiplikator
        else:
            print("Vorgang abgebrochen.")
            return

    # 3. Speicherort wählen
    speicherordner = filedialog.askdirectory(title="Wähle den Speicherort für die neuen Dateien")

    # 4. Verarbeitung der Dateien
    for dateipfad, multiplikator in multiplikatoren.items():
        try:
            neue_daten = []
            with open(dateipfad, 'r') as datei:
                for zeile in datei:
                    zeile = zeile.strip()  # Entfernt Leerzeichen am Anfang und Ende
                    if zeile:  # Verarbeitet nur nicht-leere Zeilen
                        try:
                            wert1, wert2 = zeile.split(',')
                            neuer_wert = float(wert1) * multiplikator
                            neue_daten.append(f"{neuer_wert},{wert2}")
                        except ValueError:
                            print(f"Zeile übersprungen: {zeile} in Datei: {dateipfad}")

            # 5. Speichern der neuen Dateien
            dateiname = os.path.basename(dateipfad)
            name_ohne_endung, endung = os.path.splitext(dateiname)
            neuer_dateipfad = os.path.join(speicherordner, f"{name_ohne_endung}_bearbeitet{endung}")

            with open(neuer_dateipfad, 'w') as neue_datei:
                neue_datei.write("Multiplizierte Spalte,Originale Spalte\n")  # Überschriftenzeile
                neue_datei.write('\n'.join(neue_daten))  # Schreibt alle neuen Daten zeilenweise

            print(f"Datei erstellt: {neuer_dateipfad}")

        except Exception as e:
            print(f"Fehler beim Verarbeiten von {dateipfad}: {e}")

if __name__ == "__main__":
    dateien_verarbeiten()