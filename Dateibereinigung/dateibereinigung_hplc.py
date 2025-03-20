import pandas as pd
import os
import glob

# Quell- und Zielordner festlegen
source_folder = "/Users/musamoin/Desktop/BA_Musa-Moin_FS25/Messungen/HPLC/csv_raw"       
destination_folder = "/Users/musamoin/Desktop/BA_Musa-Moin_FS25/Messungen/HPLC/csv_clean"     

# Zielordner anlegen, falls er nicht existiert
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

# Suche nach allen CSV-Dateien (unabhängig von Groß-/Kleinschreibung)
csv_files = glob.glob(os.path.join(source_folder, "*.csv")) + glob.glob(os.path.join(source_folder, "*.CSV"))

if not csv_files:
    print("Keine CSV-Dateien im Quellordner gefunden.")
else:
    for csv_file in csv_files:
        try:
            # Lese die CSV-Datei mit UTF‑16-Encoding ein (ohne expliziten Delimiter)
            df = pd.read_csv(csv_file, encoding="utf-16", engine="python")
            
            # Falls nur eine Spalte vorhanden ist, versuchen wir, diese anhand möglicher Trenner aufzuteilen
            if df.shape[1] == 1:
                possible_delimiters = [",", ";", "\t"]
                df_new = None
                for delim in possible_delimiters:
                    # Spalte aufteilen
                    df_try = df.iloc[:, 0].str.split(delim, expand=True)
                    # Falls durch diesen Trenner genau 2 Spalten entstehen, übernehmen wir das Ergebnis
                    if df_try.shape[1] == 2:
                        df_new = df_try
                        break
                if df_new is not None:
                    df = df_new
                else:
                    print(f"Datei {csv_file} konnte nicht in 2 Spalten aufgeteilt werden. Datei wird übersprungen.")
                    continue

            # Falls nach allen Versuchen nicht genau 2 Spalten vorhanden sind, wird die Datei übersprungen
            if df.shape[1] != 2:
                print(f"Datei {csv_file} hat {df.shape[1]} Spalten. Erwartet werden 2 Spalten. Datei wird übersprungen.")
                continue

            # Spalten umbenennen in "x" und "y"
            df.columns = ["x", "y"]
            
            # Export-Dateiname generieren (gleicher Name, aber mit .txt-Endung)
            base_name = os.path.basename(csv_file)
            name_without_ext = os.path.splitext(base_name)[0]
            export_file = os.path.join(destination_folder, f"{name_without_ext}.txt")
            
            # Als kommagetrennte TXT-Datei im UTF‑8-Format speichern
            df.to_csv(export_file, index=False, sep=",", encoding="utf-8")
            
            print(f"Erfolgreich exportiert: {export_file}")
        except Exception as e:
            print(f"Fehler beim Export von {csv_file}: {e}")
