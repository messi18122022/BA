import os

def clean_txt_files(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):  # Nur .txt-Dateien verarbeiten
            file_path = os.path.join(folder_path, filename)
            
            with open(file_path, "r", encoding="latin-1") as file:
                lines = file.readlines()
            
            # Prüfen, ob das File schon bereinigt ist
            if lines and lines[0].strip() == "min,µS/cm":
                print(f"Überspringe bereits bereinigte Datei: {filename}")
                continue
            
            # Nach der Zeile "min,µS/cm" suchen und den Index merken
            new_content = []
            found = False
            for line in lines:
                if "min,µS/cm" in line:
                    found = True
                    new_content = ["min,µS/cm\n"]  # Setze die erste Zeile als neuen Header
                    continue  # Diese Zeile wird nicht gespeichert, da sie der neue Header ist
                if found:
                    new_content.append(line)  # Nur die relevanten Daten speichern
            
            # Falls "min,µS/cm" gefunden wurde, Datei überschreiben
            if found:
                with open(file_path, "w", encoding="utf-8") as file:
                    file.writelines(new_content)
                print(f"Bereinigt: {filename}")
            else:
                print(f"Kein 'min,µS/cm' gefunden in: {filename}, Datei bleibt unverändert.")

# Beispielaufruf
ordner_pfad = "/Users/musamoin/Desktop/BA_Musa-Moin_FS25/Messungen/Juan/txt_files/"  # <--- Hier den tatsächlichen Pfad anpassen
clean_txt_files(ordner_pfad)
