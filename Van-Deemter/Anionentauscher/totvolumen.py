#!/usr/bin/env python3

import os
import pandas as pd

# ---------------- Konfiguration ----------------
# Pfad zur Eingabe-CSV (in derselben directory wie dieses Skript)
INPUT_CSV  = "Van-Deemter/Anionentauscher/Ergebnisse_Retentionszeiten_Peakbreiten.csv"
# Pfad zur Ausgabe-CSV
OUTPUT_CSV = "Van-Deemter/Anionentauscher/Totvolumen_Säule.csv"

def main():
    # Existenz der Eingabe-Datei prüfen
    if not os.path.isfile(INPUT_CSV):
        raise FileNotFoundError(f"CSV-Datei nicht gefunden: {INPUT_CSV}")
    # Einlesen
    df = pd.read_csv(INPUT_CSV)
    
    # Korrigierte Retentionszeit der Säule
    df["tR_column (min)"] = (
        df["retentionszeit_mit_säule (min)"] 
        - df["retentionszeit_ohne_säule (min)"]
    )
    
    # Totvolumen der Säule = Flussrate * tR_column
    df["Totvolumen_Saeule (mL)"] = (
        df["flussrate (mL/min)"] 
        * df["tR_column (min)"]
    )

    # Berechnung von Mittelwert und Standardabweichung des Totvolumens
    mean_vol = df["Totvolumen_Saeule (mL)"].mean()
    std_vol = df["Totvolumen_Saeule (mL)"].std(ddof=1)

    # Erstelle Summary-Zeilen
    summary = pd.DataFrame([
        {col: "" for col in df.columns},
        {col: "" for col in df.columns}
    ])
    summary.iloc[0, summary.columns.get_loc("Totvolumen_Saeule (mL)")] = mean_vol
    summary.iloc[1, summary.columns.get_loc("Totvolumen_Saeule (mL)")] = std_vol
    summary.index = ["Mittelwert", "Standardabweichung"]

    # An DataFrame anhängen
    df = pd.concat([df, summary.reset_index()], ignore_index=False)
    
    # Speichern der erweiterten Tabelle
    df.to_csv(OUTPUT_CSV, index=True)
    print(f"Ergebnis-Datei gespeichert: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()