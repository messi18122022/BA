import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Computer Modern']
from matplotlib.backends.backend_pdf import PdfPages

# Parameter: Flussrate in mL/min
FLOW_RATE = 1.0

def calculate_retention_volume(time, intensity, mask):
    """
    Berechnet das Retentionsvolumen mittels des 1. mathematischen Moments über einen definierten Integrationsbereich.
    Es wird t_R = (∫ t * I(t) dt) / (∫ I(t) dt) berechnet und V_R = FLOW_RATE * t_R zurückgegeben.
    """
    integration_time = time[mask]
    integration_intensity = intensity[mask]
    numerator = np.trapz(integration_time * integration_intensity, integration_time)
    denominator = np.trapz(integration_intensity, integration_time)
    if denominator == 0:
        return np.nan
    t_R = numerator / denominator
    return FLOW_RATE * t_R

def process_directory(directory, has_header=True):
    """
    Liest alle TXT-Dateien in einem Verzeichnis ein, berechnet das Retentionsvolumen
    und erstellt einen Plot für jedes Chromatogramm.
    
    Parameter:
      directory: Pfad zum Datenverzeichnis.
      has_header: True, wenn die Dateien eine Kopfzeile besitzen.
      
    Rückgabe:
      retention_volumes: Liste der berechneten Retentionsvolumina.
      file_list: Liste der verarbeiteten Dateien.
    """
    retention_volumes = []
    file_list = sorted(glob.glob(os.path.join(directory, '*.txt')))
    file_list = [f for f in file_list if os.path.basename(f) != "Zusammenfassung.txt"]

    for file in file_list:
        try:
            if has_header:
                # Annahme: Die Datei enthält eine Kopfzeile, Spaltenname z.B. "x" und "y"
                df = pd.read_csv(file, sep=r',', header=1)
            else:
                # Bei Dateien ohne explizite Kopfzeile: Erste Zeile einlesen, um Separator und Header zu bestimmen
                with open(file, 'r') as f:
                    first_line = f.readline()
                sep = ',' if ',' in first_line else r'\s+'
                # Wenn die erste Zeile alphabetische Zeichen enthält, wird sie als Header interpretiert
                if any(c.isalpha() for c in first_line):
                    df = pd.read_csv(file, sep=sep, header=0)
                else:
                    df = pd.read_csv(file, sep=sep, header=None, names=['x', 'y'])
        except Exception as e:
            print(f"Fehler beim Lesen von {file}: {e}")
            continue
        
        # Falls die Spaltennamen nicht 'x' und 'y' sind, aber mindestens zwei Spalten vorhanden sind, diese umbenennen
        if 'x' not in df.columns or 'y' not in df.columns:
            if len(df.columns) >= 2:
                df.rename(columns={df.columns[0]: 'x', df.columns[1]: 'y'}, inplace=True)
                print(f"Spalten in {file} wurden umbenannt in 'x' und 'y'.")
            else:
                print(f"Datei {file} hat nicht genügend Spalten.")
                continue
        
        # Konvertierung der Spalten in float
        df['x'] = df['x'].astype(float)
        df['y'] = df['y'].astype(float)
        
        # Filter negative Zeiten
        df = df[df['x'] >= 0]
        
        # Extrahiere Zeit und Intensität
        time = df['x'].values
        intensity = df['y'].values
        
        # Verwende die Original-Rohdaten (keine Korrektur)
        
        # Definition des Integrationsbereichs abhängig vom Dateityp
        if has_header:
            integration_start = 1.1
            integration_end = 1.7
        else:
            integration_start = 0.0
            integration_end = 0.2
        mask_integration = (time >= integration_start) & (time <= integration_end)

        # Berechnung des Retentionsvolumens
        ret_volume = calculate_retention_volume(time, intensity, mask_integration)
        retention_volumes.append(ret_volume)
        
        # Berechnung der Retentionszeit
        retention_time = ret_volume / FLOW_RATE
        
        # Erstellung der Visualisierung
        with PdfPages(file.replace('.txt', '.pdf')) as pdf:
            fig, axs = plt.subplots(1, 2, figsize=(16/2.54, 8/2.54))
            
            # Gesamtansicht
            axs[0].plot(time, intensity, label='Chromatogramm')
            axs[0].axvline(x=retention_time, color='red', linestyle='--', label=f'$t_R = {retention_time:.3f}$ min')
            axs[0].set_xlabel('Zeit (min)')
            axs[0].set_ylabel('Signal (mAU)')
            axs[0].legend(loc='upper right')
            
            # Manuelle Zoom-Einstellung
            axs[1].plot(time, intensity, label='Chromatogramm')
            if has_header:
                axs[1].set_xlim(1.0, 2.5)  # Zoombereich für data_with_column
            else:
                axs[1].set_xlim(0.0, 0.6)  # Zoombereich für data_without_column
            
            axs[1].axvline(x=retention_time, color='red', linestyle='--', label=f'$t_R = {retention_time:.3f}$ min')
            axs[1].set_xlabel('Zeit (min)')
            axs[1].set_ylabel('Signal (mAU)')
            axs[1].legend(loc='upper right')
            
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
        
        print(f"Verarbeitet: {file}, Retentionsvolumen: {ret_volume:.3f} mL, Retentionszeit: {retention_time:.3f} min, Plot: {file.replace('.txt', '.pdf')}")
        
    return retention_volumes, file_list

def save_summary(directory, retention_volumes):
    """
    Speichert eine Zusammenfassung (Mittelwert und Standardabweichung) der Retentionsvolumina
    in einer Textdatei im angegebenen Verzeichnis.
    """
    if len(retention_volumes) == 0:
        print(f"Keine Retentionsvolumen in {directory} berechnet.")
        return
    
    mean_val = np.mean(retention_volumes)
    std_val = np.std(retention_volumes, ddof=1)  # Stichproben-Standardabweichung
    summary_text = f'Mittelwert: {mean_val:.3f} mL\nStandardabweichung: {std_val:.3f} mL'
    summary_file = os.path.join(directory, 'Zusammenfassung.txt')
    
    with open(summary_file, 'w') as f:
        f.write(summary_text)
    print(f"Zusammenfassung gespeichert: {summary_file}")

if __name__ == '__main__':
    # Basisverzeichnis anpassen, falls notwendig
    base_path = 'Totvolumenbestimmung'
    
    # Verzeichnisse und Angabe, ob die Dateien eine Kopfzeile haben
    directories = {
        'data_with_column': True,
        'data_without_column': False
    }
    
    for dir_name, has_header in directories.items():
        directory_path = os.path.join(base_path, dir_name)
        print(f"\nVerarbeite Verzeichnis: {directory_path}")
        retention_volumes, files = process_directory(directory_path, has_header)
        save_summary(directory_path, retention_volumes)