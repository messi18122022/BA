import tkinter as tk
from tkinter import filedialog
import re
import PyPDF2
import matplotlib.pyplot as plt

def parse_pdf(file_path):
    data = {}
    with open(file_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    
    # Extrahiere Ident ohne den Zusatz " Ident"
    match_ident = re.search(r'(\d{4}-\d{2}-\d{2}_[^\s]+)(?=\s*Ident)', text)
    data['Ident'] = match_ident.group(1) if match_ident else None

    # Erwartete Zeilen:
    # Zeile 1: "Retentionszeit Halbwertsbreite"
    # Zeile 2: "min min"
    # Zeile 3: Zahlen, z.B. "1,065 0,063"
    pattern = r'Retentionszeit\s+Halbwertsbreite\s*\n\s*min\s+min\s*\n\s*([\d,]+)\s+([\d,]+)'
    match_rhw = re.search(pattern, text)
    if match_rhw:
        retention_str = match_rhw.group(1).replace(',', '.')
        half_str = match_rhw.group(2).replace(',', '.')
        retention_value = float(retention_str)
        # Subtrahiere 0.0002025 von der Halbwertsbreite
        half_value = float(half_str) - 0.0002025
        data['Retentionszeit'] = retention_value
        data['Halbwertsbreite'] = half_value
    else:
        data['Retentionszeit'] = None
        data['Halbwertsbreite'] = None

    # Extrahiere die Flussrate (z.B. "Fluss 1,000 mL/min")
    match_fluss = re.search(r'Fluss\s+([0-9,]+)\s*mL/min', text)
    if match_fluss:
        flow_str = match_fluss.group(1).replace(',', '.')
        data['Flussrate'] = float(flow_str)
    else:
        data['Flussrate'] = None

    return data

def read_data():
    root = tk.Tk()
    root.withdraw()

    data_dict = {'mit_saeule': [], 'ohne_saeule': []}

    # PDFs MIT Säule
    mit_saeule_files = filedialog.askopenfilenames(
        title="Wählen Sie PDF-Dateien MIT Säule aus", 
        filetypes=[("PDF Dateien", "*.pdf")]
    )
    for file in mit_saeule_files:
        data = parse_pdf(file)
        data_dict['mit_saeule'].append(data)
    
    # PDFs OHNE Säule (optional)
    ohne_saeule_files = filedialog.askopenfilenames(
        title="Wählen Sie optional PDF-Dateien OHNE Säule aus", 
        filetypes=[("PDF Dateien", "*.pdf")]
    )
    for file in ohne_saeule_files:
        data = parse_pdf(file)
        data_dict['ohne_saeule'].append(data)
    
    print("Ausgelesene Daten:")
    print(data_dict)
    return data_dict

def van_deemter_analysis(data_dict, L=150):
    """
    Berechnet die Bodenhöhe H gemäß:
    
    H = L * ((w_obs)^2 - (w_ext)^2) / (5.55*(t_obs - t_ext)^2)
    
    Falls keine Messungen OHNE Säule vorliegen, wird t_ext = 0 und w_ext = 0 gesetzt.
    L: Säulenlänge in mm.
    """
    flow_rates = []
    H_values = []
    
    mit_list = data_dict['mit_saeule']
    ohne_list = data_dict['ohne_saeule']
    
    # Prüfe, ob Messungen ohne Säule vorliegen
    use_external = len(ohne_list) > 0

    for i, m in enumerate(mit_list):
        # Für Messungen OHNE Säule entweder den entsprechenden Wert verwenden oder 0 einsetzen
        if use_external:
            # Falls auch eine entsprechende Messung ohne Säule existiert
            if i < len(ohne_list):
                o = ohne_list[i]
                t_ext = o['Retentionszeit'] if o['Retentionszeit'] is not None else 0
                w_ext = o['Halbwertsbreite'] if o['Halbwertsbreite'] is not None else 0
            else:
                t_ext = 0
                w_ext = 0
        else:
            t_ext = 0
            w_ext = 0

        # Prüfe, ob die erforderlichen Werte vorliegen
        if m['Retentionszeit'] is None or m['Halbwertsbreite'] is None or m['Flussrate'] is None:
            continue
        
        t_obs = m['Retentionszeit']
        w_obs = m['Halbwertsbreite']
        u = m['Flussrate']
        
        # Verhindere Division durch 0
        if (t_obs - t_ext) == 0:
            continue
        
        H = L * ((w_obs)**2 - (w_ext)**2) / (5.55 * (t_obs - t_ext)**2)
        flow_rates.append(u)
        H_values.append(H)
    
    print("Flussraten (u):", flow_rates)
    print("Bodenhöhen (H):", H_values)
    
    # Erstelle den Van-Deemter-Plot
    plt.figure(figsize=(8,6))
    plt.plot(flow_rates, H_values, 'o-', label='Van Deemter Plot')
    plt.xlabel('Flussrate u (mL/min)')
    plt.ylabel('Bodenhöhe H (mm)')
    plt.title('Van Deemter Plot')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    data_dict = read_data()
    van_deemter_analysis(data_dict, L=150)

if __name__ == "__main__":
    main()
