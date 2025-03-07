import tkinter as tk
from tkinter import filedialog
import re
import PyPDF2

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

    # Nutze ein Regex, das die Zeilenstruktur berücksichtigt:
    # Zeile 1: Überschrift "Retentionszeit Halbwertsbreite"
    # Zeile 2: "min min"
    # Zeile 3: Zahlen (z.B. "1,065 0,063")
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

def main():
    root = tk.Tk()
    root.withdraw()

    data_dict = {'mit_saeule': [], 'ohne_saeule': []}

    # Dialogfenster für PDFs, die MIT Säule gemessen wurden
    mit_saeule_files = filedialog.askopenfilenames(
        title="Wählen Sie PDF-Dateien MIT Säule aus", 
        filetypes=[("PDF Dateien", "*.pdf")]
    )
    for file in mit_saeule_files:
        data = parse_pdf(file)
        data_dict['mit_saeule'].append(data)
    
    # Dialogfenster für PDFs, die OHNE Säule gemessen wurden (optional)
    ohne_saeule_files = filedialog.askopenfilenames(
        title="Wählen Sie optional PDF-Dateien OHNE Säule aus", 
        filetypes=[("PDF Dateien", "*.pdf")]
    )
    for file in ohne_saeule_files:
        data = parse_pdf(file)
        data_dict['ohne_saeule'].append(data)
    
    print("Gesammelte Daten:")
    print(data_dict)

if __name__ == "__main__":
    main()
