import tkinter as tk
from tkinter import filedialog, messagebox
import re
import PyPDF2

def extract_data_from_pdf(filepath):
    """
    Liest die PDF-Datei ein, sucht nach "IP" und extrahiert die ersten drei Float-Zahlen,
    die t_R, w_1/2 und Flussrate repräsentieren sollen.
    """
    try:
        with open(filepath, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
        # Suche nach "IP" und extrahiere drei Zahlen (Beispiel: "IP ... 10.23 ... 2.34 ... 1.50")
        pattern = r"IP.*?(\d+\.\d+).*?(\d+\.\d+).*?(\d+\.\d+)"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            t_R = float(match.group(1))
            w_half = float(match.group(2))
            flussrate = float(match.group(3))
            return t_R, w_half, flussrate
        else:
            messagebox.showerror("Fehler", f"Die Daten konnten in der Datei {filepath} nicht gefunden werden.")
            return None, None, None
    except Exception as e:
        messagebox.showerror("Fehler", f"Fehler beim Lesen der Datei {filepath}:\n{e}")
        return None, None, None

def main():
    # Tkinter-Hauptfenster erstellen und ausblenden
    root = tk.Tk()
    root.withdraw()

    # Säulenlänge L in mm festlegen (anpassen falls nötig)
    L = 150

    # Dialog: PDFs für Messungen MIT Säule auswählen
    messagebox.showinfo("Messungen MIT Säule", "Bitte wählen Sie die PDF-Datei(en) für die Messungen MIT Säule aus.")
    file_paths_obs = filedialog.askopenfilenames(title="PDFs für Messungen MIT Säule", filetypes=[("PDF files", "*.pdf")])
    if not file_paths_obs:
        messagebox.showerror("Fehler", "Keine Datei für Messungen MIT Säule ausgewählt.")
        return

    # Für dieses Beispiel wird die erste ausgewählte Datei verwendet
    t_R_obs, w_half_obs, flussrate_obs = extract_data_from_pdf(file_paths_obs[0])
    if t_R_obs is None or w_half_obs is None:
        return

    # Dialog: Optionale PDFs für Messungen OHNE Säule auswählen
    messagebox.showinfo("Messungen OHNE Säule (optional)", "Bitte wählen Sie optional die PDF-Datei(en) für die Messungen OHNE Säule aus.\n(Klicken Sie auf Abbrechen, falls nicht vorhanden.)")
    file_paths_ext = filedialog.askopenfilenames(title="PDFs für Messungen OHNE Säule", filetypes=[("PDF files", "*.pdf")])
    if file_paths_ext:
        t_R_ext, w_half_ext, flussrate_ext = extract_data_from_pdf(file_paths_ext[0])
        if t_R_ext is None or w_half_ext is None:
            return
    else:
        t_R_ext = 0
        w_half_ext = 0

    # Berechnung der Van Deemter Zahl H
    try:
        # Formel: H = L / [5.55 * ((t_R_obs - t_R_ext)**2 / (w_half_obs**2 - w_half_ext**2))]
        denominator = 5.55 * ((t_R_obs - t_R_ext)**2 / (w_half_obs**2 - w_half_ext**2))
        H = L / denominator
    except ZeroDivisionError:
        messagebox.showerror("Fehler", "Division durch Null bei der Berechnung. Überprüfen Sie die extrahierten Werte.")
        return
    except Exception as e:
        messagebox.showerror("Fehler", f"Fehler bei der Berechnung:\n{e}")
        return

    # Ergebnis anzeigen
    result_text = (f"Berechnete Van Deemter Zahl H: {H:.4f}\n\n"
                   f"Messungen MIT Säule:\n  t_R = {t_R_obs}, w₁/₂ = {w_half_obs}, Flussrate = {flussrate_obs}\n"
                   f"Messungen OHNE Säule:\n  t_R = {t_R_ext}, w₁/₂ = {w_half_ext}")
    messagebox.showinfo("Ergebnis", result_text)

if __name__ == "__main__":
    main()
