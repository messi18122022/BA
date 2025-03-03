#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Van Deemter Analyse:
  • Einlesen von Messungen mit Säule und optional ohne Säule,
  • Falls keine Datei ohne Säule hochgeladen wird, erfolgt die Analyse nur anhand der Messungen mit Säule (ohne extrakolumnare Korrektur).
  • Berechnung der FWHM:
      - Bei Vorliegen beider Messungen: Extrakolumn-Korrektur wird durchgeführt.
      - Andernfalls: Es wird nur der raw Wert (fwhm_with) verwendet.
  • Berechnung der theoretischen Plattenzahl N = 5.55*(t_R)²/(FWHM)² und der Plattenhöhe H = L/N.
  • Erstellen des van Deemter Plots (H vs. Flussrate) inklusive Fit der van-Deemter-Gleichung.
  • Der Report (PDF) enthält:
      - Bei Vorliegen beider Messungen: 
           Seite 1: van Deemter Plot mit extrakolumnarer Korrektur,
           Seite 2: van Deemter Plot ohne extrakolumnare Korrektur,
      - Falls nur Messungen mit Säule vorliegen:
           Seite 1: van Deemter Plot (ohne extrakolumnare Korrektur),
      - Weiterhin: Einzelplots der Messungen und eine Ergebnisstabelle.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.signal import peak_widths, savgol_filter
from scipy.optimize import curve_fit
import tkinter as tk
from tkinter import filedialog, simpledialog
import os
import sys
import matplotlib.gridspec as gridspec

# Säulenlänge in mm (Standard)
L = 100.0

def shorten_filename(filename):
    base = os.path.basename(filename)
    index = base.find("_clt")
    if index != -1:
        return base[:index]
    return base

def read_data(filename):
    """Liest Zeit [min] und Signal aus einer Textdatei ein."""
    with open(filename, 'r', encoding='latin-1') as f:
        lines = f.readlines()
    data_start = None
    for i, line in enumerate(lines):
        if line.strip().startswith("min"):
            data_start = i + 1
            break
    if data_start is None:
        raise ValueError(f"Die Datei {filename} enthält keinen erkennbaren Header 'min,µS/cm'.")
    data = np.genfromtxt(lines[data_start:], delimiter=",")
    if data.ndim == 1 or data.shape[1] < 2:
        raise ValueError(f"Die Datei {filename} enthält keine gültigen 2-spaltigen Daten.")
    return data[:,0], data[:,1]

def gaussian_offset(x, A, mu, sigma, B):
    """Gauß-Funktion mit Offset (Baseline)."""
    return B + A * np.exp(-((x - mu)**2) / (2 * sigma**2))

def van_deemter_eq(flow, A, B, C):
    """Van-Deemter-Gleichung: H = A + B/flow + C*flow"""
    return A + B/flow + C*flow

def process_pairs(files_with, files_without, flow_min, flow_max, delta, pdf_filename="Van-Deemter/vanDeemter_Report.pdf"):
    # Prüfe, ob auch Dateien ohne Säule hochgeladen wurden
    both_available = True if files_without else False

    if both_available:
        if len(files_with) != len(files_without):
            print("Die Anzahl der Dateien mit Säule und ohne Säule stimmt nicht überein.")
            sys.exit(1)
    # Anzahl der Flussratenpunkte muss der Anzahl der Messungen (mit Säule) entsprechen
    n_points = int(round((flow_max - flow_min)/delta)) + 1
    if n_points != len(files_with):
        print(f"Fehler: {n_points} berechnete Flussratenpunkte, aber {len(files_with)} Dateien (Messungen mit Säule).")
        sys.exit(1)

    flow_rates = np.linspace(flow_min, flow_max, n_points)

    # Listen für van Deemter
    vanDeemter_H_corrected = []  # nur bei Vorliegen beider Messungen
    vanDeemter_H_raw = []        # Raw-Werte (ohne extrakolumnare Korrektur)
    measurement_figs = []
    results = []
    const = np.sqrt(8 * np.log(2))  # w₁/₂ = σ * const

    if both_available:
        # Verarbeitung, wenn beide Messungen vorliegen
        for j, (file_with, file_without) in enumerate(zip(files_with, files_without)):
            try:
                t_with, y_with = read_data(file_with)
                t_without, y_without = read_data(file_without)
            except Exception as e:
                print(f"Fehler beim Lesen der Dateien {file_with} oder {file_without}: {e}")
                continue

            # Glätten
            window_length_with = 11 if len(y_with) >= 11 else (len(y_with)//2)*2+1
            window_length_without = 11 if len(y_without) >= 11 else (len(y_without)//2)*2+1
            y_with_smooth = savgol_filter(y_with, window_length_with, polyorder=3)
            y_without_smooth = savgol_filter(y_without, window_length_without, polyorder=3)

            # Peak (mit Säule)
            peak_idx_with = np.argmax(y_with_smooth)
            retention_time_with = t_with[peak_idx_with]
            widths_with, _, _, _ = peak_widths(y_with_smooth, [peak_idx_with], rel_height=0.5)
            fwhm_with = widths_with[0]*(t_with[1]-t_with[0])

            # Peak (ohne Säule)
            peak_idx_without = np.argmax(y_without_smooth)
            retention_time_without = t_without[peak_idx_without]
            widths_without, _, _, _ = peak_widths(y_without_smooth, [peak_idx_without], rel_height=0.5)
            fwhm_without = widths_without[0]*(t_without[1]-t_without[0])

            # FWHM-Korrektur: Abzug des extrakolumnaren Beitrags
            sigma_obs = fwhm_with/const
            sigma_ext = fwhm_without/const
            sigma_col = np.sqrt(sigma_obs**2 - sigma_ext**2) if sigma_obs**2 >= sigma_ext**2 else 0
            fwhm_col = sigma_col * const

            # Berechnung der Plattenzahl und H (korrigiert)
            if fwhm_col == 0:
                N = np.nan
            else:
                N = 5.55 * (retention_time_with**2) / (fwhm_col**2)
            H_corrected = L / N if N != 0 and not np.isnan(N) else np.nan
            vanDeemter_H_corrected.append(H_corrected)

            # Berechnung der Plattenzahl und H (raw, ohne Korrektur) mit fwhm_with
            if fwhm_with == 0:
                N_raw = np.nan
            else:
                N_raw = 5.55 * (retention_time_with**2) / (fwhm_with**2)
            H_raw = L / N_raw if N_raw != 0 and not np.isnan(N_raw) else np.nan
            vanDeemter_H_raw.append(H_raw)

            # PGF Berechnung
            widths_with_10, _, _, _ = peak_widths(y_with_smooth, [peak_idx_with], rel_height=0.9)
            w1_10_with = widths_with_10[0]*(t_with[1]-t_with[0])
            widths_without_10, _, _, _ = peak_widths(y_without_smooth, [peak_idx_without], rel_height=0.9)
            w1_10_without = widths_without_10[0]*(t_without[1]-t_without[0])
            pgf_with = 1.83*(fwhm_with/w1_10_with) if w1_10_with != 0 else np.nan
            pgf_without = 1.83*(fwhm_without/w1_10_without) if w1_10_without != 0 else np.nan

            results.append({
                "Datei mit Säule": shorten_filename(file_with),
                "Datei ohne Säule": shorten_filename(file_without),
                "FWHM (Säule) [min]": fwhm_col
            })

            # Zoom-Bereiche
            def safe_zoom(t, tR, fwhm):
                z_margin = 1.5 * fwhm
                left = max(tR - z_margin, t[0])
                right = min(tR + z_margin, t[-1])
                return (t >= left) & (t <= right)
            mask_zoom_with = safe_zoom(t_with, retention_time_with, fwhm_with)
            mask_zoom_without = safe_zoom(t_without, retention_time_without, fwhm_without)

            # Gaussian-Fit
            def estimate_baseline(x, y, mask):
                idx = np.where(mask)[0]
                if len(idx) < 2:
                    return np.min(y)
                edge_count = max(1, int(0.1 * len(idx)))
                return np.mean(np.concatenate((y[idx[:edge_count]], y[idx[-edge_count:]])))
            
            try:
                fit_window_factor = 1.0
                mask_fit_with = (t_with >= retention_time_with - fit_window_factor*fwhm_with) & (t_with <= retention_time_with + fit_window_factor*fwhm_with)
                B0_with = estimate_baseline(t_with, y_with_smooth, mask_fit_with)
                p0_with = [y_with_smooth[peak_idx_with]-B0_with, retention_time_with, fwhm_with/(2*np.sqrt(2*np.log(2))), B0_with]
                popt_with, _ = curve_fit(gaussian_offset, t_with[mask_fit_with], y_with_smooth[mask_fit_with], p0=p0_with)
                gauss_fit_with = gaussian_offset(t_with, *popt_with)
            except:
                gauss_fit_with = np.zeros_like(t_with)
            
            try:
                mask_fit_without = (t_without >= retention_time_without - fit_window_factor*fwhm_without) & (t_without <= retention_time_without + fit_window_factor*fwhm_without)
                B0_without = estimate_baseline(t_without, y_without_smooth, mask_fit_without)
                p0_without = [y_without_smooth[peak_idx_without]-B0_without, retention_time_without, fwhm_without/(2*np.sqrt(2*np.log(2))), B0_without]
                popt_without, _ = curve_fit(gaussian_offset, t_without[mask_fit_without], y_without_smooth[mask_fit_without], p0=p0_without)
                gauss_fit_without = gaussian_offset(t_without, *popt_without)
            except:
                gauss_fit_without = np.zeros_like(t_without)

            # Einzelplots (3x2 Layout für Messungen mit und ohne Säule)
            fig, axs = plt.subplots(3, 2, figsize=(8.27, 11.69))
            fig.suptitle(f"Messung {j+1}: {shorten_filename(file_with)} vs. {shorten_filename(file_without)}", fontsize=14)

            # (0,0): Mit Säule – Komplett
            axs[0,0].plot(t_with, y_with, 'ko', markersize=3, label="Messdaten")
            axs[0,0].plot(t_with, y_with_smooth, 'b-', linewidth=1, label="Geglättet")
            axs[0,0].plot(t_with, gauss_fit_with, 'r-', linewidth=1, label="Gaussian Fit")
            axs[0,0].axvline(retention_time_with - fwhm_with/2, color='gray', linestyle='--', linewidth=1, alpha=0.7)
            axs[0,0].axvline(retention_time_with + fwhm_with/2, color='gray', linestyle='--', linewidth=1, alpha=0.7)
            axs[0,0].axvline(retention_time_with - w1_10_with/2, color='blue', linestyle=':', linewidth=1, alpha=0.7)
            axs[0,0].axvline(retention_time_with + w1_10_with/2, color='blue', linestyle=':', linewidth=1, alpha=0.7)
            axs[0,0].set_xlabel("Zeit [min]")
            axs[0,0].set_ylabel("Signal")
            axs[0,0].legend(fontsize=8)
            axs[0,0].set_title("Mit Säule - Komplett")

            # (0,1): Mit Säule – Zoom
            axs[0,1].plot(t_with[mask_zoom_with], y_with[mask_zoom_with], 'ko', markersize=3, label="Messdaten (Zoom)")
            axs[0,1].plot(t_with[mask_zoom_with], y_with_smooth[mask_zoom_with], 'b-', linewidth=1, label="Geglättet (Zoom)")
            axs[0,1].plot(t_with[mask_zoom_with], gauss_fit_with[mask_zoom_with], 'r-', linewidth=1, label="Gaussian Fit")
            axs[0,1].axvline(retention_time_with - fwhm_with/2, color='gray', linestyle='--', linewidth=1, alpha=0.7)
            axs[0,1].axvline(retention_time_with + fwhm_with/2, color='gray', linestyle='--', linewidth=1, alpha=0.7)
            axs[0,1].axvline(retention_time_with - w1_10_with/2, color='blue', linestyle=':', linewidth=1, alpha=0.7)
            axs[0,1].axvline(retention_time_with + w1_10_with/2, color='blue', linestyle=':', linewidth=1, alpha=0.7)
            axs[0,1].set_xlabel("Zeit [min]")
            axs[0,1].set_ylabel("Signal")
            axs[0,1].legend(fontsize=8)
            axs[0,1].set_title("Mit Säule - Zoom")

            # (1,0): Ohne Säule – Komplett
            axs[1,0].plot(t_without, y_without, 'ko', markersize=3, label="Messdaten")
            axs[1,0].plot(t_without, y_without_smooth, 'b-', linewidth=1, label="Geglättet")
            axs[1,0].plot(t_without, gauss_fit_without, 'r-', linewidth=1, label="Gaussian Fit")
            axs[1,0].axvline(retention_time_without - fwhm_without/2, color='gray', linestyle='--', linewidth=1, alpha=0.7)
            axs[1,0].axvline(retention_time_without + fwhm_without/2, color='gray', linestyle='--', linewidth=1, alpha=0.7)
            axs[1,0].axvline(retention_time_without - w1_10_without/2, color='blue', linestyle=':', linewidth=1, alpha=0.7)
            axs[1,0].axvline(retention_time_without + w1_10_without/2, color='blue', linestyle=':', linewidth=1, alpha=0.7)
            axs[1,0].set_xlabel("Zeit [min]")
            axs[1,0].set_ylabel("Signal")
            axs[1,0].legend(fontsize=8)
            axs[1,0].set_title("Ohne Säule - Komplett")

            # (1,1): Ohne Säule – Zoom
            axs[1,1].plot(t_without[mask_zoom_without], y_without[mask_zoom_without], 'ko', markersize=3, label="Messdaten (Zoom)")
            axs[1,1].plot(t_without[mask_zoom_without], y_without_smooth[mask_zoom_without], 'b-', linewidth=1, label="Geglättet (Zoom)")
            axs[1,1].plot(t_without[mask_zoom_without], gauss_fit_without[mask_zoom_without], 'r-', linewidth=1, label="Gaussian Fit")
            axs[1,1].axvline(retention_time_without - fwhm_without/2, color='gray', linestyle='--', linewidth=1, alpha=0.7)
            axs[1,1].axvline(retention_time_without + fwhm_without/2, color='gray', linestyle='--', linewidth=1, alpha=0.7)
            axs[1,1].axvline(retention_time_without - w1_10_without/2, color='blue', linestyle=':', linewidth=1, alpha=0.7)
            axs[1,1].axvline(retention_time_without + w1_10_without/2, color='blue', linestyle=':', linewidth=1, alpha=0.7)
            axs[1,1].set_xlabel("Zeit [min]")
            axs[1,1].set_ylabel("Signal")
            axs[1,1].legend(fontsize=8)
            axs[1,1].set_title("Ohne Säule - Zoom")

            # (2,0)-(2,1): Ergebnistext (ein Block)
            axs[2,0].axis('off')
            axs[2,1].axis('off')
            result_text = (
                "Ergebnisse:\n"
                "Mit Säule: PGF = " + f"{pgf_with:.2f}" + "\n"
                "Ohne Säule: PGF = " + f"{pgf_without:.2f}" + "\n\n"
                "FWHM (Säule) (korrigiert): " + f"{fwhm_col:.4f} min\n"
                "Akzeptanzkriterium: 0.8 < PGF < 1.15"
            )
            axs[2,0].text(0.05, 0.5, result_text, transform=axs[2,0].transAxes,
                          fontsize=10, va="center", bbox=dict(facecolor='white', alpha=0.5))

            fig.tight_layout()
            measurement_figs.append(fig)
            plt.close(fig)

        # Van-Deemter-Plots:
        # Seite 1: Plot mit extrakolumnarer Korrektur
        fig_vd_corr, (ax_plot_corr, ax_text_corr) = plt.subplots(2, 1, figsize=(8.27, 11.69))
        fig_vd_corr.suptitle("van Deemter Plot (mit extrakolumnarer Korrektur)", fontsize=14)
        ax_plot_corr.plot(flow_rates, vanDeemter_H_corrected, 'bo', label="Messpunkte (H, korrigiert)")
        ax_plot_corr.set_xlabel("Flussrate (mL/min)")
        ax_plot_corr.set_ylabel("H (mm)")
        try:
            valid_mask_corr = ~np.isnan(vanDeemter_H_corrected)
            flow_valid_corr = flow_rates[valid_mask_corr]
            H_valid_corr = np.array(vanDeemter_H_corrected)[valid_mask_corr]
            popt_corr, pcov_corr = curve_fit(van_deemter_eq, flow_valid_corr, H_valid_corr, p0=[1, 1, 1])
            A_fit_corr, B_fit_corr, C_fit_corr = popt_corr
            perr_corr = np.sqrt(np.diag(pcov_corr))
            H_pred_corr = van_deemter_eq(flow_valid_corr, A_fit_corr, B_fit_corr, C_fit_corr)
            ss_res_corr = np.sum((H_valid_corr - H_pred_corr)**2)
            ss_tot_corr = np.sum((H_valid_corr - np.mean(H_valid_corr))**2)
            r2_corr = 1 - ss_res_corr/ss_tot_corr

            flow_dense_corr = np.linspace(flow_min, flow_max, 200)
            H_fit_dense_corr = van_deemter_eq(flow_dense_corr, A_fit_corr, B_fit_corr, C_fit_corr)
            ax_plot_corr.plot(flow_dense_corr, H_fit_dense_corr, 'r-', label="Fit: H = A + B/flow + C*flow")
            ax_plot_corr.legend()

            ax_text_corr.axis('off')
            param_text_corr = (
                "Van-Deemter-Fit (korrigiert):\n"
                f"A = {A_fit_corr:.3e} ± {perr_corr[0]:.3e}\n"
                f"B = {B_fit_corr:.3e} ± {perr_corr[1]:.3e}\n"
                f"C = {C_fit_corr:.3e} ± {perr_corr[2]:.3e}\n\n"
                f"R² = {r2_corr:.4f}"
            )
            ax_text_corr.text(0.05, 0.9, param_text_corr, transform=ax_text_corr.transAxes, va='top',
                              fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
        except Exception as e:
            print("Fehler beim van-Deemter-Fit (korrigiert):", e)

        # Seite 2: Plot raw (ohne extrakolumnare Korrektur)
        fig_vd_raw, (ax_plot_raw, ax_text_raw) = plt.subplots(2, 1, figsize=(8.27, 11.69))
        fig_vd_raw.suptitle("van Deemter Plot (ohne extrakolumnare Korrektur)", fontsize=14)
        ax_plot_raw.plot(flow_rates, vanDeemter_H_raw, 'bo', label="Messpunkte (H, raw)")
        ax_plot_raw.set_xlabel("Flussrate (mL/min)")
        ax_plot_raw.set_ylabel("H (mm)")
        try:
            valid_mask_raw = ~np.isnan(vanDeemter_H_raw)
            flow_valid_raw = flow_rates[valid_mask_raw]
            H_valid_raw = np.array(vanDeemter_H_raw)[valid_mask_raw]
            popt_raw, pcov_raw = curve_fit(van_deemter_eq, flow_valid_raw, H_valid_raw, p0=[1, 1, 1])
            A_fit_raw, B_fit_raw, C_fit_raw = popt_raw
            perr_raw = np.sqrt(np.diag(pcov_raw))
            H_pred_raw = van_deemter_eq(flow_valid_raw, A_fit_raw, B_fit_raw, C_fit_raw)
            ss_res_raw = np.sum((H_valid_raw - H_pred_raw)**2)
            ss_tot_raw = np.sum((H_valid_raw - np.mean(H_valid_raw))**2)
            r2_raw = 1 - ss_res_raw/ss_tot_raw

            flow_dense_raw = np.linspace(flow_min, flow_max, 200)
            H_fit_dense_raw = van_deemter_eq(flow_dense_raw, A_fit_raw, B_fit_raw, C_fit_raw)
            ax_plot_raw.plot(flow_dense_raw, H_fit_dense_raw, 'r-', label="Fit: H = A + B/flow + C*flow")
            ax_plot_raw.legend()

            ax_text_raw.axis('off')
            param_text_raw = (
                "Van-Deemter-Fit (raw):\n"
                f"A = {A_fit_raw:.3e} ± {perr_raw[0]:.3e}\n"
                f"B = {B_fit_raw:.3e} ± {perr_raw[1]:.3e}\n"
                f"C = {C_fit_raw:.3e} ± {perr_raw[2]:.3e}\n\n"
                f"R² = {r2_raw:.4f}"
            )
            ax_text_raw.text(0.05, 0.9, param_text_raw, transform=ax_text_raw.transAxes, va='top',
                             fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
        except Exception as e:
            print("Fehler beim van-Deemter-Fit (raw):", e)

        # PDF-Report speichern (beide van Deemter Plots, Messungen und Tabelle)
        with PdfPages(pdf_filename) as pdf:
            pdf.savefig(fig_vd_corr)  # Seite 1: korrigierter Plot
            pdf.savefig(fig_vd_raw)   # Seite 2: raw Plot
            for fig in measurement_figs:
                pdf.savefig(fig)
            # Ergebnisstabelle
            fig_table, ax_table = plt.subplots(figsize=(8.27, 11.69))
            ax_table.axis('tight')
            ax_table.axis('off')
            table_data = [["Datei mit Säule", "Datei ohne Säule", "FWHM (Säule) [min]"]]
            for res in results:
                table_data.append([res["Datei mit Säule"], res["Datei ohne Säule"], f"{res['FWHM (Säule) [min]']:.4f}"])
            table = ax_table.table(cellText=table_data, loc="center", cellLoc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.scale(1, 1.5)
            ax_table.set_title("Übersicht der FWHM (Säule)", fontweight="bold")
            pdf.savefig(fig_table)
            plt.close('all')
        print(f"\nReport wurde erstellt: {pdf_filename}")

    else:
        # Verarbeitung, wenn nur Dateien mit Säule vorliegen (ohne extrakolumnare Korrektur)
        for j, file_with in enumerate(files_with):
            try:
                t_with, y_with = read_data(file_with)
            except Exception as e:
                print(f"Fehler beim Lesen der Datei {file_with}: {e}")
                continue

            # Glätten
            window_length_with = 11 if len(y_with) >= 11 else (len(y_with)//2)*2+1
            y_with_smooth = savgol_filter(y_with, window_length_with, polyorder=3)

            # Peak (mit Säule)
            peak_idx_with = np.argmax(y_with_smooth)
            retention_time_with = t_with[peak_idx_with]
            widths_with, _, _, _ = peak_widths(y_with_smooth, [peak_idx_with], rel_height=0.5)
            fwhm_with = widths_with[0]*(t_with[1]-t_with[0])

            # Da keine Datei ohne Säule vorliegt, wird keine extrakolumnare Korrektur durchgeführt.
            # fwhm_col entspricht also fwhm_with.
            fwhm_col = fwhm_with

            # Berechnung der Plattenzahl und H (raw)
            if fwhm_with == 0:
                N_raw = np.nan
            else:
                N_raw = 5.55 * (retention_time_with**2) / (fwhm_with**2)
            H_raw = L / N_raw if N_raw != 0 and not np.isnan(N_raw) else np.nan
            vanDeemter_H_raw.append(H_raw)

            # PGF Berechnung
            widths_with_10, _, _, _ = peak_widths(y_with_smooth, [peak_idx_with], rel_height=0.9)
            w1_10_with = widths_with_10[0]*(t_with[1]-t_with[0])
            pgf_with = 1.83*(fwhm_with/w1_10_with) if w1_10_with != 0 else np.nan

            results.append({
                "Datei": shorten_filename(file_with),
                "FWHM (Säule) [min]": fwhm_with
            })

            # Erstelle Einzelplot für die Messung (nur mit Säule)
            # Verwende GridSpec: obere Reihe mit 2 Spalten (Komplett und Zoom), untere Reihe: Ergebnistext (über beide Spalten)
            fig = plt.figure(figsize=(8.27, 11.69))
            gs = gridspec.GridSpec(2, 2, height_ratios=[3, 1])
            ax_complete = fig.add_subplot(gs[0, 0])
            ax_zoom = fig.add_subplot(gs[0, 1])
            ax_text = fig.add_subplot(gs[1, :])
            
            # Komplett-Plot
            ax_complete.plot(t_with, y_with, 'ko', markersize=3, label="Messdaten")
            ax_complete.plot(t_with, y_with_smooth, 'b-', linewidth=1, label="Geglättet")
            try:
                fit_window_factor = 1.0
                mask_fit_with = (t_with >= retention_time_with - fit_window_factor*fwhm_with) & (t_with <= retention_time_with + fit_window_factor*fwhm_with)
                B0_with = np.min(y_with_smooth[mask_fit_with])
                p0_with = [y_with_smooth[peak_idx_with]-B0_with, retention_time_with, fwhm_with/(2*np.sqrt(2*np.log(2))), B0_with]
                popt_with, _ = curve_fit(gaussian_offset, t_with[mask_fit_with], y_with_smooth[mask_fit_with], p0=p0_with)
                gauss_fit_with = gaussian_offset(t_with, *popt_with)
                ax_complete.plot(t_with, gauss_fit_with, 'r-', linewidth=1, label="Gaussian Fit")
            except:
                pass
            ax_complete.axvline(retention_time_with - fwhm_with/2, color='gray', linestyle='--', linewidth=1, alpha=0.7)
            ax_complete.axvline(retention_time_with + fwhm_with/2, color='gray', linestyle='--', linewidth=1, alpha=0.7)
            ax_complete.set_xlabel("Zeit [min]")
            ax_complete.set_ylabel("Signal")
            ax_complete.legend(fontsize=8)
            ax_complete.set_title("Mit Säule - Komplett")

            # Zoom-Plot
            def safe_zoom(t, tR, fwhm):
                z_margin = 1.5 * fwhm
                left = max(tR - z_margin, t[0])
                right = min(tR + z_margin, t[-1])
                return (t >= left) & (t <= right)
            mask_zoom_with = safe_zoom(t_with, retention_time_with, fwhm_with)
            ax_zoom.plot(t_with[mask_zoom_with], y_with[mask_zoom_with], 'ko', markersize=3, label="Messdaten (Zoom)")
            ax_zoom.plot(t_with[mask_zoom_with], y_with_smooth[mask_zoom_with], 'b-', linewidth=1, label="Geglättet (Zoom)")
            ax_zoom.axvline(retention_time_with - fwhm_with/2, color='gray', linestyle='--', linewidth=1, alpha=0.7)
            ax_zoom.axvline(retention_time_with + fwhm_with/2, color='gray', linestyle='--', linewidth=1, alpha=0.7)
            ax_zoom.set_xlabel("Zeit [min]")
            ax_zoom.set_ylabel("Signal")
            ax_zoom.legend(fontsize=8)
            ax_zoom.set_title("Mit Säule - Zoom")

            # Ergebnistext
            ax_text.axis('off')
            result_text = (
                "Ergebnisse:\n"
                "Mit Säule: PGF = " + f"{pgf_with:.2f}" + "\n\n"
                "FWHM (Säule): " + f"{fwhm_with:.4f} min\n"
                "Akzeptanzkriterium: 0.8 < PGF < 1.15"
            )
            ax_text.text(0.05, 0.5, result_text, transform=ax_text.transAxes,
                         fontsize=10, va="center", bbox=dict(facecolor='white', alpha=0.5))
            fig.tight_layout()
            measurement_figs.append(fig)
            plt.close(fig)

        # Van-Deemter-Plot (nur raw, da keine extrakolumnaren Daten vorliegen)
        fig_vd_raw, (ax_plot_raw, ax_text_raw) = plt.subplots(2, 1, figsize=(8.27, 11.69))
        fig_vd_raw.suptitle("van Deemter Plot (ohne extrakolumnare Korrektur)", fontsize=14)
        ax_plot_raw.plot(flow_rates, vanDeemter_H_raw, 'bo', label="Messpunkte (H, raw)")
        ax_plot_raw.set_xlabel("Flussrate (mL/min)")
        ax_plot_raw.set_ylabel("H (mm)")
        try:
            valid_mask_raw = ~np.isnan(vanDeemter_H_raw)
            flow_valid_raw = flow_rates[valid_mask_raw]
            H_valid_raw = np.array(vanDeemter_H_raw)[valid_mask_raw]
            popt_raw, pcov_raw = curve_fit(van_deemter_eq, flow_valid_raw, H_valid_raw, p0=[1, 1, 1])
            A_fit_raw, B_fit_raw, C_fit_raw = popt_raw
            perr_raw = np.sqrt(np.diag(pcov_raw))
            H_pred_raw = van_deemter_eq(flow_valid_raw, A_fit_raw, B_fit_raw, C_fit_raw)
            ss_res_raw = np.sum((H_valid_raw - H_pred_raw)**2)
            ss_tot_raw = np.sum((H_valid_raw - np.mean(H_valid_raw))**2)
            r2_raw = 1 - ss_res_raw/ss_tot_raw

            flow_dense_raw = np.linspace(flow_min, flow_max, 200)
            H_fit_dense_raw = van_deemter_eq(flow_dense_raw, A_fit_raw, B_fit_raw, C_fit_raw)
            ax_plot_raw.plot(flow_dense_raw, H_fit_dense_raw, 'r-', label="Fit: H = A + B/flow + C*flow")
            ax_plot_raw.legend()

            ax_text_raw.axis('off')
            param_text_raw = (
                "Van-Deemter-Fit (raw):\n"
                f"A = {A_fit_raw:.3e} ± {perr_raw[0]:.3e}\n"
                f"B = {B_fit_raw:.3e} ± {perr_raw[1]:.3e}\n"
                f"C = {C_fit_raw:.3e} ± {perr_raw[2]:.3e}\n\n"
                f"R² = {r2_raw:.4f}"
            )
            ax_text_raw.text(0.05, 0.9, param_text_raw, transform=ax_text_raw.transAxes, va='top',
                             fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
        except Exception as e:
            print("Fehler beim van-Deemter-Fit (raw):", e)

        with PdfPages(pdf_filename) as pdf:
            pdf.savefig(fig_vd_raw)  # Nur eine Seite: raw Plot
            for fig in measurement_figs:
                pdf.savefig(fig)
            # Ergebnisstabelle (nur Dateien mit Säule)
            fig_table, ax_table = plt.subplots(figsize=(8.27, 11.69))
            ax_table.axis('tight')
            ax_table.axis('off')
            table_data = [["Datei", "FWHM (Säule) [min]"]]
            for res in results:
                table_data.append([res["Datei"], f"{res['FWHM (Säule) [min]']:.4f}"])
            table = ax_table.table(cellText=table_data, loc="center", cellLoc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.scale(1, 1.5)
            ax_table.set_title("Übersicht der FWHM (Säule)", fontweight="bold")
            pdf.savefig(fig_table)
            plt.close('all')
        print(f"\nReport wurde erstellt: {pdf_filename}")

if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()
    files_with = filedialog.askopenfilenames(title="Wähle die Dateien aus (Messungen mit Säule)",
                                              filetypes=[("Textdateien", "*.txt"), ("Alle Dateien", "*.*")])
    files_with = list(files_with)
    if not files_with:
        print("Keine Dateien für Messungen mit Säule ausgewählt.")
        sys.exit(1)
    files_without = filedialog.askopenfilenames(title="Optional: Wähle die Dateien aus (Messungen ohne Säule)",
                                                 filetypes=[("Textdateien", "*.txt"), ("Alle Dateien", "*.*")])
    files_without = list(files_without)
    flow_min = float(simpledialog.askstring("Flussrate", "Minimale Flussrate (mL/min):", initialvalue="0.1"))
    flow_max = float(simpledialog.askstring("Flussrate", "Maximale Flussrate (mL/min):", initialvalue="2.0"))
    delta = float(simpledialog.askstring("Flussrate", "Schrittweite (mL/min):", initialvalue="0.1"))
    
    process_pairs(files_with, files_without, flow_min, flow_max, delta)
