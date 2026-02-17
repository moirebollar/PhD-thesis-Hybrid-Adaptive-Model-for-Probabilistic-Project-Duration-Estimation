"""
Report Generation Module.

Generates Excel and PDF reports with formatted tables, embedded charts,
and appendix-ready formatting for thesis inclusion.
"""

import io
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .monte_carlo import SimulationResults


def generate_excel_report(
    results: SimulationResults,
    activities_df: pd.DataFrame,
    tornado_df: Optional[pd.DataFrame] = None,
    criticality_df: Optional[pd.DataFrame] = None,
    scenario_df: Optional[pd.DataFrame] = None,
    bayesian_summary: Optional[pd.DataFrame] = None,
    output_path: Optional[str] = None,
) -> str:
    """
    Generate a comprehensive Excel report.

    Parameters
    ----------
    results : SimulationResults
    activities_df : pd.DataFrame
    tornado_df : pd.DataFrame, optional
    criticality_df : pd.DataFrame, optional
    scenario_df : pd.DataFrame, optional
    bayesian_summary : pd.DataFrame, optional
    output_path : str, optional

    Returns
    -------
    str : Path to generated Excel file.
    """
    if output_path is None:
        output_path = "simulation_report.xlsx"

    with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
        workbook = writer.book

        # Formats
        header_format = workbook.add_format({
            "bold": True, "bg_color": "#2F5496", "font_color": "white",
            "border": 1, "text_wrap": True, "align": "center",
        })
        number_format = workbook.add_format({"num_format": "0.0", "border": 1})
        pct_format = workbook.add_format({"num_format": "0.0%", "border": 1})
        title_format = workbook.add_format({
            "bold": True, "font_size": 14, "align": "center",
        })

        # --- Sheet 1: Summary ---
        ws_summary = workbook.add_worksheet("Summary")
        writer.sheets["Summary"] = ws_summary

        ws_summary.merge_range("A1:D1", "Monte Carlo Simulation Results", title_format)
        ws_summary.write("A3", "Metric", header_format)
        ws_summary.write("B3", "Value", header_format)

        stats = results.statistics
        summary_rows = [
            ("Number of Iterations", f"{results.n_iterations:,}"),
            ("Elapsed Time (s)", f"{results.elapsed_seconds:.1f}"),
            ("Converged", "Yes" if results.converged else "No"),
            ("", ""),
            ("Mean Duration (days)", f"{stats['mean']:.1f}"),
            ("Median Duration (days)", f"{stats['median']:.1f}"),
            ("Standard Deviation", f"{stats['std']:.1f}"),
            ("Coefficient of Variation", f"{stats['cv']:.3f}"),
            ("", ""),
            ("P10 (days)", f"{stats['P10']:.1f}"),
            ("P25 (days)", f"{stats['P25']:.1f}"),
            ("P50 (days)", f"{stats['P50']:.1f}"),
            ("P80 (days)", f"{stats['P80']:.1f}"),
            ("P90 (days)", f"{stats['P90']:.1f}"),
            ("P95 (days)", f"{stats['P95']:.1f}"),
            ("", ""),
            ("Minimum (days)", f"{stats['min']:.1f}"),
            ("Maximum (days)", f"{stats['max']:.1f}"),
        ]

        for row_idx, (metric, value) in enumerate(summary_rows, 3):
            ws_summary.write(row_idx, 0, metric)
            ws_summary.write(row_idx, 1, value)

        ws_summary.set_column("A:A", 30)
        ws_summary.set_column("B:B", 20)

        # --- Sheet 2: Activity Details ---
        activity_summary = results.summary_dataframe()
        activity_summary = activity_summary.merge(
            activities_df[["id", "name", "category", "a", "m", "b"]],
            left_on="activity_id",
            right_on="id",
            how="left",
        )
        activity_summary.to_excel(writer, sheet_name="Activity Details", index=False)

        ws_act = writer.sheets["Activity Details"]
        for col_idx, col_name in enumerate(activity_summary.columns):
            ws_act.write(0, col_idx, col_name, header_format)
        ws_act.set_column("A:Z", 15)

        # --- Sheet 3: Sensitivity (Tornado) ---
        if tornado_df is not None:
            tornado_df.to_excel(writer, sheet_name="Sensitivity", index=False)
            ws_sens = writer.sheets["Sensitivity"]
            for col_idx, col_name in enumerate(tornado_df.columns):
                ws_sens.write(0, col_idx, col_name, header_format)

        # --- Sheet 4: Criticality ---
        if criticality_df is not None:
            criticality_df.to_excel(writer, sheet_name="Criticality", index=False)
            ws_crit = writer.sheets["Criticality"]
            for col_idx, col_name in enumerate(criticality_df.columns):
                ws_crit.write(0, col_idx, col_name, header_format)

        # --- Sheet 5: Scenarios ---
        if scenario_df is not None:
            scenario_df.to_excel(writer, sheet_name="Scenarios", index=False)

        # --- Sheet 6: Bayesian Updates ---
        if bayesian_summary is not None:
            bayesian_summary.to_excel(writer, sheet_name="Bayesian Updates", index=False)

        # --- Sheet 7: Histogram Data ---
        hist_data = pd.DataFrame({
            "iteration": range(1, results.n_iterations + 1),
            "total_duration": results.total_durations,
        })
        hist_data.to_excel(writer, sheet_name="Raw Data", index=False)

        # Add chart to Summary sheet
        chart = workbook.add_chart({"type": "column"})
        chart.add_series({
            "name": "Duration Distribution",
            "categories": f"='Raw Data'!$A$2:$A${min(101, results.n_iterations + 1)}",
            "values": f"='Raw Data'!$B$2:$B${min(101, results.n_iterations + 1)}",
        })
        chart.set_title({"name": "Project Duration (first 100 iterations)"})
        chart.set_x_axis({"name": "Iteration"})
        chart.set_y_axis({"name": "Duration (days)"})
        ws_summary.insert_chart("D3", chart, {"x_scale": 1.5, "y_scale": 1.2})

    return output_path


def generate_pdf_report(
    results: SimulationResults,
    activities_df: pd.DataFrame,
    figures: Optional[dict] = None,
    tornado_df: Optional[pd.DataFrame] = None,
    criticality_df: Optional[pd.DataFrame] = None,
    output_path: Optional[str] = None,
    project_name: str = "Water Recovery Plant",
) -> str:
    """
    Generate a PDF summary report.

    Parameters
    ----------
    results : SimulationResults
    activities_df : pd.DataFrame
    figures : dict, optional
        {name: matplotlib.Figure} to embed in the report.
    tornado_df : pd.DataFrame, optional
    criticality_df : pd.DataFrame, optional
    output_path : str, optional
    project_name : str

    Returns
    -------
    str : Path to generated PDF.
    """
    from fpdf import FPDF

    if output_path is None:
        output_path = "simulation_report.pdf"

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    # --- Title Page ---
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 20)
    pdf.cell(0, 30, "", ln=True)
    pdf.cell(0, 15, "Monte Carlo Simulation Report", ln=True, align="C")
    pdf.set_font("Helvetica", "", 14)
    pdf.cell(0, 10, project_name, ln=True, align="C")
    pdf.cell(0, 10, "", ln=True)
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 8, f"Iterations: {results.n_iterations:,}", ln=True, align="C")
    pdf.cell(0, 8, f"Elapsed: {results.elapsed_seconds:.1f}s", ln=True, align="C")
    pdf.cell(0, 8,
             f"Converged: {'Yes' if results.converged else 'No'}",
             ln=True, align="C")

    # --- Percentile Table ---
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 12, "1. Summary Statistics", ln=True)
    pdf.ln(5)

    stats = results.statistics
    pdf.set_font("Helvetica", "B", 10)

    # Table header
    col_widths = [50, 40]
    pdf.cell(col_widths[0], 8, "Metric", border=1, align="C")
    pdf.cell(col_widths[1], 8, "Value (days)", border=1, align="C")
    pdf.ln()

    pdf.set_font("Helvetica", "", 10)
    table_rows = [
        ("Mean", f"{stats['mean']:.1f}"),
        ("Standard Deviation", f"{stats['std']:.1f}"),
        ("CV", f"{stats['cv']:.3f}"),
        ("P10", f"{stats['P10']:.1f}"),
        ("P25", f"{stats['P25']:.1f}"),
        ("P50 (Median)", f"{stats['P50']:.1f}"),
        ("P80", f"{stats['P80']:.1f}"),
        ("P90", f"{stats['P90']:.1f}"),
        ("P95", f"{stats['P95']:.1f}"),
    ]

    for metric, value in table_rows:
        pdf.cell(col_widths[0], 7, metric, border=1)
        pdf.cell(col_widths[1], 7, value, border=1, align="C")
        pdf.ln()

    # --- Buffer Recommendations ---
    pdf.ln(10)
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 12, "2. Buffer Recommendations", ln=True)
    pdf.ln(5)
    pdf.set_font("Helvetica", "", 10)

    p50 = stats["P50"]
    p80 = stats["P80"]
    p90 = stats["P90"]

    pdf.cell(0, 7, f"Deterministic estimate (most likely): {p50:.0f} days", ln=True)
    pdf.cell(0, 7, f"P80 buffer: +{p80 - p50:.0f} days (total: {p80:.0f} days)", ln=True)
    pdf.cell(0, 7, f"P90 buffer: +{p90 - p50:.0f} days (total: {p90:.0f} days)", ln=True)
    pdf.cell(0, 7, f"Recommended commitment: P80 = {p80:.0f} days", ln=True)

    # --- Embed Figures ---
    if figures:
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 12, "3. Charts", ln=True)

        for fig_name, fig in figures.items():
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                fig.savefig(tmp.name, dpi=150, bbox_inches="tight")
                pdf.ln(5)
                pdf.set_font("Helvetica", "B", 11)
                pdf.cell(0, 8, fig_name.replace("_", " ").title(), ln=True)
                pdf.image(tmp.name, x=15, w=180)
                if pdf.get_y() > 220:
                    pdf.add_page()

    # --- Sensitivity Table ---
    if tornado_df is not None:
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 12, "4. Sensitivity Ranking", ln=True)
        pdf.ln(5)

        pdf.set_font("Helvetica", "B", 9)
        sens_cols = [8, 60, 25, 25]
        pdf.cell(sens_cols[0], 7, "#", border=1, align="C")
        pdf.cell(sens_cols[1], 7, "Activity", border=1, align="C")
        pdf.cell(sens_cols[2], 7, "Correlation", border=1, align="C")
        pdf.cell(sens_cols[3], 7, "Category", border=1, align="C")
        pdf.ln()

        pdf.set_font("Helvetica", "", 9)
        for i, (_, row) in enumerate(tornado_df.head(15).iterrows()):
            pdf.cell(sens_cols[0], 6, str(i + 1), border=1, align="C")
            name = str(row.get("name", ""))[:40]
            pdf.cell(sens_cols[1], 6, name, border=1)
            pdf.cell(sens_cols[2], 6, f"{row['correlation']:.3f}", border=1, align="C")
            pdf.cell(sens_cols[3], 6, str(row.get("category", "")), border=1, align="C")
            pdf.ln()

    # --- Criticality Table ---
    if criticality_df is not None:
        pdf.ln(10)
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 12, "5. Criticality Index", ln=True)
        pdf.ln(5)

        pdf.set_font("Helvetica", "B", 9)
        crit_cols = [8, 60, 30, 30]
        pdf.cell(crit_cols[0], 7, "#", border=1, align="C")
        pdf.cell(crit_cols[1], 7, "Activity", border=1, align="C")
        pdf.cell(crit_cols[2], 7, "Criticality Index", border=1, align="C")
        pdf.cell(crit_cols[3], 7, "Mean Duration", border=1, align="C")
        pdf.ln()

        pdf.set_font("Helvetica", "", 9)
        for i, (_, row) in enumerate(criticality_df.iterrows()):
            pdf.cell(crit_cols[0], 6, str(i + 1), border=1, align="C")
            name = str(row.get("name", ""))[:40]
            pdf.cell(crit_cols[1], 6, name, border=1)
            pdf.cell(crit_cols[2], 6, f"{row['criticality_index']:.1%}", border=1, align="C")
            pdf.cell(crit_cols[3], 6, f"{row['mean_duration']:.1f}", border=1, align="C")
            pdf.ln()

    pdf.output(output_path)
    return output_path


def generate_report_to_bytes(
    results: SimulationResults,
    activities_df: pd.DataFrame,
    format: str = "excel",
    **kwargs,
) -> bytes:
    """
    Generate report and return as bytes (for Streamlit download).

    Parameters
    ----------
    format : str
        'excel' or 'pdf'.

    Returns
    -------
    bytes
    """
    with tempfile.NamedTemporaryFile(
        suffix=".xlsx" if format == "excel" else ".pdf",
        delete=False,
    ) as tmp:
        if format == "excel":
            generate_excel_report(results, activities_df, output_path=tmp.name, **kwargs)
        else:
            generate_pdf_report(results, activities_df, output_path=tmp.name, **kwargs)

        with open(tmp.name, "rb") as f:
            return f.read()
