"""
Streamlit Dashboard for Hybrid Adaptive Model.

Multi-tab interface for data input, simulation, analysis,
Bayesian updating, and report generation.
"""

import json
import io
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from src.data_ingest import (
    read_activity_excel, build_canonical_dataframe,
    classify_activity, generate_all_templates,
)
from src.distributions import create_distribution, BetaPERT, TriangularDist
from src.monte_carlo import (
    SimulationConfig, run_simulation, deterministic_cpm,
    build_precedence_network, SimulationResults,
)
from src.risks import RiskRegistry, RiskEvent, create_default_risks
from src.sensitivity import (
    tornado_analysis, criticality_analysis, scenario_analysis,
    significance_index,
)
from src.visualizations import (
    plot_histogram, plot_cdf, plot_tornado, plot_criticality,
    plot_comparison, plot_convergence,
    plotly_histogram, plotly_cdf, plotly_tornado,
    plotly_3d_surface, plotly_network, plotly_scenario_comparison,
)
from src.bayesian_updater import (
    ConjugateBayesianUpdater, adaptive_reestimation,
)
from src.ml_module import (
    DurationPredictor, build_training_data, create_synthetic_training_data,
)
from src.earned_schedule import (
    EarnedScheduleCalculator, compare_es_with_montecarlo,
    create_planned_schedule_from_cpm,
)
from src.reports import generate_report_to_bytes


# --- Page Config ---
st.set_page_config(
    page_title="Hybrid Adaptive Model - Project Duration Estimation",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Session State Initialization ---
if "activities_df" not in st.session_state:
    st.session_state.activities_df = None
if "risk_registry" not in st.session_state:
    st.session_state.risk_registry = RiskRegistry()
if "sim_results" not in st.session_state:
    st.session_state.sim_results = None
if "sim_config" not in st.session_state:
    st.session_state.sim_config = SimulationConfig()
if "ml_predictor" not in st.session_state:
    st.session_state.ml_predictor = None
if "bayesian_updater" not in st.session_state:
    st.session_state.bayesian_updater = None
if "tornado_df" not in st.session_state:
    st.session_state.tornado_df = None
if "criticality_df" not in st.session_state:
    st.session_state.criticality_df = None


def load_config():
    config_path = Path(__file__).parent / "config.json"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


# --- Sidebar ---
st.sidebar.title("Hybrid Adaptive Model")
st.sidebar.markdown("**Probabilistic Project Duration Estimation**")
st.sidebar.markdown("---")

tab_selection = st.sidebar.radio(
    "Navigation",
    [
        "Data Input",
        "Expert Input",
        "Risk Registry",
        "Simulation",
        "Analysis",
        "Adaptive Update",
        "Reports",
    ],
)

# --- Tab: Data Input ---
if tab_selection == "Data Input":
    st.title("Data Input")
    st.markdown("Upload project activity data or use sample data.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Upload Activity Data")
        uploaded_file = st.file_uploader(
            "Upload Excel file (.xlsx)", type=["xlsx", "xls"]
        )

        if uploaded_file is not None:
            try:
                df = read_activity_excel(uploaded_file)
                st.session_state.activities_df = df
                st.success(f"Loaded {len(df)} activities.")
            except Exception as e:
                st.error(f"Error reading file: {e}")

    with col2:
        st.subheader("Or Use Sample Data")
        if st.button("Load Sample Project (Water Recovery Plant)"):
            sample_data = {
                "id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "name": [
                    "Basic & detail engineering",
                    "Permitting & approvals",
                    "Site preparation",
                    "Civil works & foundations",
                    "Electromechanical assembly",
                    "Piping installation",
                    "Instrumentation & control",
                    "Testing & verification",
                    "Commissioning",
                    "Handover & closeout",
                ],
                "predecessors": [[], [1], [1], [3], [2, 4], [4], [5, 6], [7], [8], [9]],
                "dist_type": ["betapert"] * 10,
                "a": [20, 15, 5, 20, 25, 15, 10, 8, 10, 5],
                "m": [30, 25, 10, 30, 40, 22, 15, 12, 18, 7],
                "b": [45, 50, 18, 50, 65, 35, 25, 22, 35, 14],
                "category": [
                    "design", "permitting", "site_preparation", "civil_works",
                    "electromechanical", "piping", "instrumentation",
                    "testing", "commissioning", "startup",
                ],
            }
            st.session_state.activities_df = pd.DataFrame(sample_data)
            st.success("Sample project loaded with 10 activities.")

    # Display current data
    if st.session_state.activities_df is not None:
        st.subheader("Current Activity Data")
        st.dataframe(st.session_state.activities_df, use_container_width=True)

        # Editable table
        st.subheader("Edit Activities")
        edited_df = st.data_editor(
            st.session_state.activities_df,
            num_rows="dynamic",
            use_container_width=True,
            key="activity_editor",
        )
        if st.button("Save Edits"):
            st.session_state.activities_df = edited_df
            st.success("Activities updated.")

    # Download templates
    st.markdown("---")
    st.subheader("Download Templates")
    col_t1, col_t2, col_t3 = st.columns(3)

    with col_t1:
        if st.button("Generate Activity Template"):
            from src.data_ingest import create_activity_template
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
                create_activity_template(tmp.name)
                with open(tmp.name, "rb") as f:
                    st.download_button(
                        "Download activity_template.xlsx",
                        f.read(),
                        "activity_template.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    )

    with col_t2:
        if st.button("Generate Expert Questionnaire"):
            from src.data_ingest import create_expert_questionnaire
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
                create_expert_questionnaire(tmp.name)
                with open(tmp.name, "rb") as f:
                    st.download_button(
                        "Download expert_questionnaire.xlsx",
                        f.read(),
                        "expert_questionnaire.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    )

    with col_t3:
        if st.button("Generate Risk Registry Template"):
            from src.data_ingest import create_risk_registry_template
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
                create_risk_registry_template(tmp.name)
                with open(tmp.name, "rb") as f:
                    st.download_button(
                        "Download risk_registry.xlsx",
                        f.read(),
                        "risk_registry.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    )


# --- Tab: Expert Input ---
elif tab_selection == "Expert Input":
    st.title("Expert Input")
    st.markdown("Enter or upload expert estimates from Delphi questionnaire.")

    if st.session_state.activities_df is None:
        st.warning("Please load activity data first in the Data Input tab.")
    else:
        st.subheader("Upload Expert Questionnaire")
        expert_file = st.file_uploader(
            "Upload completed expert questionnaire (.xlsx)", type=["xlsx"]
        )
        if expert_file is not None:
            try:
                expert_df = pd.read_excel(expert_file, sheet_name="Duration Estimates",
                                          engine="openpyxl")
                st.dataframe(expert_df, use_container_width=True)
                st.info("Expert data loaded. You can integrate it with the activity estimates below.")
            except Exception as e:
                st.error(f"Error: {e}")

        st.subheader("Manual Expert Entry")
        st.markdown("Adjust three-point estimates based on expert judgment:")

        df = st.session_state.activities_df
        for idx, row in df.iterrows():
            with st.expander(f"Activity {row['id']}: {row['name']}"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    new_a = st.number_input(
                        f"Optimistic (a)", value=float(row["a"]),
                        min_value=0.1, key=f"expert_a_{idx}"
                    )
                with col2:
                    new_m = st.number_input(
                        f"Most Likely (m)", value=float(row["m"]),
                        min_value=0.1, key=f"expert_m_{idx}"
                    )
                with col3:
                    new_b = st.number_input(
                        f"Pessimistic (b)", value=float(row["b"]),
                        min_value=0.1, key=f"expert_b_{idx}"
                    )
                df.at[idx, "a"] = new_a
                df.at[idx, "m"] = new_m
                df.at[idx, "b"] = new_b

        if st.button("Update Activity Estimates"):
            st.session_state.activities_df = df
            st.success("Estimates updated with expert input.")


# --- Tab: Risk Registry ---
elif tab_selection == "Risk Registry":
    st.title("Risk Registry")

    col1, col2 = st.columns([2, 1])

    with col2:
        if st.button("Load Default Risks (Thesis Example)"):
            st.session_state.risk_registry = create_default_risks()
            st.success("Loaded 5 default risks.")

        uploaded_risk = st.file_uploader("Upload Risk Registry (.xlsx)", type=["xlsx"])
        if uploaded_risk is not None:
            try:
                st.session_state.risk_registry.load_from_excel(uploaded_risk)
                st.success("Risks loaded from file.")
            except Exception as e:
                st.error(f"Error: {e}")

    with col1:
        # Display current risks
        if st.session_state.risk_registry.risks:
            risk_df = st.session_state.risk_registry.to_dataframe()
            st.dataframe(risk_df, use_container_width=True)

            summary = st.session_state.risk_registry.summary()
            st.metric("Active Risks", summary["active_risks"])
            st.metric("Expected Total Impact",
                      f"{summary['expected_total_impact']:.1f} days")
        else:
            st.info("No risks defined yet. Load defaults or upload a registry.")

    # Add new risk
    st.subheader("Add New Risk")
    with st.form("new_risk_form"):
        c1, c2 = st.columns(2)
        with c1:
            risk_id = st.text_input("Risk ID", value="R_new")
            risk_name = st.text_input("Risk Name")
            risk_prob = st.slider("Probability", 0.0, 1.0, 0.2)
            risk_category = st.selectbox(
                "Category",
                ["technical", "environmental", "regulatory", "supply_chain", "residual"],
            )
        with c2:
            impact_dist = st.selectbox("Impact Distribution", ["triangular", "betapert", "uniform"])
            impact_min = st.number_input("Impact Min (days)", value=5.0)
            impact_mode = st.number_input("Impact Most Likely (days)", value=15.0)
            impact_max = st.number_input("Impact Max (days)", value=30.0)
            applies_to = st.text_input("Affected Activities (comma-separated IDs or 'all')", value="all")

        if st.form_submit_button("Add Risk"):
            try:
                new_risk = RiskEvent(
                    risk_id=risk_id, name=risk_name, probability=risk_prob,
                    impact_dist_type=impact_dist,
                    impact_min=impact_min, impact_mode=impact_mode, impact_max=impact_max,
                    applies_to=applies_to, category=risk_category,
                )
                st.session_state.risk_registry.add_risk(new_risk)
                st.success(f"Risk '{risk_name}' added.")
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")


# --- Tab: Simulation ---
elif tab_selection == "Simulation":
    st.title("Monte Carlo Simulation")

    if st.session_state.activities_df is None:
        st.warning("Please load activity data first.")
    else:
        # Configuration
        st.subheader("Simulation Configuration")
        col1, col2, col3 = st.columns(3)

        with col1:
            n_iter = st.number_input(
                "Number of Iterations", min_value=1000, max_value=100000,
                value=st.session_state.sim_config.n_iterations, step=1000,
            )
        with col2:
            seed = st.number_input(
                "Random Seed", min_value=0, value=42,
            )
        with col3:
            pert_lambda = st.number_input(
                "PERT Lambda", min_value=1.0, max_value=10.0, value=4.0,
            )

        include_risks = st.checkbox(
            "Include Risk Events",
            value=bool(st.session_state.risk_registry.risks),
        )

        use_ml = st.checkbox("Apply ML Bias Factors", value=False)
        ml_bias = None
        if use_ml and st.session_state.ml_predictor is not None:
            ml_bias = st.session_state.ml_predictor.predict_bias_factors(
                st.session_state.activities_df
            )
            st.info(f"ML bias factors applied for {len(ml_bias)} activities.")

        # Deterministic CPM
        st.subheader("Deterministic CPM Baseline")
        try:
            det_duration, critical_path, cpm_schedule = deterministic_cpm(
                st.session_state.activities_df
            )
            st.metric("Deterministic Duration (CPM)", f"{det_duration:.0f} days")
            st.write(f"Critical Path: {' -> '.join(str(a) for a in critical_path)}")

            with st.expander("CPM Schedule Details"):
                st.dataframe(cpm_schedule, use_container_width=True)
        except Exception as e:
            st.error(f"CPM Error: {e}")

        # Run simulation
        st.subheader("Run Simulation")
        if st.button("Run Monte Carlo Simulation", type="primary"):
            config = SimulationConfig(
                n_iterations=n_iter,
                random_seed=seed,
                pert_lambda=pert_lambda,
            )

            risk_reg = st.session_state.risk_registry if include_risks else None

            progress_bar = st.progress(0)
            status_text = st.empty()

            def update_progress(iteration, total):
                pct = iteration / total
                progress_bar.progress(pct)
                status_text.text(f"Iteration {iteration:,} / {total:,}")

            with st.spinner("Running simulation..."):
                results = run_simulation(
                    st.session_state.activities_df,
                    risk_registry=risk_reg,
                    config=config,
                    ml_bias_factors=ml_bias,
                    progress_callback=update_progress,
                )

            progress_bar.progress(1.0)
            status_text.text(f"Completed {n_iter:,} iterations in {results.elapsed_seconds:.1f}s")

            st.session_state.sim_results = results
            st.session_state.sim_config = config

            # Display results
            st.subheader("Results")
            col_r1, col_r2, col_r3, col_r4 = st.columns(4)
            stats = results.statistics
            col_r1.metric("P50", f"{stats['P50']:.0f} days")
            col_r2.metric("P80", f"{stats['P80']:.0f} days")
            col_r3.metric("P90", f"{stats['P90']:.0f} days")
            col_r4.metric("Mean", f"{stats['mean']:.0f} days")

            # Charts
            tab_hist, tab_cdf, tab_conv = st.tabs(["Histogram", "S-Curve (CDF)", "Convergence"])

            with tab_hist:
                fig = plotly_histogram(results)
                st.plotly_chart(fig, use_container_width=True)

            with tab_cdf:
                fig = plotly_cdf(results)
                st.plotly_chart(fig, use_container_width=True)

            with tab_conv:
                fig = plot_convergence(results)
                st.pyplot(fig)

        # Show previous results if available
        elif st.session_state.sim_results is not None:
            results = st.session_state.sim_results
            st.subheader("Previous Simulation Results")
            stats = results.statistics
            col_r1, col_r2, col_r3, col_r4 = st.columns(4)
            col_r1.metric("P50", f"{stats['P50']:.0f} days")
            col_r2.metric("P80", f"{stats['P80']:.0f} days")
            col_r3.metric("P90", f"{stats['P90']:.0f} days")
            col_r4.metric("Mean", f"{stats['mean']:.0f} days")


# --- Tab: Analysis ---
elif tab_selection == "Analysis":
    st.title("Analysis")

    if st.session_state.sim_results is None:
        st.warning("Please run a simulation first.")
    else:
        results = st.session_state.sim_results
        activities_df = st.session_state.activities_df

        tab_sens, tab_crit, tab_scenario, tab_network = st.tabs([
            "Sensitivity (Tornado)", "Criticality", "Scenarios", "Network"
        ])

        with tab_sens:
            st.subheader("Tornado Diagram")
            torn = tornado_analysis(results, activities_df)
            st.session_state.tornado_df = torn
            fig = plotly_tornado(torn)
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Significance Index")
            sig = significance_index(results, activities_df)
            st.dataframe(sig, use_container_width=True)

        with tab_crit:
            st.subheader("Criticality Index")
            crit = criticality_analysis(results, activities_df)
            st.session_state.criticality_df = crit
            fig = plot_criticality(crit)
            st.pyplot(fig)
            st.dataframe(crit, use_container_width=True)

        with tab_scenario:
            st.subheader("Scenario Analysis")
            scenarios = {
                "Base Case": {},
                "Regulatory Change (+15%)": {"duration_factor": 1.15},
                "Extreme Weather (2x risk)": {"risk_probability_factor": 2.0},
                "Combined Adverse": {
                    "duration_factor": 1.15,
                    "risk_probability_factor": 2.0,
                },
            }

            if st.button("Run Scenario Analysis"):
                with st.spinner("Running scenarios..."):
                    scenario_config = SimulationConfig(
                        n_iterations=5000,
                        random_seed=st.session_state.sim_config.random_seed,
                    )
                    scenario_df = scenario_analysis(
                        activities_df, scenarios,
                        st.session_state.risk_registry, scenario_config,
                    )
                fig = plotly_scenario_comparison(scenario_df)
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(scenario_df, use_container_width=True)

        with tab_network:
            st.subheader("Activity Precedence Network")
            ci = results.criticality_index
            fig = plotly_network(activities_df, criticality_index=ci)
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("3D Surface: Buffer vs Probability vs Cost")
            fig_3d = plotly_3d_surface(results)
            st.plotly_chart(fig_3d, use_container_width=True)


# --- Tab: Adaptive Update ---
elif tab_selection == "Adaptive Update":
    st.title("Adaptive Update (Bayesian + Earned Schedule)")

    if st.session_state.activities_df is None:
        st.warning("Please load activity data first.")
    else:
        activities_df = st.session_state.activities_df

        st.subheader("Enter Actual Progress Data")
        st.markdown("For completed activities, enter the actual duration observed.")

        observations = {}
        for _, row in activities_df.iterrows():
            col1, col2, col3 = st.columns([3, 2, 2])
            with col1:
                st.write(f"**{row['id']}**: {row['name']}")
            with col2:
                status = st.selectbox(
                    "Status", ["Not started", "In progress", "Completed"],
                    key=f"status_{row['id']}"
                )
            with col3:
                if status == "Completed":
                    actual = st.number_input(
                        "Actual (days)", min_value=0.1,
                        value=float(row["m"]),
                        key=f"actual_{row['id']}"
                    )
                    observations[row["id"]] = actual

        if observations and st.button("Run Adaptive Re-estimation", type="primary"):
            with st.spinner("Updating model..."):
                result = adaptive_reestimation(
                    activities_df,
                    completed_observations=observations,
                    risk_registry=st.session_state.risk_registry,
                    sim_config=st.session_state.sim_config,
                    method="conjugate",
                )

            st.session_state.sim_results = result["simulation_results"]

            st.subheader("Update Results")
            comp = result["comparison"]
            col1, col2, col3 = st.columns(3)
            col1.metric(
                "Updated P50",
                f"{comp['updated']['P50']:.0f} days",
                f"{comp['delta_P50']:+.0f}",
            )
            col2.metric(
                "Updated P80",
                f"{comp['updated']['P80']:.0f} days",
            )
            col3.metric(
                "Updated P90",
                f"{comp['updated']['P90']:.0f} days",
                f"{comp['delta_P90']:+.0f}",
            )

            st.subheader("Prior vs Posterior Comparison")
            st.dataframe(result["updater_summary"], use_container_width=True)

            # Updated histogram
            fig = plotly_histogram(result["simulation_results"],
                                   title="Updated Duration Distribution")
            st.plotly_chart(fig, use_container_width=True)

        # ML Training section
        st.markdown("---")
        st.subheader("Machine Learning Model")

        if st.button("Train ML Model (Synthetic Data)"):
            with st.spinner("Generating synthetic data and training..."):
                synthetic = create_synthetic_training_data(activities_df)
                training = build_training_data(synthetic)

                predictor = DurationPredictor(model_type="random_forest")
                metrics = predictor.train(training)
                st.session_state.ml_predictor = predictor

            st.success("ML model trained.")
            st.json(metrics)

            importance = predictor.get_feature_importance()
            if importance is not None:
                st.subheader("Feature Importance")
                st.dataframe(importance, use_container_width=True)

        # Earned Schedule section
        st.markdown("---")
        st.subheader("Earned Schedule Metrics")

        if observations and st.session_state.sim_results is not None:
            try:
                _, _, cpm_sched = deterministic_cpm(activities_df)
                planned_sched = create_planned_schedule_from_cpm(activities_df, cpm_sched)
                det_dur, _, _ = deterministic_cpm(activities_df)

                es_calc = EarnedScheduleCalculator(planned_sched, det_dur)

                current_time = st.number_input(
                    "Current project time (days from start)",
                    min_value=1.0, value=60.0,
                )

                es_metrics = es_calc.calculate_es(
                    actual_time=current_time,
                    completed_activities={aid: current_time for aid in observations},
                )

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Earned Schedule", f"{es_metrics['earned_schedule']:.1f} days")
                col2.metric("SV(t)", f"{es_metrics['SV_t']:.1f} days")
                col3.metric("SPI(t)", f"{es_metrics['SPI_t']:.2f}")
                col4.metric("IEAC(t)", f"{es_metrics['IEAC_t']:.0f} days")

                st.info(f"Status: {es_metrics['status']}")

                # Compare with Monte Carlo
                comparison = compare_es_with_montecarlo(
                    es_metrics, st.session_state.sim_results.statistics
                )
                st.subheader("ES vs Monte Carlo Comparison")
                st.json(comparison)
            except Exception as e:
                st.error(f"Earned Schedule calculation error: {e}")


# --- Tab: Reports ---
elif tab_selection == "Reports":
    st.title("Report Generation")

    if st.session_state.sim_results is None:
        st.warning("Please run a simulation first.")
    else:
        results = st.session_state.sim_results
        activities_df = st.session_state.activities_df

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Excel Report")
            if st.button("Generate Excel Report"):
                with st.spinner("Generating..."):
                    excel_bytes = generate_report_to_bytes(
                        results, activities_df, format="excel",
                        tornado_df=st.session_state.tornado_df,
                        criticality_df=st.session_state.criticality_df,
                    )
                st.download_button(
                    "Download Excel Report",
                    excel_bytes,
                    "simulation_report.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )

        with col2:
            st.subheader("PDF Report")
            project_name = st.text_input("Project Name", value="Water Recovery Plant")
            if st.button("Generate PDF Report"):
                with st.spinner("Generating..."):
                    # Generate static figures for PDF
                    figures = {
                        "histogram": plot_histogram(results),
                        "s_curve": plot_cdf(results),
                        "convergence": plot_convergence(results),
                    }
                    if st.session_state.tornado_df is not None:
                        figures["tornado"] = plot_tornado(st.session_state.tornado_df)
                    if st.session_state.criticality_df is not None:
                        figures["criticality"] = plot_criticality(
                            st.session_state.criticality_df
                        )

                    pdf_bytes = generate_report_to_bytes(
                        results, activities_df, format="pdf",
                        figures=figures,
                        tornado_df=st.session_state.tornado_df,
                        criticality_df=st.session_state.criticality_df,
                        project_name=project_name,
                    )
                st.download_button(
                    "Download PDF Report",
                    pdf_bytes,
                    "simulation_report.pdf",
                    mime="application/pdf",
                )

        # Quick comparison chart
        st.markdown("---")
        st.subheader("Quick Comparison")

        actual_duration = st.number_input(
            "Actual project duration (if known, for comparison)", value=0.0
        )

        stats = results.statistics
        try:
            det_dur, _, _ = deterministic_cpm(activities_df)
        except Exception:
            det_dur = stats["P50"]

        comp_data = {"PERT/CPM": det_dur, "MC P50": stats["P50"],
                     "MC P80": stats["P80"], "MC P90": stats["P90"]}
        if actual_duration > 0:
            comp_data["Actual"] = actual_duration

        fig = plot_comparison(comp_data)
        st.pyplot(fig)
