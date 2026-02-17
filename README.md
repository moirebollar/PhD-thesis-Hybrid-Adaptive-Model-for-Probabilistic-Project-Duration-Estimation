# Hybrid Adaptive Model for Probabilistic Project Duration Estimation

Modelo hibrido adaptativo para la estimacion probabilistica del tiempo de proyectos de alta incertidumbre en empresas de desarrollo tecnologico.

---

## Quick Start

**Important:** You cannot run this app by double-clicking `app.py`. It must be launched from the terminal using the `streamlit` command.

### Step 1: Open a terminal

Press `Win + R`, type `cmd`, and press Enter. Or open Windows Terminal / PowerShell.

### Step 2: Navigate to the project folder

```
cd C:\Users\rodri\Documents\Claude\Thesis\Code
```

### Step 3: Activate the virtual environment

```
venv\Scripts\activate
```

You should see `(venv)` appear at the beginning of your terminal line. This means the environment is active and all libraries are available.

### Step 4: Launch the app

```
streamlit run app.py
```

A browser tab will open automatically at `http://localhost:8501` with the dashboard.

To stop the app, go back to the terminal and press `Ctrl + C`.

---

## What This System Does

This system estimates how long a construction project will take using probabilistic methods instead of a single fixed number. It combines four techniques:

1. **Monte Carlo Simulation** - Runs thousands of random scenarios to produce a probability distribution of project duration
2. **Risk Analysis** - Models discrete risk events (weather, regulatory changes, etc.) that may or may not occur
3. **Bayesian Updating** - When you get real progress data during execution, the model updates its predictions
4. **Machine Learning** - Uses historical project data to learn patterns and correct systematic biases

The output is not "the project will take 250 days" but rather "there is a 50% chance it finishes in 210 days, 80% chance in 240 days, and 90% chance in 265 days."

---

## Dashboard Tabs (How to Use)

The app has 7 tabs accessible from the left sidebar:

### 1. Data Input

This is where you load your project activities.

- **Upload Excel**: Click "Upload Excel file" and select your project's Gantt chart or activity list exported from Excel. The system auto-detects columns (works with Spanish and English column names).
- **Use Sample Data**: Click "Load Sample Project" to load a built-in 10-activity water recovery plant example. This is the best way to start and explore the system.
- **Edit Activities**: You can modify activity names, durations, and predecessors directly in the table.

Each activity needs:
- **ID**: A unique number
- **Name**: Activity description
- **Predecessors**: Which activities must finish before this one starts (comma-separated IDs)
- **a, m, b**: Optimistic, Most Likely, Pessimistic duration estimates (in days)
- **Distribution type**: `betapert` (recommended) or `triangular`

You can also download blank Excel templates to fill in offline.

### 2. Expert Input

For entering expert judgment from Delphi questionnaires.

- Upload a completed questionnaire Excel file, or
- Manually adjust the three-point estimates (a, m, b) for each activity using the sliders
- Click "Update Activity Estimates" to save changes

### 3. Risk Registry

Define risk events that could affect the project.

- **Load Default Risks**: Loads 5 example risks from the thesis (unexpected contaminant, extreme weather, equipment delays, regulatory change, unidentified reserve)
- **Upload Risk Registry**: Load risks from an Excel file
- **Add New Risk**: Fill in the form with probability (0-1), impact distribution, affected activities, etc.

Each risk has a probability of occurring and an impact distribution. During simulation, the system randomly decides if each risk fires in each iteration.

### 4. Simulation

The core of the system. This is where you run the Monte Carlo simulation.

**Configuration:**
- **Number of Iterations**: 10,000 is good for exploration, 50,000 for final results
- **Random Seed**: Set to any number for reproducible results
- **PERT Lambda**: Controls how much weight the "most likely" estimate gets (4 is standard)
- **Include Risk Events**: Check this to include risks from the registry
- **Apply ML Bias Factors**: Check this only after training the ML model in the Adaptive tab

**What happens when you click "Run":**
1. The system builds a precedence network from your activities
2. For each iteration, it samples a random duration for every activity from its probability distribution
3. It calculates the critical path (longest path through the network)
4. If risks are enabled, it randomly triggers risk events and adds delays
5. It records the total project duration for that iteration
6. After all iterations, it computes percentiles

**Results shown:**
- **P50**: 50% chance of finishing within this many days (median)
- **P80**: 80% chance (common for contractual commitments)
- **P90**: 90% chance (conservative estimate)
- **Histogram**: Shape of the duration distribution
- **S-Curve (CDF)**: Read off the probability of finishing before any given date
- **Convergence**: Shows that percentiles stabilize as iterations increase

**CPM Baseline**: Before running Monte Carlo, the tab shows the deterministic Critical Path Method result using only the "most likely" values. This is what traditional project management gives you - a single number with no uncertainty information.

### 5. Analysis

Available after running a simulation. Four sub-tabs:

- **Sensitivity (Tornado)**: Shows which activities have the most influence on total duration. A high Spearman correlation means that when this activity takes longer, the whole project takes longer. Focus your risk management here.
- **Criticality**: Shows how often each activity appears on the critical path across all iterations. An activity with 95% criticality is almost always on the critical path.
- **Scenarios**: Compares "what-if" scenarios (base case, regulatory change +15%, extreme weather 2x risk, combined adverse). Click "Run Scenario Analysis" to compute.
- **Network**: Interactive visualization of the activity precedence network, color-coded by criticality. Also includes a 3D surface plot of buffer vs. probability vs. cost.

### 6. Adaptive Update

This is the "adaptive" part of the model. Use it when your project is underway and you have actual progress data.

**Bayesian Updating:**
1. For each completed activity, set its status to "Completed" and enter the actual duration
2. Click "Run Adaptive Re-estimation"
3. The system updates its beliefs (narrows uncertainty for completed activities, adjusts remaining ones) and re-runs the simulation
4. You see how the P50/P80/P90 changed compared to the original estimate

**ML Model Training:**
- Click "Train ML Model" to train a Random Forest on synthetic data derived from your project
- Once trained, you can enable "Apply ML Bias Factors" in the Simulation tab
- The ML model predicts whether activities tend to run longer or shorter than estimated

**Earned Schedule:**
- After entering progress data, enter the current project time
- The system calculates ES, SPI(t), and IEAC(t) metrics
- It compares the Earned Schedule prediction with the Monte Carlo prediction

### 7. Reports

Generate downloadable reports:

- **Excel Report**: Multi-sheet workbook with summary statistics, activity details, sensitivity ranking, criticality, and raw simulation data
- **PDF Report**: Formatted document with tables and charts, ready for thesis appendix inclusion

Enter the actual project duration (if known) to see a comparison bar chart of PERT/CPM vs Monte Carlo vs Actual.

---

## Project Structure

```
Code/
|-- app.py                    # Streamlit dashboard (launch with: streamlit run app.py)
|-- requirements.txt          # Python package dependencies
|-- config.json               # Default simulation parameters
|-- venv/                     # Python virtual environment (already set up)
|-- src/
|   |-- data_ingest.py        # Loading and normalizing Excel data
|   |-- distributions.py      # Triangular and BetaPERT probability distributions
|   |-- monte_carlo.py        # Monte Carlo simulation engine + CPM
|   |-- risks.py              # Risk events and risk registry
|   |-- sensitivity.py        # Tornado, criticality, and scenario analysis
|   |-- bayesian_updater.py   # Bayesian posterior updating (conjugate + MCMC)
|   |-- ml_module.py          # Random Forest / Neural Network prediction
|   |-- earned_schedule.py    # Earned Schedule metrics (ES, SPI(t), IEAC(t))
|   |-- visualizations.py     # All charts (matplotlib + plotly)
|   |-- reports.py            # Excel and PDF report generation
|-- data/
|   |-- templates/            # Blank Excel templates for data collection
|   |-- historical/           # Place your 3 historical project Excel files here
|   |-- sample/               # Sample CSV data for testing
|-- tests/                    # Automated tests (78 tests, all passing)
```

---

## Running Tests

```
cd C:\Users\rodri\Documents\Claude\Thesis\Code
venv\Scripts\activate
pytest tests/ -v
```

This runs 78 automated tests covering distributions, Monte Carlo, Bayesian updating, and ML modules.

---

## Typical Workflow

1. **Prepare data**: Fill in `activity_template.xlsx` with your project activities, or use the sample data
2. **Load data**: Open the app, go to Data Input, upload or load sample
3. **Define risks**: Go to Risk Registry, load defaults or add your own
4. **Run simulation**: Go to Simulation tab, configure iterations, click Run
5. **Analyze results**: Go to Analysis tab, review tornado diagram and criticality
6. **During execution**: As activities complete, go to Adaptive Update, enter actual durations, re-estimate
7. **Generate reports**: Go to Reports tab, download Excel or PDF

---

## Common Issues

**"The app closes immediately when I double-click app.py"**
This is expected. Streamlit apps must be launched from the terminal with `streamlit run app.py`. Double-clicking runs it as a regular Python script, which has no effect.

**"Module not found" errors**
Make sure you activated the virtual environment first: `venv\Scripts\activate`

**"Port 8501 already in use"**
Another Streamlit instance is already running. Either use that one (open `http://localhost:8501` in your browser) or stop it with `Ctrl+C` in its terminal window.

**Charts not showing**
Try refreshing the browser page. If using a very old browser, update to a recent version of Chrome, Firefox, or Edge.

---

## Key Formulas

**BetaPERT Mean:** `mu = (a + 4*m + b) / 6`

**BetaPERT Std Dev:** `sigma = (b - a) / 6`

**Criticality Index:** `CI = (times on critical path) / (total iterations)`

**Earned Schedule:** `ES = time when Planned Value = Earned Value`

**SPI(t):** `SPI(t) = ES / AT` (> 1 = ahead, < 1 = behind)

**IEAC(t):** `IEAC(t) = AT + (PD - ES) / SPI(t)`
