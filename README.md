# Topology-Aware Financial Decision Making — reproducible pipeline

Single-file replacement for the three `TDA_Stress_GC_NotebookV2*.ipynb` notebooks.
SPY and BTC both run from `tda_stress_gc.py`.

**The internet is needed exactly once.** After `--stage fetch`, every other stage
— backtest, grid search, robustness, walk-forward, ML — runs with the network
disconnected, and refuses to touch it.

---

## 1. Setup in VS Code

### Install VS Code + the Python extension

Install VS Code, then open the Extensions panel (`Ctrl+Shift+X` / `Cmd+Shift+X`)
and install **Python** by Microsoft. That one extension pulls in Pylance and the
debugger.

### Open the project

Put these four files in one folder, e.g. `~/tda-finance/`:

```
tda-finance/
├── tda_stress_gc.py
├── requirements.txt
├── README.md
└── MANUSCRIPT_EDITS.md
```

**File → Open Folder…** and select `tda-finance`. Open the *folder*, not the
individual file — VS Code resolves the interpreter and the working directory from
the folder, and `data/raw/` and `results/` are created relative to it.

### Create the virtual environment

Open the integrated terminal (`` Ctrl+` ``) and run:

```bash
python -m venv venv
```

Then activate it:

| shell | command |
|---|---|
| Linux / macOS | `source venv/bin/activate` |
| Windows PowerShell | `.\venv\Scripts\Activate.ps1` |
| Windows CMD | `venv\Scripts\activate.bat` |

If PowerShell refuses with an execution-policy error, run once:
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

Install dependencies:

```bash
pip install -r requirements.txt
```

### Point VS Code at the venv

`Ctrl+Shift+P` → **Python: Select Interpreter** → choose the one showing
`./venv/bin/python` (or `.\venv\Scripts\python.exe`). The status bar should now
read `Python 3.x ('venv')`. If it doesn't, new terminals will use the system
Python and imports will fail.

### Verify

```bash
python tda_stress_gc.py --smoke
```

Roughly two minutes, no network. Runs 11 unit checks, the Wagner et al.
ground-truth mixup suite, an independent cross-validation of the Algorithm-2
reduction against Ripser, and the whole pipeline end-to-end on synthetic prices.
You want `unit checks : PASS`, `mixup : PASS`, `pipeline : PASS`.

Do this before anything else. If the smoke test fails, nothing downstream is
trustworthy.

---

## 2. The one-time download

**With the internet connected:**

```bash
python tda_stress_gc.py --stage fetch --asset all
```

This writes:

```
data/raw/
├── SPY_2010-01-01_2024-12-31.csv
├── BTC-USD_2017-01-01_2024-12-31.csv
└── manifest.json
```

`manifest.json` records, for each file, a SHA-256 digest, the row count and the
observed first/last date. Every later run verifies the CSV against it and logs

```
Cache verified: SPY_2010-01-01_2024-12-31.csv (3773 rows, 2010-01-04 -> 2024-12-30, sha256 a3f1c8...)
```

If the file ever changes you get a loud `CACHE CHANGED` warning instead of
silently different numbers. **Commit `data/raw/` to your repository** — those CSVs
plus `results/config_used.json` are what let a referee reproduce the paper.

### Now disconnect

Every analysis stage calls the loader with `offline=True` by default. If the
cache is missing it does **not** fall back to a download — it stops and tells you
what to run:

```
No cached data for SPY at data/raw/SPY_2010-01-01_2024-12-31.csv.
This run is offline by design. Connect once and run:

    python tda_stress_gc.py --stage fetch --asset all
```

That is deliberate: a silent re-download would pull revised prices and change
your published numbers without telling you. Pass `--allow-download` only if you
genuinely want an analysis stage to fetch missing data.

To deliberately refresh (e.g. extending the sample), reconnect and run
`--stage fetch --force-refetch`, then re-run everything and update the tables.

---

## 3. Running the analysis (offline)

```bash
python tda_stress_gc.py --asset all --stage all --jobs -1
```

Stages individually:

| command | produces |
|---|---|
| `--stage backtest` | `stress_scores.csv`, `positions.csv`, `strategy_comparison.csv` → **Tables 1–2** |
| `--stage grid` | `grid_search.csv`, `fig_grid_*.png` → **Figures 3–5** |
| `--stage robustness` | four sweep CSVs + `robustness_summary.json` → **Section 6.5** |
| `--stage walkforward` | fold metrics, Diebold–Mariano, Holm–Bonferroni, bootstrap CIs → **new Section 6.6** |
| `--stage ml` | `ml_experiment.csv` (not reported in the manuscript) |

Add `--asset spy` or `--asset btc` to run one asset. Outputs land in
`results/<asset>/{tables,figures}/`, and the exact configuration is written to
`results/config_used.json`.

**The grid search is the long one** — 72 configurations per asset, each computing
homology for every trading day. Run it overnight with `--jobs -1`. It needs no
network whatsoever.

### Running it inside VS Code

Three options, in increasing order of convenience:

1. **Terminal** — the commands above in the integrated terminal. Best for the
   long grid run.
2. **Run button** — the ▷ in the top right runs the file with no arguments, which
   is `--asset all --stage all`. Fine for a full run, no good for selecting a
   stage.
3. **Interactive window** — Command Palette → **Jupyter: Run Current File in
   Interactive Window** keeps variables alive so you can poke at intermediate
   results, which is closest to how the original notebooks felt.

For repeatable stage selection, create `.vscode/launch.json`:

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Fetch data (needs internet)",
      "type": "debugpy", "request": "launch",
      "program": "${workspaceFolder}/tda_stress_gc.py",
      "args": ["--stage", "fetch", "--asset", "all"],
      "console": "integratedTerminal"
    },
    {
      "name": "Smoke test",
      "type": "debugpy", "request": "launch",
      "program": "${workspaceFolder}/tda_stress_gc.py",
      "args": ["--smoke"],
      "console": "integratedTerminal"
    },
    {
      "name": "Backtest (both assets)",
      "type": "debugpy", "request": "launch",
      "program": "${workspaceFolder}/tda_stress_gc.py",
      "args": ["--asset", "all", "--stage", "backtest", "--jobs", "-1"],
      "console": "integratedTerminal"
    },
    {
      "name": "Grid search (slow)",
      "type": "debugpy", "request": "launch",
      "program": "${workspaceFolder}/tda_stress_gc.py",
      "args": ["--asset", "all", "--stage", "grid", "--jobs", "-1"],
      "console": "integratedTerminal"
    },
    {
      "name": "Robustness sweeps",
      "type": "debugpy", "request": "launch",
      "program": "${workspaceFolder}/tda_stress_gc.py",
      "args": ["--asset", "all", "--stage", "robustness", "--jobs", "-1"],
      "console": "integratedTerminal"
    }
  ]
}
```

Then pick a configuration from the Run and Debug panel (`Ctrl+Shift+D`) and press
F5. Breakpoints work — useful for stepping into `mixup_triples` or
`score_components`.

---

## 4. Configuration

Everything lives in the `Config` and `AssetConfig` dataclasses at the top of the
file. The per-asset embedding parameters are the argmax of the in-sample Sharpe
grid search and are declared as such:

| asset | window `w` | dimension `d` | delay `tau` |
|---|---|---|---|
| SPY | 90 | 4 | 3 |
| BTC-USD | 60 | 4 | 2 |

If you re-run `--stage grid` and the argmax moves, the script logs a warning
telling you to update either `AssetConfig` or the manuscript. Do not ignore it.

To extend the sample or add an asset, edit `Config.assets`, then re-run
`--stage fetch` (connected) followed by the analysis stages (offline).

---

## 5. Parallelism

`--jobs -1` uses all cores. Parallelised: per-day stress components, grid
configurations, walk-forward folds, robustness re-runs, bootstrap replicates.
Nested parallelism is suppressed automatically. `--jobs 1` gives deterministic
serial execution — the smoke test asserts the serial and parallel component paths
agree exactly, so `--jobs -1` is safe for published numbers.

On Windows, joblib spawns subprocesses; the `if __name__ == "__main__":` guard at
the bottom of the file is what makes that safe, so don't remove it.

---

## 6. Troubleshooting

| symptom | cause |
|---|---|
| `ModuleNotFoundError: ripser` | venv not activated, or interpreter not selected in VS Code |
| ripser fails to build on Windows | install **Microsoft C++ Build Tools**, or `conda install -c conda-forge ripser` |
| `No cached data for …` | run `--stage fetch` once with a connection |
| `CACHE CHANGED` warning | the CSV was modified or re-downloaded; your numbers may no longer match the tables |
| Grid search appears to hang | it is genuinely slow (72 configs × every trading day). Watch the joblib progress lines |
| Memory pressure during grid | lower `--jobs`; each worker holds its own price series and clouds |
