# Step 00: Project Setup

## What was done (plain English, 5-8 lines)
Initialized the project repository and directory skeleton adhering to production ML standards.
Created an isolated Python 3.13 virtual environment (`.venv`) to manage package dependencies.
Specified pinned project dependencies covering data manipulation, modeling, explainability, web API, dashboard, and testing in `requirements.txt`.
Configured a comprehensive `.gitignore` preventing accidental commits of heavy raw datasets and trained binary model artifacts.
Created package markers and directory placeholders (`.gitkeep`) for all planned modular components.
Authored `src/config.py` centralizing reproducibility seeds, filesystem paths, time-split fractions, and business fraud cost ratios.
Verified module discovery and reproducibility by successfully importing `src.config`.

## Why this step matters
Professional machine learning engineering requires strict reproducibility, modular separation of concerns, and clean project hygiene from day one. Hardcoding filesystem paths or magic numbers across multiple notebooks creates code debt and runtime bugs. Centralizing all hyperparameters, split ratios, and business cost matrices into a single configuration module guarantees consistent pipeline behavior across data preprocessing, model training, evaluation, and production serving. Furthermore, ensuring data and models are never checked into version control prevents repository bloat and potential data privacy breaches.

## Files created / changed (path + one-line purpose each)
- `.gitignore`: Prevents Git tracking of raw/processed data, model binaries, virtual envs, and cache files.
- `requirements.txt`: Specifies dependencies across ML, visualization, API, dashboard, and testing libraries.
- `src/__init__.py`: Package initialization marker designating `src` as a Python package.
- `src/config.py`: Single source of truth for random seeds, paths, split ratios, column constants, and business costs.
- `app/__init__.py`: Package initialization marker designating `app` as a Python package for API and dashboard.
- `tests/__init__.py`: Package initialization marker for unit and pipeline tests.
- `data/raw/.gitkeep`: Preserves raw data directory structure in Git while keeping data files untracked.
- `data/processed/.gitkeep`: Preserves processed data directory structure in Git.
- `models/.gitkeep`: Preserves model artifacts directory structure in Git.
- `reports/.gitkeep`: Preserves reports and evaluation metrics directory structure in Git.
- `notebooks/.gitkeep`: Preserves exploratory analysis notebooks directory structure in Git.
- `docs/explanations/00_project_setup.md`: Step documentation and interview prep guide.

## Key decisions and alternatives rejected
- **Centralized `src/config.py` vs. Decentralized script arguments / hardcoded constants:** Rejected scattering constants across notebooks and scripts because it leads to split discrepancy, leakage, or mismatched cost evaluations.
- **`pathlib.Path` vs. `os.path.join`:** Chose modern object-oriented `pathlib.Path` for cross-platform robustness and clean path manipulation.
- **Cost matrix inclusion in config:** Embedded `COST_FN = 10.0` and `COST_FP = 1.0` in `config.py` early as confirmed in PRD section 5, so model evaluation and threshold tuning can directly consume it.
- **Gitignoring data and models from the start:** Explicitly ignored `data/` and `models/*` rather than cleaning them up later to avoid repository bloat and credential/data exposure.

## Results / outputs (numbers, plots, observations)
- Folder structure verified: `data/`, `notebooks/`, `src/`, `app/`, `models/`, `reports/`, `docs/explanations/`, `tests/` all created.
- Verification command executed: `python -c "import src.config"` runs without errors.
- Random seed established: `42`.
- Time-based split ratios locked: `70%` train, `15%` validation, `15%` test.
- Fraud cost ratio locked: Missed fraud (FN) = 10x False Alarm (FP).

## How to explain this in an interview (3-5 likely questions with short answers)
- **Q: Why did you structure the project into `src/`, `app/`, and `notebooks/` instead of doing everything in Jupyter notebooks?**
  *A:* Notebooks are great for exploration, but poor for maintainability, unit testing, and production deployment. By placing reusable logic (cleaning, feature engineering, config) into `src/`, both the notebooks and the production API/Dashboard consume the exact same tested code without copy-pasting.
- **Q: How do you handle configuration and secrets across environments?**
  *A:* I centralize all static pipeline constants (seeds, split ratios, directory anchors) into `src/config.py` using `pathlib.Path`. Environment-specific secrets or overrides are loaded via environment variables and `.env` files, which are strictly gitignored.
- **Q: Why are data files and model binaries excluded from Git?**
  *A:* Git is designed for tracking diffs in source code, not multi-gigabyte binary or CSV files. Storing datasets in Git inflates repo size, slows down cloning, and risks data leakage. Instead, data is fetched via reproducible scripts, and models are tracked in dedicated artifact directories.
- **Q: Why define a business cost function (cost_FN vs. cost_FP) in configuration before even training a model?**
  *A:* In fraud detection, standard ML metrics like accuracy are useless, and pure statistical thresholds (like 0.5) are arbitrary. In real financial systems, missing a fraudster (FN) incurs direct chargebacks (costly), whereas an extra SMS verification for a false alarm (FP) has small operational cost. Defining this up front ensures all modeling choices optimize for actual dollar impact.

## Limitations / what I would improve
- In enterprise production, configurations would be backed by tools like `pydantic-settings` or `hydra` for hierarchical multi-environment configurations (dev, staging, prod).
- DVC (Data Version Control) could be integrated alongside Git if strict remote dataset versioning with AWS S3 / GCP buckets were required.
