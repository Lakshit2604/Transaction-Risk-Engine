# Step 11: Docker and Deployment

## What was done (plain English, 5-8 lines)
Engineered containerization and production deployment configurations for the complete Transaction Risk Engine system.
Created a lean, production-grade `Dockerfile` using `python:3.11-slim` with system dependencies (`libgomp1` for OpenMP LightGBM multithreading and `curl` for container healthchecks).
Configured `.dockerignore` to strictly prevent massive raw/processed datasets, virtual environments, Git history, and bytecodes from inflating container layers.
Constructed a `docker-compose.yml` multi-container specification defining decoupled `api` (FastAPI on port 8000) and `dashboard` (Streamlit on port 8501) services with healthchecks and restart policies.
Defined a declarative `render.yaml` Blueprint for zero-friction continuous deployment of both services on Render's cloud platform.
Documented container execution commands, environment variables, healthchecks, and step-by-step instructions for Hugging Face Spaces and Render hosting.

## Why this step matters
Machine learning models that only run in local Jupyter notebooks or ad-hoc virtual environments cannot be integrated into real-world banking infrastructure. Packaging the model, feature pipeline, API, and dashboard into standardized Docker containers eliminates the "it works on my machine" problem by guaranteeing identical runtime dependencies, C-library bindings (like OpenMP for LightGBM), and environment variables across developer workstations, staging environments, and cloud clusters (AWS ECS, Kubernetes, Render).

## Files created / changed (path + one-line purpose each)
- `Dockerfile`: Multi-stage Python 3.11 container specification with system libraries (`libgomp1`, `curl`), application code, model binaries, and container healthchecks.
- `.dockerignore`: Build optimization rule list excluding data, venv, git metadata, and caches from container context.
- `docker-compose.yml`: Multi-service orchestration definition coordinating the FastAPI inference service and Streamlit operations dashboard.
- `render.yaml`: Infrastructure-as-code cloud blueprint for one-click deployment to Render web services.
- `docs/explanations/11_docker_and_deployment.md`: Step documentation and deployment guide.

## Key decisions and alternatives rejected
- **Debian-slim (`python:3.11-slim`) vs. Alpine Linux:** Rejected Alpine because compiling C/C++ scientific packages (such as NumPy, SciPy, and LightGBM) requires `musl-libc` instead of `glibc`, resulting in lengthy, error-prone wheel compilation and fragile runtime dependencies. Debian-slim provides standard `glibc` with pre-built binary wheels while maintaining a small image footprint (~250MB base).
- **Installing `libgomp1`:** LightGBM relies heavily on OpenMP for high-performance multithreaded tree traversal and inference. Without `libgomp1` installed in the Linux OS layer, LightGBM crashes with `OSError: dlopen libgomp.so.1: cannot open shared object file`.
- **Decoupled Multi-Container Architecture vs Single Monolith:** In `docker-compose.yml`, separated the API service from the Dashboard service. In high-volume production, transaction ingestion APIs must autoscale horizontally under high QPS independently of analyst dashboard usage.
- **Exclusion of `data/` Directory:** Kept all training/validation datasets out of the Docker image via `.dockerignore`. Production containers only contain the serialized pipeline (`models/feature_pipeline.joblib`) and model binary (`models/best_lightgbm.joblib`), keeping container pull and startup times under 10 seconds.

## Results / outputs (numbers, plots, observations)
- **Container Configurations:**
  - `Dockerfile`: Exposes port 8000 (API) and port 8501 (Dashboard), includes automated HTTP healthcheck (`curl -f http://localhost:${PORT}/health`).
  - `docker-compose.yml`: Launches both services with unified networking:
    - API: `http://localhost:8000` (docs at `/docs`)
    - Dashboard: `http://localhost:8501`
  - `render.yaml`: Automates deployment of both services on free tier cloud hosting.
- **Local Run Commands:**
  ```bash
  # Build and run both services with Docker Compose
  docker compose up --build

  # Run standalone API container
  docker build -t transaction-risk-api .
  docker run -p 8000:8000 -e PORT=8000 transaction-risk-api

  # Run standalone Dashboard container
  docker run -p 8501:8501 -e PORT=8501 transaction-risk-api streamlit run app/dashboard.py --server.port 8501 --server.address 0.0.0.0
  ```

## How to explain this in an interview (3-5 likely questions with short answers)
- **Q: Why did you choose `python:3.11-slim` over Alpine for an ML container?**
  *A:* Alpine uses `musl` libc instead of standard `glibc`. Numerical and machine learning Python libraries like NumPy, SciPy, and LightGBM provide pre-compiled wheels for `glibc`. Attempting to install them on Alpine forces slow compilation from C/C++ source and frequently causes obscure C-runtime segmentation faults. `python:3.11-slim` provides rock-solid `glibc` compatibility and fast installs with minimal image bloat.
- **Q: Why is `libgomp1` explicitly installed in the Dockerfile?**
  *A:* LightGBM is written in C++ and uses OpenMP for multi-threaded parallel computation during histogram construction and batch tree inference. Minimal Linux distributions like Debian-slim omit OpenMP shared runtime libraries by default, so installing `libgomp1` is necessary to prevent runtime link errors.
- **Q: How would you deploy this container architecture in an enterprise cloud environment?**
  *A:* In enterprise AWS or GCP environments, I would push the Docker image to Amazon ECR / Google Artifact Registry. The FastAPI service would run on AWS ECS Fargate or Kubernetes (EKS) behind an Application Load Balancer with auto-scaling based on target CPU/request count. The Streamlit dashboard would run as a separate internal ECS service protected behind corporate SSO / VPN.
- **Q: How do you handle sensitive model updates without rebuilding the whole image?**
  *A:* While baking the model into the image is convenient for testing and reproducible immutable releases, in a CI/CD production architecture I would decouple model artifacts from container code. The container startup script would pull the latest approved model artifact from an S3 bucket or MLflow Model Registry at boot time.

## Limitations / what I would improve
- Implement a non-root user (`USER appuser`) in the Dockerfile to adhere to enterprise container security hardening guidelines.
- Add an automated GitHub Actions CI/CD workflow that runs `pytest` and automatically builds and pushes the image to Docker Hub / GHCR on every push to `main`.
