# CI/CD Workflows

This directory contains GitHub Actions workflows for automated testing and validation of the offers-reranker project.

## Structure

### Reusable Workflow

- **`reusable-ci.yaml`**: A reusable workflow that provides common CI functionality for all subprojects. It includes three independent jobs that run in parallel:
  - **Ruff job**: Linting and formatting checks
  - **MyPy job**: Type checking
  - **Pytest job**: Testing
  
  Each job runs independently, so if one fails, the others will still complete. This ensures comprehensive feedback even when some checks fail.
  
  **Optimizations:**
  - Uses `cache-dependency-glob` to cache dependencies based on `uv.lock`
  - Uses `--frozen` flag to skip dependency resolution
  - Uses `--no-install-project` to only install dependencies (faster)
  - Consistent `UV_CACHE_DIR` across jobs for better cache sharing

### Subproject Workflows

Each subproject has its own dedicated workflow that calls the reusable workflow:

- **`etl-ci.yaml`**: CI for the ETL subproject
  - Runs on changes to `etl/**`
  - Executes linting and type checking (no tests currently)

- **`training-ci.yaml`**: CI for the Training subproject
  - Runs on changes to `training/**`
  - Executes linting, type checking, and tests

- **`torchserve-ci.yaml`**: CI for the inference-torchserve subproject
  - Runs on changes to `inference-torchserve/**`
  - Executes linting, type checking, and tests

- **`mosec-ci.yaml`**: CI for the inference-mosec subproject
  - Runs on changes to `inference-mosec/**`
  - Executes linting, type checking, and tests

## Triggers

All workflows trigger on:
- Push to `main`, `master`, `develop`, and `feat/**` branches
- Pull requests to `main`, `master`, and `develop` branches
- Only when files in the respective subproject directory change

## Configuration

Each workflow can be customized with the following parameters:
- `working-directory`: The subproject directory
- `python-version`: Python version to use (default: 3.12)
- `run-tests`: Whether to run pytest
- `run-mypy`: Whether to run mypy type checking
- `run-ruff`: Whether to run ruff linting

## Development Dependencies

All subprojects now include the following dev dependencies:
- `mypy>=1.11.0` - Type checking
- `ruff>=0.8.0` - Linting and formatting
- `pytest>=8.4.2` - Testing (training and inference only)

## Migration Notes

The old `mypy.yaml` and `pytest.yaml` workflows may be outdated and can potentially be removed in favor of the new subproject-specific workflows.

