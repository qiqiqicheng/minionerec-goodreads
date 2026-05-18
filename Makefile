.PHONY: sft
sft:
	@uv sync
	@uv run python -m src.minionerec_goodreads.scripts.sft.train_sft

.PHONY: sft-stats
sft-stats:
	@uv sync
	@uv run python -m src.minionerec_goodreads.scripts.sft.prompt_length_stats

.PHONY: rl
rl:
	@uv sync --group qlora
	@uv run --group qlora python -m src.minionerec_goodreads.scripts.rl.train_rl

.PHONY: eval
eval:
	@uv sync
	@uv run python -m src.minionerec_goodreads.scripts.eval

.PHONY: sid
sid:
	@uv sync
	@uv run python -m src.minionerec_goodreads.scripts.rqvae.generate_sid

.PHONY: rqvae
rqvae:
	@uv sync
	@uv run python -m src.minionerec_goodreads.scripts.rqvae.train_rqvae

.PHONY: data
data: ## Install the virtual environment and install the pre-commit hooks
	@uv sync
	@uv run python -m src.minionerec_goodreads.scripts.rqvae.data_preprocess

.PHONY: eda
eda: ## Run raw data EDA and write notes
	@uv sync
	@uv run python -m src.minionerec_goodreads.scripts.eda

.PHONY: embedding
embedding: ## Install the virtual environment and install the pre-commit hooks
	@uv sync
	@uv run python -m src.minionerec_goodreads.scripts.rqvae.text2emb

.PHONY: install
install: ## Install the virtual environment and install the pre-commit hooks
	@echo "🚀 Creating virtual environment using uv"
	@uv sync
	@uv run pre-commit install

.PHONY: check
check: ## Run code quality tools.
	@echo "🚀 Checking lock file consistency with 'pyproject.toml'"
	@uv lock --locked
	@echo "🚀 Linting code: Running pre-commit"
	@uv run pre-commit run -a
	@echo "🚀 Checking for obsolete dependencies: Running deptry"
	@uv run deptry src

.PHONY: test
test: ## Test the code with pytest
	@echo "🚀 Testing code: Running pytest"
	@uv run python -m pytest --doctest-modules

.PHONY: build
build: clean-build ## Build wheel file
	@echo "🚀 Creating wheel file"
	@uvx --from build pyproject-build --installer uv

.PHONY: clean-build
clean-build: ## Clean build artifacts
	@echo "🚀 Removing build artifacts"
	@uv run python -c "import shutil; import os; shutil.rmtree('dist') if os.path.exists('dist') else None"

.PHONY: help
help:
	@uv run python -c "import re; \
	[[print(f'\033[36m{m[0]:<20}\033[0m {m[1]}') for m in re.findall(r'^([a-zA-Z_-]+):.*?## (.*)$$', open(makefile).read(), re.M)] for makefile in ('$(MAKEFILE_LIST)').strip().split()]"

.DEFAULT_GOAL := help
