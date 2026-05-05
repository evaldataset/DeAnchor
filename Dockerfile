FROM python:3.12-slim

WORKDIR /app

# System deps for building wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git && \
    rm -rf /var/lib/apt/lists/*

# Install Python deps (CPU-only torch for reproducibility without GPU)
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .
RUN pip install --no-cache-dir -e .

# Smoke test: imports + unit tests
RUN python -c "\
from llm_explainer.llm_inference import _parse_json_response; \
from evaluation.bootstrap_ci import paired_permutation_test; \
from evaluation.fusion_baseline import load; \
from llm_explainer.prompts.fp_explanation import build_prompt; \
print('All imports OK')"

RUN pytest tests/ -q --tb=short

# Default: validate paper numbers
CMD ["python", "scripts/validate_paper_numbers.py"]
