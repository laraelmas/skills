---
name: hugging-face-evaluation-manager
description: Add and manage evaluation results in Hugging Face model cards. Supports extracting eval tables from README content and importing scores from Artificial Analysis API. Works with the model-index metadata format.
---

# Overview
This skill provides tools to add structured evaluation results to Hugging Face model cards. It supports two primary methods for adding evaluation data: extracting existing evaluation tables from README content and importing benchmark scores from Artificial Analysis.

## Integration with HF Ecosystem
- **Model Cards**: Updates model-index metadata for leaderboard integration
- **Artificial Analysis**: Direct API integration for benchmark imports
- **Papers with Code**: Compatible with their model-index specification
- **Jobs**: Run evaluations directly on Hugging Face Jobs with `uv` integration

# Version
1.2.0

# Dependencies
- huggingface_hub>=0.26.0
- markdown-it-py>=3.0.0
- python-dotenv>=1.2.1
- pyyaml>=6.0.3
- requests>=2.32.5
- inspect-ai>=0.3.0
- re (built-in)

# IMPORTANT: Using This Skill

**Use `--help` for the latest workflow guidance.** Works with plain Python or `uv run`:
```bash
uv run scripts/evaluation_manager.py --help
uv run scripts/evaluation_manager.py inspect-tables --help
uv run scripts/evaluation_manager.py extract-readme --help
```
Key workflow (matches CLI help):
1) `inspect-tables` → find table numbers/columns  
2) `extract-readme --table N` → prints YAML by default  
3) add `--apply` (push) or `--create-pr` to write changes

# Core Capabilities

## 1. Inspect and Extract Evaluation Tables from README
- **Inspect Tables**: Use `inspect-tables` to see all tables in a README with structure, columns, and sample rows
- **Parse Markdown Tables**: Accurate parsing using markdown-it-py (ignores code blocks and examples)
- **Table Selection**: Use `--table N` to extract from a specific table (required when multiple tables exist)
- **Format Detection**: Recognize common formats (benchmarks as rows, columns, or comparison tables with multiple models)
- **Column Matching**: Automatically identify model columns/rows; choose explicitly with `--model-name-override` (by text) or `--model-column-index` (by index from inspect output)
- **YAML Generation**: Convert selected table to model-index YAML format
- **Task Typing**: `--task-type` sets the `task.type` field in model-index output (e.g., `text-generation`, `summarization`)

## 2. Import from Artificial Analysis
- **API Integration**: Fetch benchmark scores directly from Artificial Analysis
- **Automatic Formatting**: Convert API responses to model-index format
- **Metadata Preservation**: Maintain source attribution and URLs
- **PR Creation**: Automatically create pull requests with evaluation updates

## 3. Model-Index Management
- **YAML Generation**: Create properly formatted model-index entries
- **Merge Support**: Add evaluations to existing model cards without overwriting
- **Validation**: Ensure compliance with Papers with Code specification
- **Batch Operations**: Process multiple models efficiently

## 4. Run Evaluations on HF Jobs
- **Inspect-AI Integration**: Run standard evaluations using the `inspect-ai` library
- **UV Integration**: Seamlessly run Python scripts with ephemeral dependencies on HF infrastructure
- **Zero-Config**: No Dockerfiles or Space management required
- **Hardware Selection**: Configure CPU or GPU hardware for the evaluation job
- **Secure Execution**: Handles API tokens safely via secrets passed through the CLI

# Usage Instructions

The skill includes Python scripts in `scripts/` to perform operations.

### Prerequisites
- Preferred: use `uv run` (PEP 723 header auto-installs deps)
- Or install manually: `pip install huggingface-hub markdown-it-py python-dotenv pyyaml requests`
- Set `HF_TOKEN` environment variable with Write-access token
- For Artificial Analysis: Set `AA_API_KEY` environment variable
- `.env` is loaded automatically if `python-dotenv` is installed

### Method 1: Extract from README (CLI workflow)

Recommended flow (matches `--help`):
```bash
# 1) Inspect tables to get table numbers and column hints
uv run scripts/evaluation_manager.py inspect-tables --repo-id "username/model"

# 2) Extract a specific table (prints YAML by default)
uv run scripts/evaluation_manager.py extract-readme \
  --repo-id "username/model" \
  --table 1 \
  [--model-name-override "<column header/model name>"] \
  [--model-column-index <column index shown by inspect-tables>]

# 3) Apply changes (push or PR)
uv run scripts/evaluation_manager.py extract-readme \
  --repo-id "username/model" \
  --table 1 \
  --apply       # push directly
# or
uv run scripts/evaluation_manager.py extract-readme \
  --repo-id "username/model" \
  --table 1 \
  --create-pr   # open a PR
```

Validation checklist:
- By default YAML is printed; compare against the README table before applying.
- Use `--model-name-override` when your model column/row is not an exact match.
- For transposed tables (models as rows), ensure only one row is extracted.

### Method 2: Import from Artificial Analysis

Fetch benchmark scores from Artificial Analysis API and add them to a model card.

**Basic Usage:**
```bash
AA_API_KEY="your-api-key" python scripts/evaluation_manager.py import-aa \
  --creator-slug "anthropic" \
  --model-name "claude-sonnet-4" \
  --repo-id "username/model-name"
```

**With Environment File:**
```bash
# Create .env file
echo "AA_API_KEY=your-api-key" >> .env
echo "HF_TOKEN=your-hf-token" >> .env

# Run import
python scripts/evaluation_manager.py import-aa \
  --creator-slug "anthropic" \
  --model-name "claude-sonnet-4" \
  --repo-id "username/model-name"
```

**Create Pull Request:**
```bash
python scripts/evaluation_manager.py import-aa \
  --creator-slug "anthropic" \
  --model-name "claude-sonnet-4" \
  --repo-id "username/model-name" \
  --create-pr
```

### Method 3: Run Evaluation Job

Submit an evaluation job on Hugging Face infrastructure using the `hf jobs uv run` CLI.

**Direct CLI Usage:**
```bash
HF_TOKEN=$HF_TOKEN \
hf jobs uv run hf_model_evaluation/scripts/inspect_eval_uv.py \
  --flavor cpu-basic \
  --secret HF_TOKEN=$HF_TOKEN \
  -- --model "meta-llama/Llama-2-7b-hf" \
     --task "mmlu"
```

**GPU Example (A10G):**
```bash
HF_TOKEN=$HF_TOKEN \
hf jobs uv run hf_model_evaluation/scripts/inspect_eval_uv.py \
  --flavor a10g-small \
  --secret HF_TOKEN=$HF_TOKEN \
  -- --model "meta-llama/Llama-2-7b-hf" \
     --task "gsm8k"
```

**Python Helper (optional):**
```bash
python scripts/run_eval_job.py \
  --model "meta-llama/Llama-2-7b-hf" \
  --task "mmlu" \
  --hardware "t4-small"
```

### Commands Reference

**Top-level help and version:**
```bash
uv run scripts/evaluation_manager.py --help
uv run scripts/evaluation_manager.py --version
```

**Inspect Tables (start here):**
```bash
uv run scripts/evaluation_manager.py inspect-tables --repo-id "username/model-name"
```

**Extract from README:**
```bash
uv run scripts/evaluation_manager.py extract-readme \
  --repo-id "username/model-name" \
  --table N \
  [--model-name-override "Column Header or Model Name"] \
  [--model-column-index N] \
  [--task-type "text-generation"] \
  [--dataset-name "Custom Benchmarks"] \
  [--apply | --create-pr]
```

**Import from Artificial Analysis:**
```bash
AA_API_KEY=... uv run scripts/evaluation_manager.py import-aa \
  --creator-slug "creator-name" \
  --model-name "model-slug" \
  --repo-id "username/model-name" \
  [--create-pr]
```

**View / Validate:**
```bash
uv run scripts/evaluation_manager.py show --repo-id "username/model-name"
uv run scripts/evaluation_manager.py validate --repo-id "username/model-name"
```

**Run Evaluation Job:**
```bash
hf jobs uv run hf_model_evaluation/scripts/inspect_eval_uv.py \
  --flavor "cpu-basic|t4-small|..." \
  --secret HF_TOKEN=$HF_TOKEN \
  -- --model "model-id" \
     --task "task-name"
```

or use the Python helper:

```bash
python scripts/run_eval_job.py \
  --model "model-id" \
  --task "task-name" \
  --hardware "cpu-basic|t4-small|..."
```

### Model-Index Format

The generated model-index follows this structure:

```yaml
model-index:
  - name: Model Name
    results:
      - task:
          type: text-generation
        dataset:
          name: Benchmark Dataset
          type: benchmark_type
        metrics:
          - name: MMLU
            type: mmlu
            value: 85.2
          - name: HumanEval
            type: humaneval
            value: 72.5
        source:
          name: Source Name
          url: https://source-url.com
```

WARNING: Do not use markdown formatting in the model name. Use the exact name from the table. Only use urls in the source.url field.

### Error Handling
- **Table Not Found**: Script will report if no evaluation tables are detected
- **Invalid Format**: Clear error messages for malformed tables
- **API Errors**: Retry logic for transient Artificial Analysis API failures
- **Token Issues**: Validation before attempting updates
- **Merge Conflicts**: Preserves existing model-index entries when adding new ones
- **Space Creation**: Handles naming conflicts and hardware request failures gracefully

### Best Practices

1. **Always start with `inspect-tables`**: See table structure and get the correct extraction command
2. **Use `--help` for guidance**: Run `inspect-tables --help` to see the complete workflow
3. **Use `--dry-run` first**: Preview YAML output before applying changes
4. **Verify extracted values**: Compare YAML output against the README table manually
5. **Use `--table N` for multi-table READMEs**: Required when multiple evaluation tables exist
6. **Use `--model-name-override` for comparison tables**: Copy the exact column header from `inspect-tables` output
7. **Create PRs for Others**: Use `--create-pr` when updating models you don't own
8. **One model per repo**: Only add the main model's results to model-index
9. **No markdown in YAML names**: The model name field in YAML should be plain text

### Model Name Matching

When extracting evaluation tables with multiple models (either as columns or rows), the script uses **exact normalized token matching**:

- Removes markdown formatting (bold `**`, links `[]()`  )
- Normalizes names (lowercase, replace `-` and `_` with spaces)
- Compares token sets: `"OLMo-3-32B"` → `{"olmo", "3", "32b"}` matches `"**Olmo 3 32B**"` or `"[Olmo-3-32B](...)`
- Only extracts if tokens match exactly (handles different word orders and separators)
- Fails if no exact match found (rather than guessing from similar names)

**For column-based tables** (benchmarks as rows, models as columns):
- Finds the column header matching the model name
- Extracts scores from that column only

**For transposed tables** (models as rows, benchmarks as columns):
- Finds the row in the first column matching the model name
- Extracts all benchmark scores from that row only

This ensures only the correct model's scores are extracted, never unrelated models or training checkpoints. 

### Common Patterns

**Update Your Own Model:**
```bash
# Extract from README and push directly
python scripts/evaluation_manager.py extract-readme \
  --repo-id "your-username/your-model" \
  --task-type "text-generation"
```

**Update Someone Else's Model:**
```bash
# Create a PR instead of direct push
python scripts/evaluation_manager.py extract-readme \
  --repo-id "other-username/their-model" \
  --create-pr
```

**Import Fresh Benchmarks:**
```bash
# Get latest scores from Artificial Analysis
python scripts/evaluation_manager.py import-aa \
  --creator-slug "anthropic" \
  --model-name "claude-sonnet-4" \
  --repo-id "anthropic/claude-sonnet-4" \
  --create-pr
```

### Troubleshooting

**Issue**: "No evaluation tables found in README"
- **Solution**: Check if README contains markdown tables with numeric scores

**Issue**: "Could not find model 'X' in transposed table"
- **Solution**: The script will display available models. Use `--model-name-override` with the exact name from the list
- **Example**: `--model-name-override "**Olmo 3-32B**"`

**Issue**: "AA_API_KEY not set"
- **Solution**: Set environment variable or add to .env file

**Issue**: "Token does not have write access"
- **Solution**: Ensure HF_TOKEN has write permissions for the repository

**Issue**: "Model not found in Artificial Analysis"
- **Solution**: Verify creator-slug and model-name match API values

**Issue**: "Payment required for hardware"
- **Solution**: Add a payment method to your Hugging Face account to use non-CPU hardware

### Integration Examples

**Python Script Integration:**
```python
import subprocess
import os

def update_model_evaluations(repo_id, readme_content):
    """Update model card with evaluations from README."""
    result = subprocess.run([
        "python", "scripts/evaluation_manager.py",
        "extract-readme",
        "--repo-id", repo_id,
        "--create-pr"
    ], capture_output=True, text=True)

    if result.returncode == 0:
        print(f"Successfully updated {repo_id}")
    else:
        print(f"Error: {result.stderr}")
```
