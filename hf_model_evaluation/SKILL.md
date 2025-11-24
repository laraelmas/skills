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

# Version
1.0.0

# Dependencies
- huggingface_hub>=1.1.4
- python-dotenv>=1.2.1
- pyyaml>=6.0.3
- requests>=2.32.5
- re (built-in)

# Core Capabilities

## 1. Extract Evaluation Tables from README
- **Parse Markdown Tables**: Automatically detect and parse evaluation tables in model READMEs
- **Multiple Table Support**: Handle models with multiple benchmark tables
- **Format Detection**: Recognize common evaluation table formats (benchmarks as rows or columns)
- **Smart Conversion**: Convert parsed tables to model-index YAML format

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

# Usage Instructions

The skill includes a Python script `scripts/evaluation_manager.py` to perform operations.

### Prerequisites
- Install dependencies: `uv add huggingface_hub python-dotenv pyyaml requests`
- Set `HF_TOKEN` environment variable with Write-access token
- For Artificial Analysis: Set `AA_API_KEY` environment variable
- Activate virtual environment: `source .venv/bin/activate`

### Method 1: Extract from README

Extract evaluation tables from a model's existing README and add them to model-index metadata.

**Basic Usage:**
```bash
python scripts/evaluation_manager.py extract-readme \
  --repo-id "username/model-name"
```

**With Custom Task Type:**
```bash
python scripts/evaluation_manager.py extract-readme \
  --repo-id "username/model-name" \
  --task-type "text-generation" \
  --dataset-name "Custom Benchmarks"
```

**Dry Run (Preview Only):**
```bash
python scripts/evaluation_manager.py extract-readme \
  --repo-id "username/model-name" \
  --dry-run
```

#### Supported Table Formats

**Format 1: Benchmarks as Rows**
```markdown
| Benchmark | Score |
|-----------|-------|
| MMLU      | 85.2  |
| HumanEval | 72.5  |
```

**Format 2: Benchmarks as Columns**
```markdown
| MMLU | HumanEval | GSM8K |
|------|-----------|-------|
| 85.2 | 72.5      | 91.3  |
```

**Format 3: Multiple Metrics**
```markdown
| Benchmark | Accuracy | F1 Score |
|-----------|----------|----------|
| MMLU      | 85.2     | 0.84     |
```

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

### Commands Reference

**List Available Commands:**
```bash
python scripts/evaluation_manager.py --help
```

**Extract from README:**
```bash
python scripts/evaluation_manager.py extract-readme \
  --repo-id "username/model-name" \
  [--task-type "text-generation"] \
  [--dataset-name "Custom Benchmarks"] \
  [--dry-run] \
  [--create-pr]
```

**Import from Artificial Analysis:**
```bash
python scripts/evaluation_manager.py import-aa \
  --creator-slug "creator-name" \
  --model-name "model-slug" \
  --repo-id "username/model-name" \
  [--create-pr]
```

**View Current Evaluations:**
```bash
python scripts/evaluation_manager.py show \
  --repo-id "username/model-name"
```

**Validate Model-Index:**
```bash
python scripts/evaluation_manager.py validate \
  --repo-id "username/model-name"
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

### Advanced Usage

**Extract Multiple Tables:**
```bash
# The script automatically detects and processes all evaluation tables
python scripts/evaluation_manager.py extract-readme \
  --repo-id "username/model-name" \
  --merge-tables
```

**Custom Metric Mapping:**
```bash
# Use a JSON file to map column names to metric types
python scripts/evaluation_manager.py extract-readme \
  --repo-id "username/model-name" \
  --metric-mapping "$(cat metric_mapping.json)"
```

Example `metric_mapping.json`:
```json
{
  "MMLU": {"type": "mmlu", "name": "Massive Multitask Language Understanding"},
  "HumanEval": {"type": "humaneval", "name": "Code Generation (HumanEval)"},
  "GSM8K": {"type": "gsm8k", "name": "Grade School Math"}
}
```

**Batch Processing:**
```bash
# Process multiple models from a list
while read repo_id; do
  python scripts/evaluation_manager.py extract-readme --repo-id "$repo_id"
done < models.txt
```

### Error Handling
- **Table Not Found**: Script will report if no evaluation tables are detected
- **Invalid Format**: Clear error messages for malformed tables
- **API Errors**: Retry logic for transient Artificial Analysis API failures
- **Token Issues**: Validation before attempting updates
- **Merge Conflicts**: Preserves existing model-index entries when adding new ones

### Best Practices

1. **Use Dry Run First**: Always preview changes with `--dry-run` before committing
2. **Validate After Updates**: Run `validate` command to ensure proper formatting
3. **Source Attribution**: Include source information for traceability
4. **Regular Updates**: Keep evaluation scores current as new benchmarks emerge
5. **Create PRs for Others**: Use `--create-pr` when updating models you don't own

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

**Issue**: "AA_API_KEY not set"
- **Solution**: Set environment variable or add to .env file

**Issue**: "Token does not have write access"
- **Solution**: Ensure HF_TOKEN has write permissions for the repository

**Issue**: "Model not found in Artificial Analysis"
- **Solution**: Verify creator-slug and model-name match API values

### Integration Examples

**CI/CD Pipeline:**
```yaml
# .github/workflows/update-evals.yml
name: Update Evaluations
on:
  schedule:
    - cron: '0 0 * * 0'  # Weekly updates
jobs:
  update:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Update from Artificial Analysis
        env:
          AA_API_KEY: ${{ secrets.AA_API_KEY }}
          HF_TOKEN: ${{ secrets.HF_TOKEN }}
        run: |
          python scripts/evaluation_manager.py import-aa \
            --creator-slug "${{ github.repository_owner }}" \
            --model-name "${{ github.event.repository.name }}" \
            --repo-id "${{ github.repository }}" \
            --create-pr
```

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
