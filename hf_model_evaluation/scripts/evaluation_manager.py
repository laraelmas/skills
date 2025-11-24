# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "huggingface-hub>=1.1.4",
#     "python-dotenv>=1.2.1",
#     "pyyaml>=6.0.3",
#     "requests>=2.32.5",
# ]
# ///

"""
Manage evaluation results in Hugging Face model cards.

This script provides two methods:
1. Extract evaluation tables from model README files
2. Import evaluation scores from Artificial Analysis API

Both methods update the model-index metadata in model cards.
"""

import argparse
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import dotenv
import requests
import yaml
from huggingface_hub import ModelCard

dotenv.load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
AA_API_KEY = os.getenv("AA_API_KEY")


# ============================================================================
# Method 1: Extract Evaluations from README
# ============================================================================


def extract_tables_from_markdown(markdown_content: str) -> List[str]:
    """Extract all markdown tables from content."""
    # Pattern to match markdown tables
    table_pattern = r"(\|[^\n]+\|(?:\r?\n\|[^\n]+\|)+)"
    tables = re.findall(table_pattern, markdown_content)
    return tables


def parse_markdown_table(table_str: str) -> Tuple[List[str], List[List[str]]]:
    """
    Parse a markdown table string into headers and rows.

    Returns:
        Tuple of (headers, data_rows)
    """
    lines = [line.strip() for line in table_str.strip().split("\n")]

    # Remove separator line (the one with dashes)
    lines = [line for line in lines if not re.match(r"^\|[\s\-:]+\|$", line)]

    if len(lines) < 2:
        return [], []

    # Parse header
    header = [cell.strip() for cell in lines[0].split("|")[1:-1]]

    # Parse data rows
    data_rows = []
    for line in lines[1:]:
        cells = [cell.strip() for cell in line.split("|")[1:-1]]
        if cells:
            data_rows.append(cells)

    return header, data_rows


def is_evaluation_table(header: List[str], rows: List[List[str]]) -> bool:
    """Determine if a table contains evaluation results."""
    if not header or not rows:
        return False

    # Check if first column looks like benchmark names
    benchmark_keywords = [
        "benchmark", "task", "dataset", "eval", "test", "metric",
        "mmlu", "humaneval", "gsm", "hellaswag", "arc", "winogrande",
        "truthfulqa", "boolq", "piqa", "siqa"
    ]

    first_col = header[0].lower()
    has_benchmark_header = any(keyword in first_col for keyword in benchmark_keywords)

    # Check if there are numeric values in the table
    has_numeric_values = False
    for row in rows:
        for cell in row:
            try:
                float(cell.replace("%", "").replace(",", ""))
                has_numeric_values = True
                break
            except ValueError:
                continue
        if has_numeric_values:
            break

    return has_benchmark_header or has_numeric_values


def extract_metrics_from_table(
    header: List[str],
    rows: List[List[str]],
    table_format: str = "auto"
) -> List[Dict[str, Any]]:
    """
    Extract metrics from parsed table data.

    Args:
        header: Table column headers
        rows: Table data rows
        table_format: "rows" (benchmarks as rows), "columns" (benchmarks as columns), or "auto"

    Returns:
        List of metric dictionaries with name, type, and value
    """
    metrics = []

    if table_format == "auto":
        # Heuristic: if first row has mostly numeric values, benchmarks are columns
        try:
            numeric_count = sum(
                1 for cell in rows[0] if cell and
                re.match(r"^\d+\.?\d*%?$", cell.replace(",", "").strip())
            )
            table_format = "columns" if numeric_count > len(rows[0]) / 2 else "rows"
        except (IndexError, ValueError):
            table_format = "rows"

    if table_format == "rows":
        # Benchmarks are in rows, scores in columns
        for row in rows:
            if not row:
                continue

            benchmark_name = row[0].strip()
            if not benchmark_name:
                continue

            # Extract numeric values from remaining columns
            for i, cell in enumerate(row[1:], start=1):
                try:
                    # Remove common suffixes and convert to float
                    value_str = cell.replace("%", "").replace(",", "").strip()
                    if not value_str:
                        continue

                    value = float(value_str)

                    # Determine metric name
                    metric_name = benchmark_name
                    if len(header) > i and header[i].lower() not in ["score", "value", "result"]:
                        metric_name = f"{benchmark_name} ({header[i]})"

                    metrics.append({
                        "name": metric_name,
                        "type": benchmark_name.lower().replace(" ", "_"),
                        "value": value
                    })
                    break  # Only take first numeric value per row
                except (ValueError, IndexError):
                    continue

    else:  # table_format == "columns"
        # Benchmarks are in columns
        if not rows:
            return metrics

        # Use first data row for values
        data_row = rows[0]

        for i, benchmark_name in enumerate(header):
            if not benchmark_name or i >= len(data_row):
                continue

            try:
                value_str = data_row[i].replace("%", "").replace(",", "").strip()
                if not value_str:
                    continue

                value = float(value_str)

                metrics.append({
                    "name": benchmark_name,
                    "type": benchmark_name.lower().replace(" ", "_"),
                    "value": value
                })
            except ValueError:
                continue

    return metrics


def extract_evaluations_from_readme(
    repo_id: str,
    task_type: str = "text-generation",
    dataset_name: str = "Benchmarks",
    dataset_type: str = "benchmark"
) -> Optional[List[Dict[str, Any]]]:
    """
    Extract evaluation results from a model's README.

    Args:
        repo_id: Hugging Face model repository ID
        task_type: Task type for model-index (e.g., "text-generation")
        dataset_name: Name for the benchmark dataset
        dataset_type: Type identifier for the dataset

    Returns:
        Model-index formatted results or None if no evaluations found
    """
    try:
        card = ModelCard.load(repo_id, token=HF_TOKEN)
        readme_content = card.content

        if not readme_content:
            print(f"No README content found for {repo_id}")
            return None

        # Extract all tables
        tables = extract_tables_from_markdown(readme_content)

        if not tables:
            print(f"No tables found in README for {repo_id}")
            return None

        # Parse and filter evaluation tables
        all_metrics = []
        for table_str in tables:
            header, rows = parse_markdown_table(table_str)

            if is_evaluation_table(header, rows):
                metrics = extract_metrics_from_table(header, rows)
                all_metrics.extend(metrics)

        if not all_metrics:
            print(f"No evaluation tables found in README for {repo_id}")
            return None

        # Build model-index structure
        model_name = repo_id.split("/")[-1] if "/" in repo_id else repo_id

        results = [{
            "task": {"type": task_type},
            "dataset": {
                "name": dataset_name,
                "type": dataset_type
            },
            "metrics": all_metrics,
            "source": {
                "name": "Model README",
                "url": f"https://huggingface.co/{repo_id}"
            }
        }]

        return results

    except Exception as e:
        print(f"Error extracting evaluations from README: {e}")
        return None


# ============================================================================
# Method 2: Import from Artificial Analysis
# ============================================================================


def get_aa_model_data(creator_slug: str, model_name: str) -> Optional[Dict[str, Any]]:
    """
    Fetch model evaluation data from Artificial Analysis API.

    Args:
        creator_slug: Creator identifier (e.g., "anthropic", "openai")
        model_name: Model slug/identifier

    Returns:
        Model data dictionary or None if not found
    """
    if not AA_API_KEY:
        raise ValueError("AA_API_KEY environment variable is not set")

    url = "https://artificialanalysis.ai/api/v2/data/llms/models"
    headers = {"x-api-key": AA_API_KEY}

    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()

        data = response.json().get("data", [])

        for model in data:
            creator = model.get("model_creator", {})
            if creator.get("slug") == creator_slug and model.get("slug") == model_name:
                return model

        print(f"Model {creator_slug}/{model_name} not found in Artificial Analysis")
        return None

    except requests.RequestException as e:
        print(f"Error fetching data from Artificial Analysis: {e}")
        return None


def aa_data_to_model_index(
    model_data: Dict[str, Any],
    dataset_name: str = "Artificial Analysis Benchmarks",
    dataset_type: str = "artificial_analysis",
    task_type: str = "evaluation"
) -> List[Dict[str, Any]]:
    """
    Convert Artificial Analysis model data to model-index format.

    Args:
        model_data: Raw model data from AA API
        dataset_name: Dataset name for model-index
        dataset_type: Dataset type identifier
        task_type: Task type for model-index

    Returns:
        Model-index formatted results
    """
    model_name = model_data.get("name", model_data.get("slug", "unknown-model"))
    evaluations = model_data.get("evaluations", {})

    if not evaluations:
        print(f"No evaluations found for model {model_name}")
        return []

    metrics = []
    for key, value in evaluations.items():
        if value is not None:
            metrics.append({
                "name": key.replace("_", " ").title(),
                "type": key,
                "value": value
            })

    results = [{
        "task": {"type": task_type},
        "dataset": {
            "name": dataset_name,
            "type": dataset_type
        },
        "metrics": metrics,
        "source": {
            "name": "Artificial Analysis API",
            "url": "https://artificialanalysis.ai"
        }
    }]

    return results


def import_aa_evaluations(
    creator_slug: str,
    model_name: str,
    repo_id: str
) -> Optional[List[Dict[str, Any]]]:
    """
    Import evaluation results from Artificial Analysis for a model.

    Args:
        creator_slug: Creator identifier in AA
        model_name: Model identifier in AA
        repo_id: Hugging Face repository ID to update

    Returns:
        Model-index formatted results or None if import fails
    """
    model_data = get_aa_model_data(creator_slug, model_name)

    if not model_data:
        return None

    results = aa_data_to_model_index(model_data)
    return results


# ============================================================================
# Model Card Update Functions
# ============================================================================


def update_model_card_with_evaluations(
    repo_id: str,
    results: List[Dict[str, Any]],
    create_pr: bool = False,
    commit_message: Optional[str] = None
) -> bool:
    """
    Update a model card with evaluation results.

    Args:
        repo_id: Hugging Face repository ID
        results: Model-index formatted results
        create_pr: Whether to create a PR instead of direct push
        commit_message: Custom commit message

    Returns:
        True if successful, False otherwise
    """
    if not HF_TOKEN:
        raise ValueError("HF_TOKEN environment variable is not set")

    try:
        # Load existing card
        card = ModelCard.load(repo_id, token=HF_TOKEN)

        # Get model name
        model_name = repo_id.split("/")[-1] if "/" in repo_id else repo_id

        # Create or update model-index
        model_index = [{
            "name": model_name,
            "results": results
        }]

        # Merge with existing model-index if present
        if "model-index" in card.data:
            existing = card.data["model-index"]
            if isinstance(existing, list) and existing:
                # Keep existing name if present
                if "name" in existing[0]:
                    model_index[0]["name"] = existing[0]["name"]

                # Merge results
                existing_results = existing[0].get("results", [])
                model_index[0]["results"].extend(existing_results)

        card.data["model-index"] = model_index

        # Prepare commit message
        if not commit_message:
            commit_message = f"Add evaluation results to {model_name}"

        commit_description = (
            "This commit adds structured evaluation results to the model card. "
            "The results are formatted using the model-index specification and "
            "will be displayed in the model card's evaluation widget."
        )

        # Push update
        card.push_to_hub(
            repo_id,
            token=HF_TOKEN,
            commit_message=commit_message,
            commit_description=commit_description,
            create_pr=create_pr
        )

        action = "Pull request created" if create_pr else "Model card updated"
        print(f"✓ {action} successfully for {repo_id}")
        return True

    except Exception as e:
        print(f"Error updating model card: {e}")
        return False


def show_evaluations(repo_id: str) -> None:
    """Display current evaluations in a model card."""
    try:
        card = ModelCard.load(repo_id, token=HF_TOKEN)

        if "model-index" not in card.data:
            print(f"No model-index found in {repo_id}")
            return

        model_index = card.data["model-index"]

        print(f"\nEvaluations for {repo_id}:")
        print("=" * 60)

        for model_entry in model_index:
            model_name = model_entry.get("name", "Unknown")
            print(f"\nModel: {model_name}")

            results = model_entry.get("results", [])
            for i, result in enumerate(results, 1):
                print(f"\n  Result Set {i}:")

                task = result.get("task", {})
                print(f"    Task: {task.get('type', 'unknown')}")

                dataset = result.get("dataset", {})
                print(f"    Dataset: {dataset.get('name', 'unknown')}")

                metrics = result.get("metrics", [])
                print(f"    Metrics ({len(metrics)}):")
                for metric in metrics:
                    name = metric.get("name", "Unknown")
                    value = metric.get("value", "N/A")
                    print(f"      - {name}: {value}")

                source = result.get("source", {})
                if source:
                    print(f"    Source: {source.get('name', 'Unknown')}")

        print("\n" + "=" * 60)

    except Exception as e:
        print(f"Error showing evaluations: {e}")


def validate_model_index(repo_id: str) -> bool:
    """Validate model-index format in a model card."""
    try:
        card = ModelCard.load(repo_id, token=HF_TOKEN)

        if "model-index" not in card.data:
            print(f"✗ No model-index found in {repo_id}")
            return False

        model_index = card.data["model-index"]

        if not isinstance(model_index, list):
            print("✗ model-index must be a list")
            return False

        for i, entry in enumerate(model_index):
            if "name" not in entry:
                print(f"✗ Entry {i} missing 'name' field")
                return False

            if "results" not in entry:
                print(f"✗ Entry {i} missing 'results' field")
                return False

            for j, result in enumerate(entry["results"]):
                if "task" not in result:
                    print(f"✗ Result {j} in entry {i} missing 'task' field")
                    return False

                if "dataset" not in result:
                    print(f"✗ Result {j} in entry {i} missing 'dataset' field")
                    return False

                if "metrics" not in result:
                    print(f"✗ Result {j} in entry {i} missing 'metrics' field")
                    return False

        print(f"✓ Model-index format is valid for {repo_id}")
        return True

    except Exception as e:
        print(f"Error validating model-index: {e}")
        return False


# ============================================================================
# CLI Interface
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Manage evaluation results in Hugging Face model cards"
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Extract from README command
    extract_parser = subparsers.add_parser(
        "extract-readme",
        help="Extract evaluation tables from model README"
    )
    extract_parser.add_argument("--repo-id", type=str, required=True, help="HF repository ID")
    extract_parser.add_argument("--task-type", type=str, default="text-generation", help="Task type")
    extract_parser.add_argument("--dataset-name", type=str, default="Benchmarks", help="Dataset name")
    extract_parser.add_argument("--dataset-type", type=str, default="benchmark", help="Dataset type")
    extract_parser.add_argument("--create-pr", action="store_true", help="Create PR instead of direct push")
    extract_parser.add_argument("--dry-run", action="store_true", help="Preview without updating")

    # Import from AA command
    aa_parser = subparsers.add_parser(
        "import-aa",
        help="Import evaluation scores from Artificial Analysis"
    )
    aa_parser.add_argument("--creator-slug", type=str, required=True, help="AA creator slug")
    aa_parser.add_argument("--model-name", type=str, required=True, help="AA model name")
    aa_parser.add_argument("--repo-id", type=str, required=True, help="HF repository ID")
    aa_parser.add_argument("--create-pr", action="store_true", help="Create PR instead of direct push")

    # Show evaluations command
    show_parser = subparsers.add_parser(
        "show",
        help="Display current evaluations in model card"
    )
    show_parser.add_argument("--repo-id", type=str, required=True, help="HF repository ID")

    # Validate command
    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate model-index format"
    )
    validate_parser.add_argument("--repo-id", type=str, required=True, help="HF repository ID")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Execute command
    if args.command == "extract-readme":
        results = extract_evaluations_from_readme(
            repo_id=args.repo_id,
            task_type=args.task_type,
            dataset_name=args.dataset_name,
            dataset_type=args.dataset_type
        )

        if not results:
            print("No evaluations extracted")
            return

        if args.dry_run:
            print("\nPreview of extracted evaluations:")
            print(yaml.dump({"model-index": [{"name": args.repo_id.split("/")[-1], "results": results}]}, sort_keys=False))
        else:
            update_model_card_with_evaluations(
                repo_id=args.repo_id,
                results=results,
                create_pr=args.create_pr,
                commit_message="Extract evaluation results from README"
            )

    elif args.command == "import-aa":
        results = import_aa_evaluations(
            creator_slug=args.creator_slug,
            model_name=args.model_name,
            repo_id=args.repo_id
        )

        if not results:
            print("No evaluations imported")
            return

        update_model_card_with_evaluations(
            repo_id=args.repo_id,
            results=results,
            create_pr=args.create_pr,
            commit_message=f"Add Artificial Analysis evaluations for {args.model_name}"
        )

    elif args.command == "show":
        show_evaluations(args.repo_id)

    elif args.command == "validate":
        validate_model_index(args.repo_id)


if __name__ == "__main__":
    main()
