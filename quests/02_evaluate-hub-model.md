# Week 1: Evaluate a Hub Model

**Goal:** Add evaluation results to model cards across the Hub. Together, we're building a distributed leaderboard of open source model performance.

>[!NOTE]
> Bonus XP for contributing to the leaderboard application. Open a PR [on the hub](https://huggingface.co/spaces/humanitys-last-hackathon/distributed-leaderboard/discussions) or [on GitHub](https://github.com/huggingface/skills/blob/main/apps/evals-leaderboard/app.py) to get your XP.

## Why This Matters

Model cards without evaluation data are hard to compare. By adding structured eval results to `model-index` metadata, we make models searchable, sortable, and easier to choose between. Your contributions power leaderboards and help the community find the best models for their needs. Also, by doing this in a distributed way, we can share our evaluation results with the community.

## The Skill

Use `hf_model_evaluation/` for this quest. Key capabilities:

- Extract evaluation tables from existing README content
- Import benchmark scores from Artificial Analysis
- Run your own evals with inspect-ai on HF Jobs
- Update model-index metadata (Papers with Code compatible)

```bash
# Preview what would be extracted
python hf_model_evaluation/scripts/evaluation_manager.py extract-readme \
  --repo-id "model-author/model-name" --dry-run
```

## XP Tiers

### 🐢 Starter — 50 XP

**Extract evaluation results from one benchmark and update its model card.**

1. Pick a Hub model without evaluation data from *trending models* on the hub
2. Use the skill to extract or add a benchmark score
3. Create a PR (or push directly if you own the model)

**What counts:** One model, one dataset, metric visible in model card metadata.

### 🐕 Standard — 100 XP

**Import scores from third-party benchmarks like Artificial Analysis.**

1. Find a model with benchmark data on external sites
2. Use `import-aa` to fetch scores from Artificial Analysis API
3. Create a PR with properly attributed evaluation data

**What counts:** Undefined benchmark scores and merged PRs.

```bash
AA_API_KEY="your-key" python hf_model_evaluation/scripts/evaluation_manager.py import-aa \
  --creator-slug "anthropic" --model-name "claude-sonnet-4" \
  --repo-id "target/model" --create-pr
```

### 🦁 Advanced — 200 XP

**Run your own evaluation with inspect-ai and publish results.**

1. Choose an eval task (MMLU, GSM8K, HumanEval, etc.)
2. Run the evaluation on HF Jobs infrastructure
3. Update the model card with your results and methodology

**What counts:** Original eval run and merged PR.

```bash
HF_TOKEN=$HF_TOKEN hf jobs uv run hf_model_evaluation/scripts/inspect_eval_uv.py \
  --flavor a10g-small --secret HF_TOKEN=$HF_TOKEN \
  -- --model "meta-llama/Llama-2-7b-hf" --task "mmlu"
```

## Tips

- Always use `--dry-run` first to preview changes before pushing
- Check for transposed tables where models are rows and benchmarks are columns
- Be careful with PRs for models you don't own — most maintainers appreciate eval contributions but be respectful.
- Manually validate the extracted scores and close PRs if needed.

## Resources

- [SKILL.md](../hf_model_evaluation/SKILL.md) — Full skill documentation
- [Example Usage](../hf_model_evaluation/examples/USAGE_EXAMPLES.md) — Worked examples
- [Metric Mapping](../hf_model_evaluation/examples/metric_mapping.json) — Standard metric types

