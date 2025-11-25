# Hugging Face Skills

This repository contains skills for your coding agent to use when working with Hugging Face. It's compatible Claude Code, Google Gemini, and OpenAI Codex.

Skills are self-contained folders that package instructions, scripts, and resources. Each folder includes a `SKILL.md` file with YAML frontmatter (name and description) followed by the guidance your coding agent follows while the skill is active. 

> [!NOTE]
> 'Skills' is actually an Anthropic term used within Claude Code and not adopted by other agent tools, but we love it! OpenAI Codex uses an `AGENTS.md` file to define the instructions for your coding agent. Google Gemini uses 'extensions' to define the instructions for your coding agent in a `gemini-extension.json` file.

## 🎄 Humanity's Last Hackathon (of 2025)

**December 2025** — We're kicking off this skills library with a community hackathon. Use coding agents to ship real contributions: evaluate models, build datasets, fine-tune LLMs, and earn XP on a community leaderboard.

| Week | Quest | Skill |
|------|-------|-------|
| 1 | [Evaluate a Hub Model](quests/02_evaluate-hub-model.md) | `hf_model_evaluation/` |
| 2 | [Publish a Hub Dataset](quests/03_publish-hub-dataset.md) | `hf_dataset_creator/` |
| 3 | [Supervised Fine-Tuning](quests/04_sft-finetune-hub.md) | `hf-llm-trainer/` |

**Get started:** [Join the hackathon org](https://huggingface.co/organizations/humanitys-last-hackathon/share/KrqrmBxkETjvevFbfkXeezcyMbgMjjMaOp) → Read [quests/01_start.md](quests/01_start.md) → Pick a quest

---

## Using Hugging Face skills with your coding agent

### Repository layout

- `hf_dataset_creator/` – prompts, templates, and scripts for creating structured training data sets. Includes reusable templates in `templates/`, sample prompts in `examples/`, and orchestration code in `scripts/dataset_manager.py`.
- `hf_model_evaluation/` – instructions plus utilities for orchestrating evaluation jobs, generating reports, and mapping metrics. The `scripts/` folder hosts entry points such as `run_eval_job.py`, `evaluation_manager.py`, and supporting tools. Requirements live in `requirements.txt`.
- `hf-llm-trainer/` – a comprehensive training skill with `SKILL.md` guidance, helper scripts (e.g., `train_sft_example.py`, `convert_to_gguf.py`, cost estimators), and deep reference docs under `references/`.
- `hf-paper-publisher/` – tools for publishing and managing research papers on Hugging Face Hub. Index papers from arXiv, link papers to models/datasets, generate professional research articles from templates, and manage paper authorship. Includes paper templates (`standard`, `modern`, `arxiv`, `ml-report`) and citation generation utilities.
- `pyproject.toml` defines minimal tooling metadata for the repository, and `LICENSE` covers usage.

### Install these skills in Claude Code

1. Register the repository as a plugin marketplace:
   ```
   /plugin marketplace add huggingface/skills
   ```
2. In Claude Code, open **Browse and install plugins** → **huggingface-skills** (the marketplace you just added).
3. Choose the folder you want (`hf-llm-trainer`, `hf_model_evaluation`, `hf_dataset_creator`, or `hf-paper-publisher`) and select **Install now**.
4. Prefer commands? After registering the marketplace, run `/plugin install <skill-folder>@huggingface-skills` (for example, `/plugin install hf-llm-trainer@huggingface-skills`).

### Use these skills with Codex

Codex looks for long-lived instructions in `AGENTS.md`, so copy or symlink this repository’s `AGENTS.md` into your Codex profile (default `~/.codex/AGENTS.md`) or point `CODEX_HOME=$(pwd)/.codex` at the workspace before launching the CLI to scope the guidance per project. Codex rebuilds its instruction chain every run, so after updating the file you can verify the active guidance with `codex --ask-for-approval never "Summarize the current instructions."` or `codex --cd hf-llm-trainer --ask-for-approval never "Show which instruction files are active."` to confirm the Hugging Face skills are being read. Keep the skill folders accessible from that profile so Codex can reuse the templates, scripts, and references you mention inside the instructions. [Codex AGENTS guide](https://developers.openai.com/codex/guides/agents-md)

### Install the Gemini CLI extension

This repository already includes a `gemini-extension.json`, so you can treat it as a Gemini CLI extension when you prefer the Gemini tooling. Install it locally with `gemini extensions install /Users/ben/code/skills --consent` (or pass the GitHub URL) and restart Gemini CLI; the extension will appear as `huggingface-skills` because the CLI copies each extension into `~/.gemini/extensions`. Refresh it with `gemini extensions update huggingface-skills` when the repo changes, temporarily disable or scope it with `gemini extensions disable|enable huggingface-skills --scope workspace`, and, when you want live edits without reinstalling, run `gemini extensions link /Users/ben/code/skills` so the CLI reads directly from this working tree. [Gemini CLI extensions](https://geminicli.com/docs/extensions/#installing-an-extension)

### Use an installed skill

Once a skill is installed, mention it directly while giving your coding agent instructions:

- "Use the HF LLM trainer skill to estimate the GPU memory needed for a 70B model run."
- "Use the HF model evaluation skill to launch `run_eval_job.py` on the latest checkpoint."
- "Use the HF dataset creator skill to draft new few-shot classification templates."
- "Use the HF paper publisher skill to index my arXiv paper and link it to my model."

Your coding agent automatically loads the corresponding `SKILL.md` instructions and helper scripts while it completes the task.

### Contribute or customize a skill

1. Copy one of the existing skill folders (for example, `hf_dataset_creator/`) and rename it.
2. Update the new folder’s `SKILL.md` frontmatter:
   ```markdown
   ---
   name: my-skill-name
   description: Describe what the skill does and when to use it
   ---

   # Skill Title
   Guidance + examples + guardrails
   ```
3. Add or edit supporting scripts, templates, and documents referenced by your instructions.
4. Reinstall or reload the skill bundle in your coding agent so the updated folder is available.

### Additional references
- Browse the latest instructions, scripts, and templates directly at [huggingface/skills](https://github.com/huggingface/skills).
- Review Hugging Face documentation for the specific libraries or workflows you reference inside each skill.
