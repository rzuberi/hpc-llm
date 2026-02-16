# llm_ollama SLURM CLI

CLI for running Ollama on SLURM GPUs.

- `llm ask`: one-off prompt/response job.
- `llm chat`: chatbot REPL; each turn submits a new `sbatch` job and keeps conversation context.

## Setup

1. Create/update the conda env and install this project:

```bash
bash scripts/setup_env.sh
conda activate llm_ollama
```

2. Install Ollama user-space binary (no sudo):

```bash
bash scripts/install_ollama_user.sh
```

3. If needed, add Ollama to PATH:

```bash
export PATH="$HOME/.local/bin:$PATH"
```

## Usage

One-off ask:

```bash
llm ask "hello" --model llama3.1:8b --no-system
```

Continuous chat (repeated sbatch jobs, no `sinteractive` required):

```bash
llm chat --model llama3.1:8b --system "You are concise and technical."
```

In chat mode, type `/exit`, `/quit`, or Ctrl-D to end the session.

## Run Logs

All runs are written under `llm_runs/`.

One-off ask runs:

```text
llm_runs/ask/YYYYMMDD/HHMMSS_<rand>/
```

Each ask run includes:

- `request.json`
- `meta.json`
- `sbatch.sh`
- `stdout.log`
- `stderr.log`
- `answer.txt`

Chat sessions:

```text
llm_runs/chat/YYYYMMDD/HHMMSS_<rand>/
```

Each chat session includes:

- `session.json`
- `meta.json`
- `transcript.jsonl`
- `turns/0001_<rand>/...` (per-turn: `request.json`, `meta.json`, `sbatch.sh`, `stdout.log`, `stderr.log`, `answer.txt`)

## Defaults and Overrides

Defaults:

- partition: `cuda`
- resources: `--gres=gpu:1 --cpus-per-gpu=12 --mem=64G --time=00:15:00`
- model: `llama3.1:8b`

CLI flags override defaults per command.

Environment variable overrides:

- `LLM_DEFAULT_MODEL`
- `LLM_PARTITION`
- `LLM_GRES`
- `LLM_CPUS_PER_GPU`
- `LLM_MEM`
- `LLM_TIME`
- `LLM_CONDA_ENV` (default: `llm_ollama`)
- `LLM_CONDA_BASE` (optional explicit conda base path for compute-node activation)
- `LLM_RUNS_DIR` (default: `./llm_runs`)
- `LLM_POLL_INTERVAL` (seconds, default: `5`)
- `OLLAMA_MODELS` (model storage location)
