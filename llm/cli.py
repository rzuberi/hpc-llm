from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from llm.chat import register_chat_command
from llm import slurm

app = typer.Typer(
    help="Submit Ollama inference jobs to SLURM and return the answer.",
    no_args_is_help=True,
)
console = Console(stderr=True)


@app.callback()
def callback() -> None:
    """LLM CLI command group."""


@app.command("ask")
def ask(
    prompt: str = typer.Argument(..., help="Prompt text to send to the model."),
    model: str | None = typer.Option(None, "--model", help="Ollama model name/tag."),
    system: str | None = typer.Option(None, "--system", help="Optional system prompt."),
    no_system: bool = typer.Option(False, "--no-system", help="Force empty system prompt."),
    partition: str | None = typer.Option(None, "--partition", help="SLURM partition."),
    gres: str | None = typer.Option(None, "--gres", help="SLURM --gres value."),
    cpus_per_gpu: int | None = typer.Option(None, "--cpus-per-gpu", help="SLURM --cpus-per-gpu value."),
    mem: str | None = typer.Option(None, "--mem", help="SLURM --mem value."),
    time_limit: str | None = typer.Option(None, "--time", help="SLURM --time value."),
    conda_env: str | None = typer.Option(None, "--conda-env", help="Conda env to activate inside SLURM job."),
    poll_interval: int | None = typer.Option(
        None,
        "--poll-interval",
        help="Polling interval in seconds for squeue/sacct.",
    ),
) -> None:
    slurm.ensure_slurm_commands()

    model_name = slurm.resolve_model_name(model)
    system_prompt = slurm.resolve_system_prompt(system, no_system)
    slurm_options = slurm.build_slurm_options(
        partition=partition,
        gres=gres,
        cpus_per_gpu=cpus_per_gpu,
        mem=mem,
        time_limit=time_limit,
        conda_env=conda_env,
        poll_interval=poll_interval,
    )

    runs_root = slurm.resolve_runs_root()
    run_dir = slurm.create_ask_run_dir(runs_root)

    result = slurm.run_inference_turn(
        run_dir=run_dir,
        model=model_name,
        prompt=prompt,
        system_prompt=system_prompt,
        slurm=slurm_options,
        submit_cwd=Path.cwd(),
        on_submitted=lambda job_id, path=run_dir: console.print(
            f"Submitted job [bold]{job_id}[/bold]. Run dir: {path}"
        ),
    )

    if not result.success:
        if result.job_id == "SUBMIT_FAILED":
            console.print("[red]Failed to submit job with sbatch.[/red]")
        else:
            console.print(
                f"[red]Job {result.job_id} ended with state {result.state}. "
                f"Inspect stderr log: {result.stderr_log}[/red]"
            )
        if result.error_message:
            console.print(result.error_message)
        raise typer.Exit(code=1)

    answer = result.answer_text
    typer.echo(answer, nl=not answer.endswith("\n"))


register_chat_command(app, console)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
