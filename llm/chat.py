from __future__ import annotations

import getpass
import json
import socket
from pathlib import Path
from typing import Any

import typer
from pydantic import BaseModel
from rich.console import Console

from llm import slurm

CHAT_EXIT_COMMANDS = {"/exit", "/quit"}


class SessionPayload(BaseModel):
    session_id: str
    model: str
    system_prompt: str
    created_at_utc: str


class SessionMetaPayload(BaseModel):
    user: str
    submit_host: str
    submit_cwd: str
    last_updated_at_utc: str
    slurm_partition: str
    slurm_gres: str
    slurm_cpus_per_gpu: int
    slurm_mem: str
    slurm_time: str
    conda_env: str
    poll_interval: int
    max_context_chars: int


def _append_transcript_event(path: Path, *, role: str, text: str, turn: int) -> None:
    event = {
        "t": slurm.utc_now_iso(),
        "role": role,
        "text": text,
        "turn": turn,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False) + "\n")


def _read_transcript_events(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    if not path.exists():
        return events

    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            raw = raw.strip()
            if not raw:
                continue
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if not isinstance(parsed, dict):
                continue
            events.append(parsed)
    return events


def _build_context_prompt(transcript_path: Path, max_context_chars: int) -> str:
    role_prefix = {
        "system": "SYSTEM",
        "user": "USER",
        "assistant": "ASSISTANT",
    }
    events = _read_transcript_events(transcript_path)

    messages: list[str] = []
    for event in events:
        role = str(event.get("role", "")).strip().lower()
        text = str(event.get("text", ""))
        if role not in role_prefix:
            continue
        messages.append(f"{role_prefix[role]}: {text}")

    if not messages:
        return ""

    joined = "\n\n".join(messages)
    if len(joined) <= max_context_chars:
        return joined

    kept: list[str] = []
    total_len = 0
    for message in reversed(messages):
        extra = len(message) + (2 if kept else 0)
        if total_len + extra > max_context_chars:
            if not kept:
                kept.append(message[-max_context_chars:])
            break
        kept.append(message)
        total_len += extra

    kept.reverse()
    return "\n\n".join(kept)


def _write_session_meta(path: Path, payload: SessionMetaPayload) -> None:
    slurm.write_json(path, payload)


def register_chat_command(app: typer.Typer, console: Console) -> None:
    @app.command("chat")
    def chat(
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
        max_context_chars: int = typer.Option(
            12000,
            "--max-context-chars",
            help="Max context characters included in each turn prompt.",
        ),
    ) -> None:
        if max_context_chars <= 0:
            raise typer.BadParameter("--max-context-chars must be > 0.")

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
        session_dir = slurm.create_chat_session_dir(runs_root)
        session_dir.joinpath("turns").mkdir(parents=True, exist_ok=True)

        session_file = session_dir / "session.json"
        meta_file = session_dir / "meta.json"
        transcript_file = session_dir / "transcript.jsonl"

        created_at = slurm.utc_now_iso()
        session_payload = SessionPayload(
            session_id=session_dir.name,
            model=model_name,
            system_prompt=system_prompt,
            created_at_utc=created_at,
        )
        slurm.write_json(session_file, session_payload)

        session_meta = SessionMetaPayload(
            user=getpass.getuser(),
            submit_host=socket.gethostname(),
            submit_cwd=str(Path.cwd()),
            last_updated_at_utc=created_at,
            slurm_partition=slurm_options.partition,
            slurm_gres=slurm_options.gres,
            slurm_cpus_per_gpu=slurm_options.cpus_per_gpu,
            slurm_mem=slurm_options.mem,
            slurm_time=slurm_options.time_limit,
            conda_env=slurm_options.conda_env,
            poll_interval=slurm_options.poll_interval,
            max_context_chars=max_context_chars,
        )
        _write_session_meta(meta_file, session_meta)

        if system_prompt:
            _append_transcript_event(
                transcript_file,
                role="system",
                text=system_prompt,
                turn=0,
            )

        console.print(f"Chat session logs: {session_dir.resolve()}")
        console.print("Type /exit or /quit to end the session.")

        turn = 0
        while True:
            try:
                user_input = input("you> ")
            except EOFError:
                console.print("")
                break

            stripped = user_input.strip()
            if stripped in CHAT_EXIT_COMMANDS:
                break
            if stripped == "":
                continue

            turn += 1
            _append_transcript_event(
                transcript_file,
                role="user",
                text=user_input,
                turn=turn,
            )

            context_prompt = _build_context_prompt(transcript_file, max_context_chars)
            if not context_prompt:
                context_prompt = f"USER: {user_input}"

            turn_dir = slurm.create_chat_turn_dir(session_dir, turn)
            result = slurm.run_inference_turn(
                run_dir=turn_dir,
                model=model_name,
                prompt=context_prompt,
                system_prompt="",
                slurm=slurm_options,
                submit_cwd=Path.cwd(),
            )

            session_meta.last_updated_at_utc = slurm.utc_now_iso()
            _write_session_meta(meta_file, session_meta)

            if not result.success:
                console.print(
                    f"[red]Turn {turn} failed with state {result.state}. "
                    f"Inspect {result.stderr_log}[/red]"
                )
                if result.error_message:
                    console.print(result.error_message, style="red")
                continue

            assistant_text = result.answer_text.rstrip("\n")
            console.print(f"assistant> {assistant_text}", markup=False)
            _append_transcript_event(
                transcript_file,
                role="assistant",
                text=result.answer_text,
                turn=turn,
            )

        session_meta.last_updated_at_utc = slurm.utc_now_iso()
        _write_session_meta(meta_file, session_meta)
        console.print("Session ended.")
