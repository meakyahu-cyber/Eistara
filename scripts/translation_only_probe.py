from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if os.fspath(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(PROJECT_ROOT))

from eistara.adapters.llm import OpenAICompatibleLlmClient, OpenAICompatibleSettings, RequestsHttpTransport
from eistara.config import ConfigLoader
from eistara.core.jobs import StageName
from eistara.core.pipeline import StageContext
from eistara.core.translation import PublishTranslationStageRunner
from eistara.core.translation.pacing import count_chinese_chars, count_english_words


DEFAULT_SAMPLES = [
    "Why Flying in America Got So Bad",
    "Why One Fast Food Chain Has Two Names",
    "How Do Gems Form",
    "How This Odd Shaped Cracker Beat Every Other Cracker",
    "How Nuclear Power Works",
    "The Entire History of Prehistoric Amazonia",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run translation only against archived source artifacts.")
    parser.add_argument("--history-dir", default="history")
    parser.add_argument("--config", default="config.local.yaml")
    parser.add_argument("--out-root", default="validation_runs")
    parser.add_argument("--name", default="")
    parser.add_argument("--sample", action="append", default=[])
    args = parser.parse_args()

    root = Path.cwd()
    history_dir = Path(args.history_dir)
    if not history_dir.is_absolute():
        history_dir = root / history_dir
    out_root = Path(args.out_root)
    if not out_root.is_absolute():
        out_root = root / out_root
    run_name = args.name or f"translation_prompt_probe_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = out_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    config = ConfigLoader(args.config).load()
    client = OpenAICompatibleLlmClient(
        OpenAICompatibleSettings(
            base_url=config.api.base_url,
            model=config.api.model,
            api_key=config.api.key,
            response_format_json=config.api.llm_support_json,
            timeout_sec=config.api.timeout_sec,
            user_agent=config.api.user_agent,
            trust_env_proxy=config.api.trust_env_proxy,
            proxy_url=config.api.proxy_url,
            max_retries=config.api.max_retries,
            retry_base_delay_sec=config.api.retry_base_delay_sec,
            retry_max_delay_sec=config.api.retry_max_delay_sec,
            persist_cache=False,
        ),
        RequestsHttpTransport(trust_env=config.api.trust_env_proxy, proxy_url=config.api.proxy_url),
    )
    runner = PublishTranslationStageRunner(client, config.translation_settings())

    samples = args.sample or DEFAULT_SAMPLES
    rows: list[dict[str, Any]] = []
    for sample in samples:
        history_job = _resolve_history_job(history_dir, sample)
        job_out = run_dir / _safe_dir_name(history_job.name)
        output_dir = job_out / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        previous_job_dir: str | None = None
        previous_output_dir: str | None = None
        try:
            task = _task_for_history_job(history_job, output_dir)
            previous_job_dir = os.environ.get("EISTARA_JOB_DIR")
            previous_output_dir = os.environ.get("EISTARA_OUTPUT_DIR")
            os.environ["EISTARA_JOB_DIR"] = os.fspath(job_out)
            os.environ["EISTARA_OUTPUT_DIR"] = os.fspath(output_dir)
            try:
                result = runner.run(
                    StageContext(
                        job_id=history_job.name,
                        job_dir=job_out,
                        task=task,
                        stage=StageName.TRANSLATE,
                        attempt=1,
                        config=config.raw,
                        artifacts={},
                    )
                )
            finally:
                _restore_env("EISTARA_JOB_DIR", previous_job_dir)
                _restore_env("EISTARA_OUTPUT_DIR", previous_output_dir)
            if result.status != "done":
                raise RuntimeError(f"Translation did not finish: {result.status} {result.warnings}")
            row = _metrics(history_job, output_dir)
        except Exception as exc:
            row = {
                "name": history_job.name,
                "status": "failed",
                "error": str(exc),
            }
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    (run_dir / "summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (run_dir / "summary.md").write_text(_markdown(rows), encoding="utf-8")
    print(f"summary: {run_dir / 'summary.md'}")
    return 0


def _task_for_history_job(history_job: Path, output_dir: Path) -> dict[str, Any]:
    log_dir = history_job / "output" / "log"
    task_path = history_job / "task.json"
    task_data: dict[str, Any] = {}
    if task_path.exists():
        task_data = json.loads(task_path.read_text(encoding="utf-8"))
    task = {
        "id": f"translation_probe_{_safe_dir_name(history_job.name)}",
        "title": history_job.name,
        "output_dir": os.fspath(output_dir),
        "publish_source_lines": os.fspath(log_dir / "publish_source_lines.txt"),
        "cleaned_chunks": os.fspath(log_dir / "cleaned_chunks.xlsx"),
        "terminology_json": os.fspath(log_dir / "terminology.json"),
        "source_language": task_data.get("source_language") or "en",
        "target_language": task_data.get("target_language") or "zh",
    }
    for key in ("publish_source_lines", "cleaned_chunks", "terminology_json"):
        if not Path(task[key]).exists():
            raise FileNotFoundError(f"Missing {key} for {history_job.name}: {task[key]}")
    return task


def _metrics(history_job: Path, output_dir: Path) -> dict[str, Any]:
    df = pd.read_excel(output_dir / "log" / "publish_translation.xlsx")
    source_text = "".join(str(value) for value in df["Source"])
    translated_text = "".join(str(value) for value in df["Translation"])
    english_words = sum(count_english_words(str(value)) for value in df["Source"])
    han_chars = count_chinese_chars(translated_text)
    target_nonspace_chars = len(re.sub(r"\s+", "", translated_text))
    source_nonspace_chars = len(re.sub(r"\s+", "", source_text))
    report = json.loads((output_dir / "log" / "publish_translate_report.json").read_text(encoding="utf-8"))
    enabled = []
    for batch in report.get("batches") or []:
        pacing = dict(batch.get("pacing") or {})
        if pacing.get("enabled") is True:
            enabled.append(pacing)
    return {
        "name": history_job.name,
        "english_words": english_words,
        "han_chars": han_chars,
        "han_per_en_word": round(han_chars / english_words, 3) if english_words else None,
        "target_nonspace_chars": target_nonspace_chars,
        "target_nonspace_per_en_word": round(target_nonspace_chars / english_words, 3) if english_words else None,
        "target_nonspace_per_source_nonspace": (
            round(target_nonspace_chars / source_nonspace_chars, 3) if source_nonspace_chars else None
        ),
        "enabled_batches": len(enabled),
        "enabled_below_min": sum(
            1
            for pacing in enabled
            if int(pacing.get("actual_zh_chars") or 0) < int(pacing.get("min_zh_chars") or 0)
        ),
        "enabled_actual_minus_min": sum(
            int(pacing.get("actual_zh_chars") or 0) - int(pacing.get("min_zh_chars") or 0)
            for pacing in enabled
        ),
    }


def _markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Translation Prompt Probe",
        "",
        "| job | status | EN words | Han chars | Han/EN | total chars/EN | enabled below min | enabled actual-min | error |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        if row.get("status") == "failed":
            lines.append(f"| {row['name']} | failed |  |  |  |  |  |  | {row.get('error', '')} |")
            continue
        lines.append(
            f"| {row['name']} | done | {row['english_words']} | {row['han_chars']} | "
            f"{row['han_per_en_word']} | {row['target_nonspace_per_en_word']} | "
            f"{row['enabled_below_min']}/{row['enabled_batches']} | {row['enabled_actual_minus_min']} |"
            " |"
        )
    return "\n".join(lines) + "\n"


def _resolve_history_job(history_dir: Path, name: str) -> Path:
    direct = history_dir / name
    if direct.exists():
        return direct
    lowered = name.casefold()
    matches = [path for path in history_dir.iterdir() if path.is_dir() and lowered in path.name.casefold()]
    if not matches:
        matches = [path for path in history_dir.iterdir() if path.is_dir() and path.name.casefold().startswith(lowered)]
    if len(matches) != 1:
        raise FileNotFoundError(f"Could not resolve history job {name!r}: {matches}")
    return matches[0]


def _safe_dir_name(name: str) -> str:
    safe = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", name).strip(" .")
    return safe or "job"


def _restore_env(name: str, value: str | None) -> None:
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value


if __name__ == "__main__":
    raise SystemExit(main())
