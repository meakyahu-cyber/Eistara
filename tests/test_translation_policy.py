from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from eistara.core.jobs import StageName
from eistara.core.jobs.store import STATE_FILE, TASK_FILE
from eistara.core.pipeline import StageContext
from eistara.core.scheduler import SchedulerService
from eistara.core.translation import (
    PublishTranslationStageRunner,
    PublishTranslationService,
    TranslationItem,
    TranslationSettings,
    TranslationStageRunner,
    has_excess_latin_text,
    split_batches,
)
from eistara.core.translation.llm import ScriptedLlmClient
from eistara.core.translation.pacing import build_pacing_budget


def test_latin_validator_allows_technical_tokens_and_units() -> None:
    text = "这款 Micron 6600 ion NVMe SSD 支持 122TB，并面向 PB 级数据中心。"

    assert has_excess_latin_text(text, target_language="Simplified Chinese") is False


def test_latin_validator_matches_v1_chatgpt_allowlist() -> None:
    text = "这里保留 ChatGPT 这个产品名。"

    assert has_excess_latin_text(text, target_language="Simplified Chinese") is False


def test_latin_validator_catches_long_english_residue() -> None:
    text = "这句话没有翻译，and this part is still a long English sentence left behind."

    assert has_excess_latin_text(text, target_language="Simplified Chinese") is True


def test_latin_validator_allows_domains_and_title_case_names() -> None:
    text = "The Real Engineering Channel and The Anatomy of Things: go.nebula.tv/practicalengineering"

    assert has_excess_latin_text(text, target_language="Simplified Chinese") is False


def test_latin_validator_allows_short_lowercase_loanword_terms() -> None:
    text = "nasi lemak, bak kut teh"

    assert has_excess_latin_text(text, target_language="Simplified Chinese") is False


def test_latin_validator_still_catches_sentence_case_english_residue() -> None:
    text = "This is still a long untranslated English sentence left inside the Chinese translation."

    assert has_excess_latin_text(text, target_language="Simplified Chinese") is True


def test_split_batches_respects_line_and_char_limits() -> None:
    items = [TranslationItem(id=index, source="x" * 50) for index in range(1, 6)]
    settings = TranslationSettings(max_batch_lines=2, max_batch_chars=200)

    batches = split_batches(items, settings)

    assert [len(batch) for batch in batches] == [2, 2, 1]


def test_pacing_budget_enables_length_constraint_for_dense_batches() -> None:
    items = [
        TranslationItem(
            id=1,
            source=" ".join(["word"] * 80),
            start="00:00:00,000",
            end="00:00:20,000",
            duration_sec=20,
        )
    ]
    settings = TranslationSettings(use_summary=False, min_pacing_source_sec=1, pacing_budget_enabled=True)

    budget = build_pacing_budget(items, settings)

    assert budget.enabled is True
    assert budget.level in {"hard", "critical"}
    assert budget.min_zh_chars == 134
    assert budget.ideal_min_zh_chars == 92
    assert budget.ideal_max_zh_chars == 96
    assert budget.target_zh_chars == 136
    assert budget.hard_zh_chars == 136
    assert budget.max_zh_chars_per_en_word == 1.70
    assert budget.pressure_at_min_zh_ratio > budget.soft_pressure


def test_pacing_budget_does_not_limit_low_pressure_batches() -> None:
    items = [
        TranslationItem(
            id=1,
            source=" ".join(["word"] * 40),
            start="00:00:00,000",
            end="00:00:30,000",
            duration_sec=30,
        )
    ]
    settings = TranslationSettings(use_summary=False, min_pacing_source_sec=1, pacing_budget_enabled=True)

    budget = build_pacing_budget(items, settings)

    assert budget.enabled is False
    assert budget.level == "short"
    assert budget.reason == "natural_language_priority_pressure_below_watch_limit"
    assert budget.min_zh_chars == 0
    assert budget.target_zh_chars == 0
    assert budget.hard_zh_chars == 0


def test_pacing_budget_only_warns_without_counts_for_watch_pressure_batches() -> None:
    items = [
        TranslationItem(
            id=1,
            source=" ".join(["word"] * 55),
            start="00:00:00,000",
            end="00:00:20,000",
            duration_sec=20,
        )
    ]
    settings = TranslationSettings(use_summary=False, min_pacing_source_sec=1, pacing_budget_enabled=True)

    budget = build_pacing_budget(items, settings)

    assert 1.05 <= budget.predicted_pressure < 1.10
    assert budget.enabled is False
    assert budget.reason == "watch_pressure_avoid_unnecessary_expansion"
    assert budget.target_zh_chars == 0
    assert budget.hard_zh_chars == 0


def test_pacing_budget_uses_graded_upper_limits_for_high_pressure_batches() -> None:
    settings = TranslationSettings(use_summary=False, min_pacing_source_sec=1, pacing_budget_enabled=True)

    soft = build_pacing_budget([TranslationItem(id=1, source=" ".join(["word"] * 57), duration_sec=20)], settings)
    hard = build_pacing_budget([TranslationItem(id=1, source=" ".join(["word"] * 62), duration_sec=20)], settings)
    critical = build_pacing_budget([TranslationItem(id=1, source=" ".join(["word"] * 66), duration_sec=20)], settings)

    assert 1.10 <= soft.predicted_pressure < 1.18
    assert soft.enabled is True
    assert soft.target_zh_chars_per_en_word == 1.72
    assert soft.max_zh_chars_per_en_word == 1.75
    assert soft.target_zh_chars == 99
    assert soft.hard_zh_chars == 100

    assert 1.18 <= hard.predicted_pressure < 1.25
    assert hard.enabled is True
    assert hard.target_zh_chars_per_en_word == 1.72
    assert hard.max_zh_chars_per_en_word == 1.73
    assert hard.target_zh_chars == 107
    assert hard.hard_zh_chars == 108

    assert critical.predicted_pressure >= 1.25
    assert critical.enabled is True
    assert critical.target_zh_chars_per_en_word == 1.70
    assert critical.max_zh_chars_per_en_word == 1.70
    assert critical.target_zh_chars == 113
    assert critical.hard_zh_chars == 113


def test_publish_prompt_includes_pacing_budget_for_high_pressure_batch() -> None:
    client = ScriptedLlmClient([{"translations": [{"id": 1, "text": "短句"}]}])
    service = PublishTranslationService(
        client,
        TranslationSettings(use_summary=False, enforce_latin=False, min_pacing_source_sec=1, pacing_budget_enabled=True),
    )
    item = TranslationItem(id=1, source=" ".join(["word"] * 80), duration_sec=20)

    service.translate([item])

    prompt = client.calls[0]["prompt"]
    assert "Dubbing pacing budget:" in prompt
    assert "target_total_chinese_chars" in prompt
    assert "suggested_zh_chars" in prompt
    assert "this is a per-batch lock" in prompt
    assert "minimum is a quality floor" in prompt
    assert "Count only Chinese Han characters" in prompt
    assert "ideal_pressure_chinese_chars" not in prompt


def test_publish_prompt_omits_suggested_counts_for_low_pressure_batch() -> None:
    client = ScriptedLlmClient([{"translations": [{"id": 1, "text": "鑷劧琛ㄨ揪"}]}])
    service = PublishTranslationService(
        client,
        TranslationSettings(use_summary=False, enforce_latin=False, min_pacing_source_sec=1),
    )
    item = TranslationItem(id=1, source=" ".join(["word"] * 40), duration_sec=30)

    service.translate([item])

    prompt = client.calls[0]["prompt"]
    assert "Dubbing pacing budget" not in prompt
    assert "pacing" not in prompt.lower()
    assert "target_total_chinese_chars" not in prompt
    assert "suggested_zh_chars" not in prompt


def test_publish_prompt_warns_without_counts_for_watch_pressure_batch() -> None:
    client = ScriptedLlmClient([{"translations": [{"id": 1, "text": "鑷劧琛ㄨ揪"}]}])
    service = PublishTranslationService(
        client,
        TranslationSettings(use_summary=False, enforce_latin=False, min_pacing_source_sec=1),
    )
    item = TranslationItem(id=1, source=" ".join(["word"] * 55), duration_sec=20)

    service.translate([item])

    prompt = client.calls[0]["prompt"]
    assert "Dubbing pacing budget" not in prompt
    assert "pacing" not in prompt.lower()
    assert "target_total_chinese_chars" not in prompt
    assert "suggested_zh_chars" not in prompt


def test_single_line_fallback_bypasses_cache_after_latin_failure(monkeypatch) -> None:
    monkeypatch.setattr("eistara.core.translation.service.time.sleep", lambda _seconds: None)
    client = ScriptedLlmClient(
        [
            {"translations": [{"id": 1, "text": "This is still a long untranslated English sentence"}]},
            {"translations": [{"id": 1, "text": "This is still a long untranslated English sentence"}]},
            {"translations": [{"id": 1, "text": "This is still a long untranslated English sentence"}]},
            {"translations": [{"id": 1, "text": "This is still a long untranslated English sentence"}]},
            {"translations": [{"id": 1, "text": "This is still a long untranslated English sentence"}]},
            {"translations": [{"id": 1, "text": "This is still a long untranslated English sentence"}]},
            {"translations": [{"id": 1, "text": "这是一句可接受的翻译，保留 NVMe。"}]},
        ]
    )
    service = PublishTranslationService(client)

    result = service.translate([TranslationItem(id=1, source="This is a sentence about NVMe.")])

    assert result.translations[1] == "这是一句可接受的翻译，保留 NVMe。"
    assert all(call["use_cache"] is True for call in client.calls[:6])
    assert client.calls[6]["use_cache"] is False
    assert result.warnings


def test_batch_failure_splits_and_recovers(monkeypatch) -> None:
    monkeypatch.setattr("eistara.core.translation.service.time.sleep", lambda _seconds: None)
    client = ScriptedLlmClient(
        [
            ValueError("temporary batch failure"),
            ValueError("temporary batch failure"),
            ValueError("temporary batch failure"),
            ValueError("temporary batch failure"),
            ValueError("temporary batch failure"),
            ValueError("temporary batch failure"),
            {"translations": [{"id": 1, "text": "第一句"}]},
            {"translations": [{"id": 2, "text": "第二句"}]},
        ]
    )
    service = PublishTranslationService(client)

    result = service.translate([TranslationItem(id=1, source="one"), TranslationItem(id=2, source="two")])

    assert result.translations == {1: "第一句", 2: "第二句"}
    assert len(client.calls) == 8


def test_rate_limit_batch_failure_does_not_split(monkeypatch) -> None:
    monkeypatch.setattr("eistara.core.translation.service.time.sleep", lambda _seconds: None)
    client = ScriptedLlmClient(
        [
            RuntimeError(
                'LLM 429 request error: {"error":{"message":"Concurrency limit exceeded for account, please retry later"}}'
            )
        ]
    )
    service = PublishTranslationService(client)

    try:
        service.translate([TranslationItem(id=1, source="one"), TranslationItem(id=2, source="two")])
    except RuntimeError as exc:
        assert "429" in str(exc)
    else:
        raise AssertionError("rate limit errors must fail without batch splitting")

    assert len(client.calls) == 1


def test_translation_stage_runner_updates_scheduler_outputs(tmp_path: Path) -> None:
    jobs_dir = tmp_path / "jobs"
    job_dir = jobs_dir / "job_0001_translate"
    job_dir.mkdir(parents=True)
    (job_dir / TASK_FILE).write_text(
        json.dumps(
            {
                "id": job_dir.name,
                "translation_items": [{"id": 1, "source": "Hello world"}],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (job_dir / STATE_FILE).write_text(
        json.dumps(
            {
                "job_id": job_dir.name,
                "status": "pending",
                "current_stage": None,
                "completed_stages": ["download", "transcribe"],
                "failed_stage": None,
                "attempts": {},
                "error": None,
                "artifacts": {},
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    service = SchedulerService(jobs_dir)
    service.register(TranslationStageRunner(ScriptedLlmClient([{"translations": [{"id": 1, "text": "你好，世界"}]}])))

    assert service.run_one_ready_stage() is True
    state = json.loads((job_dir / STATE_FILE).read_text(encoding="utf-8"))
    assert state["completed_stages"] == ["download", "transcribe", "translate"]
    assert state["artifacts"]["translation_count"] == 1


def test_publish_translation_stage_runner_reads_subtitle_rows_and_writes_tts_segments(tmp_path: Path) -> None:
    subtitle_rows_json = tmp_path / "subtitle_rows.json"
    subtitle_rows_json.write_text(
        json.dumps(
            {
                "rows": [
                    {"start_sec": 0.0, "end_sec": 1.5, "source": "Hello world", "target": "", "speaker": "SPEAKER_01"},
                    {"start_sec": 2.0, "end_sec": 3.0, "source": "Good night", "target": ""},
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    runner = PublishTranslationStageRunner(
        ScriptedLlmClient(
            [
                {"theme": "Greeting video", "terms": [{"src": "Hello", "tgt": "你好", "note": "greeting"}]},
                {
                    "translations": [
                        {"id": 1, "text": "你好，世界"},
                        {"id": 2, "text": "晚安"},
                    ]
                }
            ]
        )
    )

    result = runner.run(
        StageContext(
            job_id="job",
            job_dir=tmp_path,
            task={"subtitle_rows_json": str(subtitle_rows_json)},
            stage=StageName.TRANSLATE,
            attempt=1,
        )
    )

    translations_json = Path(result.outputs["translations_json"])
    assert translations_json.exists()
    assert translations_json == tmp_path / "output" / "internal" / "translations.json"
    assert Path(result.outputs["translation"]).exists()
    assert Path(result.outputs["subtitles"]).exists()
    assert Path(result.outputs["audio_script"]).exists()
    assert Path(result.outputs["publish_source_lines"]).read_text(encoding="utf-8").splitlines() == ["Hello world", "Good night"]
    assert Path(result.outputs["source_srt"]).exists()
    assert Path(result.outputs["translated_srt"]).exists()
    assert Path(result.outputs["audio_source_srt"]).exists()
    assert Path(result.outputs["audio_translated_srt"]).exists()
    assert Path(result.outputs["publish_translate_report"]).exists()
    report = json.loads(Path(result.outputs["publish_translate_report"]).read_text(encoding="utf-8"))
    assert report["summary"]["batch_count"] == 1
    assert report["summary"]["totals"]["source_items"] == 2
    assert report["summary"]["totals"]["english_words"] == 4
    assert report["summary"]["totals"]["source_duration_sec"] == 2.5
    assert report["summary"]["totals"]["actual_zh_chars"] > 0
    assert report["summary"]["pacing"]["predicted_pressure"]["max"] is not None
    assert pd.read_excel(result.outputs["translation"]).to_dict(orient="records")[0]["Source"] == "Hello world"
    assert result.outputs["translation_count"] == 2
    assert result.outputs["translations"] == {1: "你好，世界", 2: "晚安"}
    assert result.outputs["tts_segments_count"] == 2
    assert result.outputs["tts_segments"][0]["source"] == "Hello world"
    assert result.outputs["tts_segments"][0]["speaker"] == "SPEAKER_01"
    assert result.outputs["tts_segments"][0]["metadata"]["speaker"] == "SPEAKER_01"
    assert result.outputs["tts_segments"][0]["text"] == "你好，世界"
    assert result.outputs["tts_segments"][0]["output_path"].endswith("output\\audio\\tmp\\1_0_temp.wav") or result.outputs["tts_segments"][0][
        "output_path"
    ].endswith("output/audio/tmp/1_0_temp.wav")


def test_publish_translation_stage_runner_prefers_timed_subtitle_rows_over_plain_source_lines(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    log_dir = output_dir / "log"
    log_dir.mkdir(parents=True)
    publish_source_lines = log_dir / "publish_source_lines.txt"
    publish_source_lines.write_text("Hello world\nGood night\n", encoding="utf-8")
    pd.DataFrame(
        [
            {"text": '"Hello"', "start": 0.0, "end": 0.4, "speaker_id": None},
            {"text": '"world"', "start": 0.4, "end": 1.0, "speaker_id": None},
            {"text": '"Good"', "start": 2.0, "end": 2.4, "speaker_id": None},
            {"text": '"night"', "start": 2.4, "end": 3.0, "speaker_id": None},
        ]
    ).to_excel(log_dir / "cleaned_chunks.xlsx", index=False)
    subtitle_rows_json = tmp_path / "subtitle_rows.json"
    subtitle_rows_json.write_text(
        json.dumps(
            {
                "rows": [
                    {"start_sec": 9.0, "end_sec": 10.0, "source": "Timed hello", "target": ""},
                    {"start_sec": 11.0, "end_sec": 12.5, "source": "Timed night", "target": ""},
                ]
            }
        ),
        encoding="utf-8",
    )
    runner = PublishTranslationStageRunner(
        ScriptedLlmClient(
            [
                {"theme": "Night video", "terms": [{"src": "Good night", "tgt": "晚安", "note": "farewell"}]},
                {
                    "translations": [
                        {"id": 1, "text": "Line one"},
                        {"id": 2, "text": "Line two"},
                    ]
                }
            ]
        )
    )

    result = runner.run(
        StageContext(
            job_id="job",
            job_dir=tmp_path,
            task={"output_dir": str(output_dir), "subtitle_rows_json": str(subtitle_rows_json)},
            stage=StageName.TRANSLATE,
            attempt=1,
        )
    )

    assert Path(result.outputs["publish_source_lines"]).read_text(encoding="utf-8").splitlines() == ["Timed hello", "Timed night"]
    assert result.outputs["tts_segments_count"] == 2
    assert result.outputs["tts_segments"][0]["source"] == "Timed hello"
    assert result.outputs["tts_segments"][0]["start"] == "00:00:09,000"
    assert result.outputs["tts_segments"][0]["end"] == "00:00:10,000"
    assert "00:00:09,000 --> 00:00:10,000" in Path(result.outputs["source_srt"]).read_text(encoding="utf-8")
    report = json.loads(Path(result.outputs["publish_translate_report"]).read_text(encoding="utf-8"))
    assert report["summary"]["totals"]["source_duration_sec"] == 2.5


def test_publish_translation_stage_runner_generates_v1_summary_and_appends_custom_terms(tmp_path: Path) -> None:
    custom_terms = tmp_path / "custom_terms.xlsx"
    pd.DataFrame([{"src": "OpenAI", "tgt": "OpenAI", "note": "brand name"}]).to_excel(custom_terms, index=False)
    client = ScriptedLlmClient(
        [
            {"theme": "AI video", "terms": [{"src": "GPU", "tgt": "GPU", "note": "hardware"}]},
            {"translations": [{"id": 1, "text": "Translated line"}]},
        ]
    )
    runner = PublishTranslationStageRunner(
        client,
        TranslationSettings(target_language="English", summary_length=11, enforce_latin=False),
    )

    result = runner.run(
        StageContext(
            job_id="job",
            job_dir=tmp_path,
            task={
                "translation_items": [{"id": 1, "source": "Hello world from OpenAI"}],
                "custom_terms": str(custom_terms),
            },
            artifacts={"language": "en"},
            stage=StageName.TRANSLATE,
            attempt=1,
        )
    )

    terminology_path = Path(result.outputs["terminology_json"])
    terminology = json.loads(terminology_path.read_text(encoding="utf-8"))
    assert terminology["theme"] == "AI video"
    assert terminology["terms"] == [
        {"src": "GPU", "tgt": "GPU", "note": "hardware"},
        {"src": "OpenAI", "tgt": "OpenAI", "note": "brand name"},
    ]
    assert [call["log_title"] for call in client.calls] == ["summary", "translate_publish_fast"]
    assert "specializing in en comprehension" in client.calls[0]["prompt"]
    assert "<text>\nHello world\n</text>" in client.calls[0]["prompt"]
    assert "OpenAI: OpenAI (brand name)" in client.calls[0]["prompt"]


def test_publish_translation_stage_runner_uses_existing_terminology_without_summary_call(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    log_dir = output_dir / "log"
    log_dir.mkdir(parents=True)
    terminology_json = log_dir / "terminology.json"
    terminology_json.write_text(
        json.dumps({"theme": "Existing theme", "terms": [{"src": "SSD", "tgt": "SSD", "note": "storage"}]}),
        encoding="utf-8",
    )
    client = ScriptedLlmClient([{"translations": [{"id": 1, "text": "Translated line"}]}])
    runner = PublishTranslationStageRunner(
        client,
        TranslationSettings(target_language="English", enforce_latin=False),
    )

    result = runner.run(
        StageContext(
            job_id="job",
            job_dir=tmp_path,
            task={"output_dir": str(output_dir), "translation_items": [{"id": 1, "source": "SSD test"}]},
            stage=StageName.TRANSLATE,
            attempt=1,
        )
    )

    assert result.outputs["terminology_json"] == str(terminology_json)
    assert [call["log_title"] for call in client.calls] == ["translate_publish_fast"]
    assert "Existing theme" in client.calls[0]["prompt"]
