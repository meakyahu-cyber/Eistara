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
    build_dubbing_length_constraints,
    build_localization_constraints,
    has_excess_latin_text,
    split_batches,
)
from eistara.core.translation.llm import ScriptedLlmClient
from eistara.core.translation.pacing import estimate_spoken_cost_units
from eistara.core.timeline import build_source_windows


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


def test_spoken_cost_counts_numbers_percentages_and_latin_tokens() -> None:
    assert estimate_spoken_cost_units("到2023年，开车仅22%。") > 5
    assert estimate_spoken_cost_units("Shrek 2。") > 0
    assert estimate_spoken_cost_units("AI API") >= 5


def test_localization_hard_cap_reserves_seam_gap() -> None:
    item = TranslationItem(id=1, source="hello", duration_sec=10)
    settings = TranslationSettings(
        localization_chars_per_sec=4.2,
        localization_max_audio_speed=1.10,
        localization_seam_gap_sec=0.12,
    )

    constraints = build_localization_constraints([item], {1: "\u4e00" * 60}, settings)

    assert constraints[0].usable_window_sec == 9.88
    assert constraints[0].max_chinese_chars == 45
    assert constraints[0].max_spoken_cost == 39
    assert constraints[0].over_limit is True


def test_localization_over_budget_sets_spoken_cost_floor() -> None:
    item = TranslationItem(id=1, source="hello", duration_sec=7.523)
    settings = TranslationSettings(
        localization_chars_per_sec=4.2,
        localization_spoken_cost_per_sec=3.6,
        localization_max_audio_speed=1.10,
        localization_seam_gap_sec=0.12,
    )

    constraints = build_localization_constraints([item], {1: "\u4e00" * 33}, settings)

    assert constraints[0].max_spoken_cost == 29
    assert constraints[0].draft_spoken_cost == 33
    assert constraints[0].min_spoken_cost == 27
    assert constraints[0].over_limit is True
    prompt_window = constraints[0].to_prompt_dict()
    assert prompt_window["min_spoken_cost"] == 27
    assert prompt_window["max_spoken_cost"] == 29


def test_first_pass_dubbing_length_constraint_uses_target_and_hard_caps() -> None:
    item = TranslationItem(id=1, source="hello", duration_sec=10)
    settings = TranslationSettings(
        localization_chars_per_sec=4.2,
        localization_max_audio_speed=1.10,
        localization_seam_gap_sec=0.12,
    )

    constraints = build_dubbing_length_constraints([item], settings)

    assert constraints[0].usable_window_sec == 9.88
    assert constraints[0].target_chinese_chars == 41
    assert constraints[0].max_chinese_chars == 45
    assert constraints[0].target_spoken_cost == 35
    assert constraints[0].max_spoken_cost == 39
    prompt_window = constraints[0].to_prompt_dict()
    assert prompt_window["hard_limit_applies"] is True
    assert prompt_window["max_spoken_cost"] == 39
    assert "target_chinese_chars" not in prompt_window
    assert "max_chinese_chars" not in prompt_window
    assert "chars_per_sec" not in prompt_window


def test_source_window_uses_gap_until_next_start() -> None:
    items = [
        TranslationItem(id=1, source="first", start=0, end=7, duration_sec=7),
        TranslationItem(id=2, source="second", start=10, end=12, duration_sec=2),
    ]

    windows = build_source_windows(items, max_gap_after_sec=6.0)
    constraints = build_localization_constraints(
        [items[0]],
        {1: "\u4e00" * 60},
        TranslationSettings(
            localization_chars_per_sec=4.2,
            localization_max_audio_speed=1.10,
            localization_seam_gap_sec=0.12,
        ),
        source_windows=windows,
    )

    assert windows[1].source_duration_sec == 7
    assert windows[1].owned_gap_after_sec == 3
    assert windows[1].window_duration_sec == 10
    assert constraints[0].window_duration_sec == 10
    assert constraints[0].max_chinese_chars == 45


def test_source_window_caps_long_gap_after_segment() -> None:
    items = [
        TranslationItem(id=1, source="first", start=0, end=7, duration_sec=7),
        TranslationItem(id=2, source="second", start=30, end=32, duration_sec=2),
    ]

    windows = build_source_windows(items, max_gap_after_sec=6.0)

    assert windows[1].owned_gap_after_sec == 6
    assert windows[1].window_duration_sec == 13


def test_localization_without_timing_is_capped_by_first_pass_length() -> None:
    item = TranslationItem(id=1, source="hello")

    constraints = build_localization_constraints([item], {1: "\u4e00" * 12}, TranslationSettings())

    assert constraints[0].window_duration_sec is None
    assert constraints[0].draft_chinese_chars == 12
    assert constraints[0].max_chinese_chars == 12
    assert constraints[0].to_prompt_dict()["hard_limit_applies"] is True


def test_localization_timing_cap_never_exceeds_first_pass_length() -> None:
    item = TranslationItem(id=1, source="hello", duration_sec=10)

    constraints = build_localization_constraints(
        [item],
        {1: "\u4e00" * 12},
        TranslationSettings(
            localization_chars_per_sec=4.2,
            localization_max_audio_speed=1.10,
            localization_seam_gap_sec=0.12,
        ),
    )

    assert constraints[0].window_duration_sec == 10
    assert constraints[0].draft_chinese_chars == 12
    assert constraints[0].max_chinese_chars == 12
    assert constraints[0].over_limit is False


def test_localization_second_pass_runs_after_first_pass() -> None:
    item = TranslationItem(id=1, source="hello world", duration_sec=10)
    client = ScriptedLlmClient(
        [
            {"translations": [{"id": 1, "text": "\u4e00" * 39}]},
            {"translations": [{"id": 1, "text": "\u4e8c" * 20}]},
        ]
    )
    service = PublishTranslationService(
        client,
        TranslationSettings(
            use_summary=False,
            enforce_latin=False,
            localization_chars_per_sec=4.2,
            localization_max_audio_speed=1.10,
            localization_seam_gap_sec=0.12,
        ),
    )

    result = service.translate([item])

    assert result.translations[1] == "\u4e8c" * 20
    assert len(client.calls) == 2
    assert client.calls[0]["log_title"] == "translate_publish_fast"
    assert "dubbing_window" in client.calls[0]["prompt"]
    assert '"target_chinese_chars"' not in client.calls[0]["prompt"]
    assert '"max_chinese_chars"' not in client.calls[0]["prompt"]
    assert '"max_spoken_cost": 39' in client.calls[0]["prompt"]
    assert client.calls[1]["log_title"] == "translate_publish_localize"
    assert "max_spoken_cost" in client.calls[1]["prompt"]
    assert "max_chinese_chars" not in client.calls[1]["prompt"]
    assert "seam_gap_sec" in client.calls[1]["prompt"]
    assert result.localization_reports[0]["summary"]["draft_over_limit_count"] == 0


def test_spoken_cost_over_budget_retries_until_it_fits(monkeypatch) -> None:
    monkeypatch.setattr("eistara.core.translation.service.time.sleep", lambda _seconds: None)
    item = TranslationItem(id=1, source="hello world", duration_sec=10)
    client = ScriptedLlmClient(
        [
            {"translations": [{"id": 1, "text": "\u4e00" * 60}]},
            {"translations": [{"id": 1, "text": "\u4e8c" * 60}]},
            {"translations": [{"id": 1, "text": "\u4e09" * 35}]},
        ]
    )
    service = PublishTranslationService(
        client,
        TranslationSettings(
            use_summary=False,
            enforce_latin=False,
            localization_chars_per_sec=4.2,
            localization_max_audio_speed=1.10,
            localization_seam_gap_sec=0.12,
        ),
    )

    result = service.translate([item])

    assert result.translations[1] == "\u4e09" * 35
    assert len(client.calls) == 3
    assert client.calls[1]["use_cache"] is True
    assert client.calls[2]["use_cache"] is False
    assert result.localization_reports[0]["summary"]["draft_over_limit_count"] == 1
    assert result.localization_reports[0]["summary"]["final_over_limit_count"] == 0
    assert result.localization_reports[0]["summary"]["final_under_min_count"] == 0


def test_spoken_cost_under_floor_retries_until_it_fits(monkeypatch) -> None:
    monkeypatch.setattr("eistara.core.translation.service.time.sleep", lambda _seconds: None)
    item = TranslationItem(id=1, source="hello world", duration_sec=7.523)
    client = ScriptedLlmClient(
        [
            {"translations": [{"id": 1, "text": "\u4e00" * 33}]},
            {"translations": [{"id": 1, "text": "\u4e8c" * 24}]},
            {"translations": [{"id": 1, "text": "\u4e09" * 27}]},
        ]
    )
    service = PublishTranslationService(
        client,
        TranslationSettings(
            use_summary=False,
            enforce_latin=False,
            localization_chars_per_sec=4.2,
            localization_spoken_cost_per_sec=3.6,
            localization_max_audio_speed=1.10,
            localization_seam_gap_sec=0.12,
        ),
    )

    result = service.translate([item])

    assert result.translations[1] == "\u4e09" * 27
    assert len(client.calls) == 3
    assert client.calls[1]["use_cache"] is True
    assert client.calls[2]["use_cache"] is False
    report = result.localization_reports[0]
    assert report["items"][0]["min_spoken_cost"] == 27
    assert report["items"][0]["final_spoken_cost"] == 27
    assert report["items"][0]["final_under_min"] is False
    assert report["summary"]["final_under_min_count"] == 0


def test_semantic_review_repairs_localization_omission() -> None:
    source = (
        "Indeed, the work of American professor Caleb E. Finch has not only found a doubling of life expectancy "
        "since the year 1800, perhaps not surprising when we consider modern medicine, diet and living standards, "
        "but also another, more ancient increase."
    )
    draft = "美国教授芬奇的研究发现，自1800年以来人类预期寿命翻了一番，考虑到现代医学、饮食和生活水平，这并不意外，但他还发现了一次更古老的寿命增长。"
    flawed = "美国教授芬奇的研究发现，自1800年以来人类寿命翻了一番，这并不意外，但他还发现了更古老的一次增长。"
    repaired = "芬奇发现，1800年以来人类寿命翻番，考虑到医学、饮食和生活水平并不意外；但他还发现了更古老的一次增长。"
    item = TranslationItem(id=326, source=source, duration_sec=30)
    client = ScriptedLlmClient(
        [
            {"translations": [{"id": 326, "text": draft}]},
            {"translations": [{"id": 326, "text": flawed}]},
            {
                "issues": [
                    {
                        "id": 326,
                        "severity": "major",
                        "issue_type": "omission",
                        "missing_meaning": "modern medicine, diet and living standards",
                        "repair_instruction": "恢复现代医学、饮食和生活水平这组三项解释原因。",
                    }
                ]
            },
            {"translations": [{"id": 326, "text": repaired}]},
        ]
    )
    service = PublishTranslationService(
        client,
        TranslationSettings(use_summary=False, enforce_latin=False),
    )

    result = service.translate([item])

    assert result.translations[326] == repaired
    assert [call["log_title"] for call in client.calls] == [
        "translate_publish_fast",
        "translate_publish_localize",
        "translate_publish_localize_review",
        "translate_publish_localize_repair",
    ]
    assert "modern medicine, diet and living standards" in client.calls[2]["prompt"]
    semantic = result.localization_reports[0]["semantic_review"]
    assert semantic["candidate_count"] == 1
    assert semantic["issue_count"] == 1
    assert semantic["repaired_count"] == 1
    assert semantic["repaired_ids"] == [326]
    assert any("semantic review found 1 issue" in warning for warning in result.warnings)


def test_over_budget_first_pass_is_retranslated_in_second_pass() -> None:
    item = TranslationItem(id=1, source="hello world", duration_sec=10)
    draft = "\u4e00" * 60
    localized = "\u4e8c" * 35
    client = ScriptedLlmClient(
        [
            {"translations": [{"id": 1, "text": draft}]},
            {"translations": [{"id": 1, "text": localized}]},
        ]
    )
    service = PublishTranslationService(
        client,
        TranslationSettings(
            use_summary=False,
            enforce_latin=False,
            localization_chars_per_sec=4.2,
            localization_max_audio_speed=1.10,
            localization_seam_gap_sec=0.12,
        ),
    )

    result = service.translate([item])

    assert result.translations[1] == localized
    assert client.calls[1]["log_title"] == "translate_publish_localize"
    assert draft in client.calls[1]["prompt"]
    assert result.localization_reports[0]["summary"]["draft_over_limit_count"] == 1


def test_localization_windows_are_global_across_batches() -> None:
    items = [
        TranslationItem(id=1, source="first", start=0, end=7, duration_sec=7),
        TranslationItem(id=2, source="second", start=10, end=12, duration_sec=2),
    ]
    client = ScriptedLlmClient(
        [
            {"translations": [{"id": 1, "text": "\u4e00" * 20}]},
            {"translations": [{"id": 2, "text": "\u4e09" * 5}]},
            {"translations": [{"id": 1, "text": "\u4e8c" * 20}]},
            {"translations": [{"id": 2, "text": "\u56db" * 5}]},
        ]
    )
    service = PublishTranslationService(
        client,
        TranslationSettings(
            max_batch_lines=1,
            use_summary=False,
            enforce_latin=False,
            localization_chars_per_sec=4.2,
            localization_max_audio_speed=1.10,
            localization_seam_gap_sec=0.12,
        ),
    )

    service.translate(items)

    assert [call["log_title"] for call in client.calls] == [
        "translate_publish_fast",
        "translate_publish_fast",
        "translate_publish_localize",
        "translate_publish_localize",
    ]
    first_localization_prompt = client.calls[2]["prompt"]
    assert '"window_duration_sec": 10.0' in client.calls[0]["prompt"]
    assert '"max_chinese_chars"' not in client.calls[0]["prompt"]
    assert '"window_duration_sec": 10.0' in first_localization_prompt
    assert '"max_chinese_chars"' not in first_localization_prompt


def test_publish_prompt_uses_only_item_window_budget_for_dense_batch() -> None:
    client = ScriptedLlmClient([{"translations": [{"id": 1, "text": "短句"}]}])
    service = PublishTranslationService(
        client,
        TranslationSettings(use_summary=False, enforce_latin=False),
    )
    client.responses.append({"translations": [{"id": 1, "text": "短句"}]})
    item = TranslationItem(id=1, source=" ".join(["word"] * 80), duration_sec=20)

    service.translate([item])

    prompt = client.calls[0]["prompt"]
    assert "Dubbing pacing budget" not in prompt
    assert "this batch" not in prompt
    assert "target_total_chinese_chars" not in prompt
    assert "suggested_zh_chars" not in prompt
    assert '"dubbing_window"' in prompt
    assert '"max_spoken_cost"' in prompt
    assert "ideal_pressure_chinese_chars" not in prompt


def test_publish_prompt_omits_suggested_counts_for_low_pressure_batch() -> None:
    client = ScriptedLlmClient([{"translations": [{"id": 1, "text": "鑷劧琛ㄨ揪"}]}])
    service = PublishTranslationService(
        client,
        TranslationSettings(use_summary=False, enforce_latin=False),
    )
    client.responses.append({"translations": [{"id": 1, "text": "自然表达"}]})
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
        TranslationSettings(use_summary=False, enforce_latin=False),
    )
    client.responses.append({"translations": [{"id": 1, "text": "自然表达"}]})
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
    client.responses.append(client.responses[-1])
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
    first_recovery = client.responses[-2]["translations"][0]["text"]
    second_recovery = client.responses[-1]["translations"][0]["text"]
    client.responses.append(
        {"translations": [{"id": 1, "text": first_recovery}, {"id": 2, "text": second_recovery}]}
    )
    service = PublishTranslationService(client)

    result = service.translate([TranslationItem(id=1, source="one"), TranslationItem(id=2, source="two")])

    assert result.translations == {1: "第一句", 2: "第二句"}
    assert [call["log_title"] for call in client.calls[-3:]] == [
        "translate_publish_fast",
        "translate_publish_fast",
        "translate_publish_localize",
    ]
    assert len(client.calls) == 9


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
    service.register(
        TranslationStageRunner(
            ScriptedLlmClient(
                [
                    {"translations": [{"id": 1, "text": "翻译"}]},
                    {"translations": [{"id": 1, "text": "润色"}]},
                ]
            )
        )
    )

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

    runner.llm.responses.append(runner.llm.responses[-1])
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
    assert "pacing" not in report["summary"]
    assert report["summary"]["spoken_cost"]["actual_spoken_cost_per_sec"]["max"] is not None
    assert pd.read_excel(result.outputs["translation"]).to_dict(orient="records")[0]["Source"] == "Hello world"
    assert result.outputs["translation_count"] == 2
    assert result.outputs["translations"] == {1: "你好，世界", 2: "晚安"}
    assert result.outputs["tts_segments_count"] == 2
    assert result.outputs["tts_segments"][0]["source"] == "Hello world"
    assert result.outputs["tts_segments"][0]["speaker"] == "SPEAKER_01"
    assert result.outputs["tts_segments"][0]["metadata"]["speaker"] == "SPEAKER_01"
    tts_segments_json = Path(result.outputs["tts_segments_json"])
    assert tts_segments_json == tmp_path / "output" / "internal" / "tts_segments.json"
    assert json.loads(tts_segments_json.read_text(encoding="utf-8"))["segments"][0]["source"] == "Hello world"
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

    runner.settings = TranslationSettings(enforce_latin=False)
    runner.llm.responses.append(runner.llm.responses[-1])
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

    runner.llm.responses.append(runner.llm.responses[-1])
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
    assert [call["log_title"] for call in client.calls] == ["summary", "translate_publish_fast", "translate_publish_localize"]
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

    runner.llm.responses.append(runner.llm.responses[-1])
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
    assert [call["log_title"] for call in client.calls] == ["translate_publish_fast", "translate_publish_localize"]
    assert "Existing theme" in client.calls[0]["prompt"]
