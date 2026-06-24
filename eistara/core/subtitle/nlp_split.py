from __future__ import annotations

import itertools
import os
import string
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_SPACY_MODEL_MAP = {
    "en": "en_core_web_md",
    "ru": "ru_core_news_md",
    "fr": "fr_core_news_md",
    "ja": "ja_core_news_md",
    "es": "es_core_news_md",
    "de": "de_core_news_md",
    "it": "it_core_news_md",
    "zh": "zh_core_web_md",
}
DEFAULT_LANGUAGE_SPLIT_WITH_SPACE = ("en", "es", "fr", "de", "it", "ru")
DEFAULT_LANGUAGE_SPLIT_WITHOUT_SPACE = ("zh", "ja")


@dataclass(frozen=True, slots=True)
class NlpSplitResult:
    split_by_nlp: Path
    language: str
    model: str
    sentence_count: int
    warnings: tuple[str, ...] = ()


def generate_split_by_nlp(
    cleaned_chunks: str | Path,
    output_dir: str | Path,
    *,
    language: str | None = None,
    spacy_model_map: dict[str, str] | None = None,
    language_split_with_space: tuple[str, ...] | list[str] = DEFAULT_LANGUAGE_SPLIT_WITH_SPACE,
    language_split_without_space: tuple[str, ...] | list[str] = DEFAULT_LANGUAGE_SPLIT_WITHOUT_SPACE,
) -> NlpSplitResult:
    output_dir = Path(output_dir)
    log_dir = output_dir / "log"
    log_dir.mkdir(parents=True, exist_ok=True)
    split_by_mark_file = log_dir / "split_by_mark.txt"
    split_by_comma_file = log_dir / "split_by_comma.txt"
    split_by_connector_file = log_dir / "split_by_connector.txt"
    split_by_nlp = log_dir / "split_by_nlp.txt"

    selected_language = _normalize_language(language)
    model_map = dict(DEFAULT_SPACY_MODEL_MAP)
    model_map.update(spacy_model_map or {})
    model = model_map.get(selected_language, "en_core_web_md")
    nlp, warnings = _load_nlp(selected_language, model)
    joiner = get_joiner(selected_language, language_split_with_space, language_split_without_space)

    sentences = split_by_mark(read_cleaned_chunk_texts(cleaned_chunks), nlp, joiner)
    _write_lines(split_by_mark_file, sentences)
    sentences = split_by_comma_lines(sentences, nlp)
    _write_lines(split_by_comma_file, sentences)
    _remove_if_exists(split_by_mark_file)
    sentences = split_by_connector_lines(sentences, nlp)
    _write_lines(split_by_connector_file, sentences, trim_final_newline=True)
    _remove_if_exists(split_by_comma_file)
    sentences = split_long_by_root_lines(sentences, nlp, selected_language, joiner)
    sentences = _drop_or_merge_punctuation_only(sentences)
    _write_lines(split_by_nlp, sentences)
    _remove_if_exists(split_by_connector_file)

    return NlpSplitResult(
        split_by_nlp=split_by_nlp,
        language=selected_language,
        model=model,
        sentence_count=len(sentences),
        warnings=tuple(warnings),
    )


def read_cleaned_chunk_texts(cleaned_chunks: str | Path) -> list[str]:
    chunks = pd.read_excel(cleaned_chunks)
    if "text" not in chunks:
        return []
    return [str(text).strip('"').strip() for text in chunks["text"].to_list() if str(text).strip('"').strip()]


def get_joiner(
    language: str,
    language_split_with_space: tuple[str, ...] | list[str] = DEFAULT_LANGUAGE_SPLIT_WITH_SPACE,
    language_split_without_space: tuple[str, ...] | list[str] = DEFAULT_LANGUAGE_SPLIT_WITHOUT_SPACE,
) -> str:
    if language in language_split_with_space:
        return " "
    if language in language_split_without_space:
        return ""
    return " "


def split_by_mark(texts: list[str], nlp: Any, joiner: str) -> list[str]:
    input_text = joiner.join(texts)
    if not input_text.strip():
        return []
    doc = nlp(input_text)
    sentences_by_mark: list[str] = []
    current_sentence: list[str] = []

    for sent in doc.sents:
        text = sent.text.strip()
        if not text:
            continue
        if current_sentence and (
            text.startswith("-")
            or text.startswith("...")
            or current_sentence[-1].endswith("-")
            or current_sentence[-1].endswith("...")
        ):
            current_sentence.append(text)
        else:
            if current_sentence:
                sentences_by_mark.append(" ".join(current_sentence))
                current_sentence = []
            current_sentence.append(text)

    if current_sentence:
        sentences_by_mark.append(" ".join(current_sentence))
    return _merge_standalone_sentence_punctuation(sentences_by_mark)


def split_by_comma_lines(sentences: list[str], nlp: Any) -> list[str]:
    result: list[str] = []
    for sentence in sentences:
        result.extend(split_by_comma(sentence.strip(), nlp))
    return result


def split_by_comma(text: str, nlp: Any) -> list[str]:
    doc = nlp(text)
    sentences: list[str] = []
    start = 0
    for token in doc:
        if token.text not in {",", "，"}:
            continue
        if analyze_comma(start, doc, token):
            sentences.append(doc[start : token.i].text.strip())
            start = token.i + 1
    sentences.append(doc[start:].text.strip())
    return [sentence for sentence in sentences if sentence]


def analyze_comma(start: int, doc: Any, token: Any) -> bool:
    left_phrase = doc[max(start, token.i - 9) : token.i]
    right_phrase = doc[token.i + 1 : min(len(doc), token.i + 10)]
    suitable_for_splitting = is_valid_phrase(right_phrase)

    left_words = [item for item in left_phrase if not item.is_punct]
    right_words = list(itertools.takewhile(lambda item: not item.is_punct, right_phrase))
    if len(left_words) <= 3 or len(right_words) <= 3:
        suitable_for_splitting = False
    return suitable_for_splitting


def is_valid_phrase(phrase: Any) -> bool:
    has_subject = any(token.dep_ in {"nsubj", "nsubjpass"} or token.pos_ == "PRON" for token in phrase)
    has_verb = any(token.pos_ in {"VERB", "AUX"} for token in phrase)
    return has_subject and has_verb


def split_by_connector_lines(sentences: list[str], nlp: Any) -> list[str]:
    result: list[str] = []
    for sentence in sentences:
        result.extend(split_by_connectors(sentence.strip(), nlp=nlp))
    return result


def split_by_connectors(text: str, *, context_words: int = 5, nlp: Any) -> list[str]:
    doc = nlp(text)
    sentences = [doc.text]

    while True:
        split_occurred = False
        new_sentences: list[str] = []
        for sentence in sentences:
            doc = nlp(sentence)
            start = 0
            for i, token in enumerate(doc):
                split_before = analyze_connectors(doc, token)
                if i + 1 < len(doc) and doc[i + 1].text in {"'s", "'re", "'ve", "'ll", "'d"}:
                    continue

                left_words = doc[max(0, token.i - context_words) : token.i]
                right_words = doc[token.i + 1 : min(len(doc), token.i + context_words + 1)]
                left_text = [word.text for word in left_words if not word.is_punct]
                right_text = [word.text for word in right_words if not word.is_punct]

                if len(left_text) >= context_words and len(right_text) >= context_words and split_before:
                    new_sentences.append(doc[start : token.i].text.strip())
                    start = token.i
                    split_occurred = True
                    break
            if start < len(doc):
                new_sentences.append(doc[start:].text.strip())

        if not split_occurred:
            break
        sentences = new_sentences

    return [sentence for sentence in sentences if sentence]


def analyze_connectors(doc: Any, token: Any) -> bool:
    lang = getattr(doc, "lang_", "en")
    connectors_by_language = {
        "en": {"that", "which", "where", "when", "because", "but", "and", "or"},
        "zh": {"因为", "所以", "但是", "而且", "虽然", "如果", "即使", "尽管"},
        "ja": {"けれども", "しかし", "だから", "それで", "ので", "のに", "ため"},
        "fr": {"que", "qui", "où", "quand", "parce que", "mais", "et", "ou"},
        "ru": {"что", "который", "где", "когда", "потому что", "но", "и", "или"},
        "es": {"que", "cual", "donde", "cuando", "porque", "pero", "y", "o"},
        "de": {"dass", "welche", "wo", "wann", "weil", "aber", "und", "oder"},
        "it": {"che", "quale", "dove", "quando", "perché", "ma", "e", "o"},
    }
    if token.text.lower() not in connectors_by_language.get(lang, set()):
        return False

    mark_dep = "mark"
    det_pron_deps = {"case"} if lang == "ja" else {"det", "pron"}
    noun_pos = {"NOUN", "PROPN"}
    if lang == "en" and token.text.lower() == "that":
        return token.dep_ == mark_dep and token.head.pos_ == "VERB"
    if token.dep_ in det_pron_deps and token.head.pos_ in noun_pos:
        return False
    return True


def split_long_by_root_lines(sentences: list[str], nlp: Any, language: str, joiner: str) -> list[str]:
    result: list[str] = []
    for sentence in sentences:
        doc = nlp(sentence.strip())
        if len(doc) > 60:
            split_sentences = split_long_sentence(doc, joiner)
            if any(len(nlp(item)) > 60 for item in split_sentences):
                split_sentences = [
                    subsentence
                    for item in split_sentences
                    for subsentence in split_extremely_long_sentence(nlp(item), language, joiner)
                ]
            result.extend(split_sentences)
        else:
            result.append(sentence.strip())
    return result


def split_long_sentence(doc: Any, joiner: str) -> list[str]:
    tokens = [token.text for token in doc]
    n = len(tokens)
    dp = [float("inf")] * (n + 1)
    dp[0] = 0
    prev = [0] * (n + 1)

    for i in range(1, n + 1):
        for j in range(max(0, i - 100), i):
            if i - j >= 30:
                token = doc[i - 1]
                if j == 0 or token.is_sent_end or token.pos_ in {"VERB", "AUX"} or token.dep_ == "ROOT":
                    if dp[j] + 1 < dp[i]:
                        dp[i] = dp[j] + 1
                        prev[i] = j

    sentences: list[str] = []
    i = n
    while i > 0:
        j = prev[i]
        if j == i:
            j = max(0, i - 60)
        sentences.append(joiner.join(tokens[j:i]).strip())
        i = j
    return list(reversed([sentence for sentence in sentences if sentence]))


def split_extremely_long_sentence(doc: Any, language: str, joiner: str) -> list[str]:
    tokens = [token.text for token in doc]
    n = len(tokens)
    num_parts = (n + 59) // 60
    part_length = max(1, n // num_parts)
    sentences: list[str] = []
    for index in range(num_parts):
        start = index * part_length
        end = start + part_length if index < num_parts - 1 else n
        sentences.append(joiner.join(tokens[start:end]))
    return sentences


def _normalize_language(language: str | None) -> str:
    text = str(language or "").strip().lower()
    if not text or text == "auto":
        return "en"
    return text.split("-")[0].split("_")[0]


@lru_cache(maxsize=8)
def _load_nlp(language: str, model: str):
    import spacy

    warnings: list[str] = []
    try:
        nlp = spacy.load(model)
        return nlp, tuple(warnings)
    except Exception as exc:
        fallback_language = language if language in {"en", "zh", "ja", "fr", "ru", "es", "de", "it"} else "en"
        warnings.append(f"Failed to load spaCy model {model!r}; using blank {fallback_language!r} sentencizer: {exc}")
        nlp = spacy.blank(fallback_language)
        if "sentencizer" not in nlp.pipe_names:
            nlp.add_pipe("sentencizer")
        return nlp, tuple(warnings)


def _write_lines(path: Path, lines: list[str], *, trim_final_newline: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines)
    if text and not trim_final_newline:
        text += "\n"
    path.write_text(text, encoding="utf-8")


def _remove_if_exists(path: Path) -> None:
    try:
        os.remove(path)
    except FileNotFoundError:
        return


def _merge_standalone_sentence_punctuation(sentences: list[str]) -> list[str]:
    standalone = {",", ".", "，", "。", "；", "！", "？", "、"}
    result: list[str] = []
    for sentence in sentences:
        if sentence.strip() in standalone and result:
            result[-1] += sentence
        else:
            result.append(sentence)
    return result


def _drop_or_merge_punctuation_only(sentences: list[str]) -> list[str]:
    punctuation = set(string.punctuation + "'" + '"' + "，。；！？、")
    result: list[str] = []
    for sentence in sentences:
        stripped = sentence.strip()
        if not stripped:
            continue
        if all(char in punctuation for char in stripped):
            if result:
                result[-1] += sentence
            continue
        result.append(sentence)
    return result
