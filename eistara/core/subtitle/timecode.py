from __future__ import annotations


def format_srt_timecode(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    whole_seconds = int(seconds % 60)
    milliseconds = int(round((seconds - int(seconds)) * 1000))
    if milliseconds >= 1000:
        whole_seconds += 1
        milliseconds -= 1000
    if whole_seconds >= 60:
        minutes += 1
        whole_seconds -= 60
    if minutes >= 60:
        hours += 1
        minutes -= 60
    return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d},{milliseconds:03d}"


def parse_srt_timecode(value: str) -> float:
    hms, milliseconds = str(value).strip().split(",")
    hours, minutes, seconds = hms.split(":")
    return int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(milliseconds) / 1000


def parse_time_seconds(value: object) -> float:
    text = str(value or "").strip()
    if not text:
        return 0.0
    try:
        return float(text)
    except ValueError:
        pass
    text = text.replace(",", ".")
    parts = text.split(":")
    try:
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        if len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
    except ValueError:
        return 0.0
    return 0.0


def format_srt_timestamp(start_sec: float, end_sec: float) -> str:
    return f"{format_srt_timecode(start_sec)} --> {format_srt_timecode(end_sec)}"


def parse_srt_timestamp(timestamp: str) -> tuple[float, float]:
    start_text, end_text = [part.strip() for part in str(timestamp).split("-->")]
    return parse_srt_timecode(start_text), parse_srt_timecode(end_text)
