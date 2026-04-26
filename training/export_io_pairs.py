import argparse
import json
from pathlib import Path
from typing import Any


def _parse_json_text(text: str) -> Any:
    text = text.strip()
    if "```" in text:
        text = "\n".join(
            line for line in text.splitlines()
            if not line.strip().startswith("```")
        ).strip()

    start = text.find("{")
    end = text.rfind("}") + 1
    if start < 0 or end <= start:
        return None

    try:
        return json.loads(text[start:end])
    except json.JSONDecodeError:
        return None


def _normalize_task(task: str) -> str:
    return task.replace("monthly_", "weekly_", 1)


def export_io_pairs(input_path: Path, output_path: Path) -> None:
    records = []
    with input_path.open() as f:
        for line in f:
            if line.strip():
                raw = json.loads(line)
                tag = raw.get("tag", "")
                split_tag = tag.split("/", 1)
                phase = split_tag[1] if len(split_tag) == 2 else None
                response_json = _parse_json_text(raw.get("response", ""))
                prompt = raw.get("prompt", "")

                records.append({
                    "tag": tag,
                    "phase": phase,
                    "episode": raw.get("ep"),
                    "day": raw.get("day"),
                    "task": _normalize_task(raw.get("task", "")),
                    "raw_task": raw.get("task"),
                    "seed": raw.get("seed"),
                    "input": prompt,
                    "output": raw.get("response", ""),
                    "output_json": response_json,
                    "has_tool_error": "ERROR" in prompt,
                })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(records, f, indent=2)

    print(f"wrote {len(records)} records to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="Path to io_log.jsonl")
    parser.add_argument("output", type=Path, help="Path to output JSON")
    args = parser.parse_args()
    export_io_pairs(args.input, args.output)


if __name__ == "__main__":
    main()
