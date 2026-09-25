#!/usr/bin/env python3

import json
import csv
import sys
from pathlib import Path

INPUT_PATH = r"qwen2_5vl_instruct_results.json"
OUTPUT_PATH = "affordance_results_2_5vl_filtered.csv"

FIELDNAMES = [
    "task",
    "image",
    "object",
    "affordance",
    "region",
    "action",
    "has_affordance",
    "status",
]

EXCLUDED_TASKS = {"exhaust_fire", "wine"}

# Correct misspelled task names in the CSV
TASK_NAME_CORRECTIONS = {
    "extnigh_fire": "extinguish_fire"
}


def load_data(path):
    with open(path, encoding="utf-8-sig") as f:
        return json.load(f)


def normalize_object_entry(obj):
    if isinstance(obj, str):
        return obj, None

    if isinstance(obj, dict):
        name = obj.get("object", "")
        embedded_aff = (
            obj.get("affordance")
            if isinstance(obj.get("affordance"), dict)
            else None
        )
        return name, embedded_aff

    return str(obj), None


def rows_for_image(task, image, info):
    raw_objects = info.get("objects") or []
    affordances = list(info.get("affordances") or [])
    status = info.get("status", "")

    objects = []

    for raw_obj in raw_objects:
        name, embedded_aff = normalize_object_entry(raw_obj)
        objects.append(name)

        if embedded_aff:
            affordances.append(embedded_aff)

    aff_by_object = {}

    for aff in affordances:
        if not isinstance(aff, dict):
            continue

        obj_name = aff.get("object", "")
        aff_by_object.setdefault(obj_name, []).append(aff)

    emitted = False

    for obj in objects:
        matches = aff_by_object.get(obj)

        if matches:
            for aff in matches:
                yield {
                    "task": task,
                    "image": image,
                    "object": obj,
                    "affordance": aff.get("affordance", ""),
                    "region": aff.get("region", ""),
                    "action": aff.get("action", ""),
                    "has_affordance": "Yes",
                    "status": status,
                }
                emitted = True

        else:
            yield {
                "task": task,
                "image": image,
                "object": obj,
                "affordance": "",
                "region": "",
                "action": "",
                "has_affordance": "No",
                "status": status,
            }
            emitted = True

    listed_objects = set(objects)

    for obj_name, matches in aff_by_object.items():
        if obj_name in listed_objects:
            continue

        for aff in matches:
            yield {
                "task": task,
                "image": image,
                "object": obj_name,
                "affordance": aff.get("affordance", ""),
                "region": aff.get("region", ""),
                "action": aff.get("action", ""),
                "has_affordance": "Yes",
                "status": status,
            }
            emitted = True

    if not emitted:
        yield {
            "task": task,
            "image": image,
            "object": "",
            "affordance": "",
            "region": "",
            "action": "",
            "has_affordance": "No",
            "status": status,
        }


def main():
    in_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(INPUT_PATH)
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(OUTPUT_PATH)

    data = load_data(in_path)

    row_count = 0

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()

        for task, images in data.items():

            # Skip these tasks completely
            if task in EXCLUDED_TASKS:
                continue

            # Correct misspelled task name
            corrected_task = TASK_NAME_CORRECTIONS.get(task, task)

            for image, info in images.items():
                for row in rows_for_image(corrected_task, image, info):
                    writer.writerow(row)
                    row_count += 1

    print(f"Wrote {row_count} rows to {out_path}")
    print(f"Excluded tasks: {', '.join(sorted(EXCLUDED_TASKS))}")
    print("Task name corrections applied:")
    
    for wrong, correct in TASK_NAME_CORRECTIONS.items():
        print(f"  {wrong} -> {correct}")


if __name__ == "__main__":
    main()