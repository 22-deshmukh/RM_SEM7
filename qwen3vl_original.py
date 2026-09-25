import json
import os
import re
import subprocess
import sys
import time


# ============================================================
# PATH CONFIGURATION
# ============================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

CLI_PATH = os.path.join(
    SCRIPT_DIR,
    "llama.cpp",
    "build",
    "bin",
    "Release",
    "llama-mtmd-cli.exe"
)

MODEL_PATH = os.path.join(
    SCRIPT_DIR,
    "Qwen3VL-2B-Thinking-Q4_K_M.gguf"
)

MMPROJ_PATH = os.path.join(
    SCRIPT_DIR,
    "mmproj-Qwen3VL-2B-Thinking-F16.gguf"
)

DATASET_DIR = os.path.join(
    SCRIPT_DIR,
    "data"
)

OUTPUT_JSON = os.path.join(
    SCRIPT_DIR,
    "full_dataset_evaluation_results.json"
)


# ============================================================
# RUN QWEN
# ============================================================

def run_qwen_vl_cli(image_path, prompt, max_retries=2):
    """
    Run Qwen3-VL through llama-mtmd-cli.

    Returns:
        combined stdout + stderr
        or None if execution completely fails.
    """

    cmd = [
        CLI_PATH,

        "-m",
        MODEL_PATH,

        "--mmproj",
        MMPROJ_PATH,

        "--image",
        image_path,

        "-p",
        prompt,

        "--temp",
        "0.2",

        "--ctx-size",
        "8192"
    ]

    for attempt in range(1, max_retries + 1):

        print(
            f"       Running model "
            f"(attempt {attempt}/{max_retries})...",
            flush=True
        )

        try:

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300
            )

            stdout = result.stdout or ""
            stderr = result.stderr or ""

            # ------------------------------------------------
            # llama.cpp logs may be in stderr while answer
            # may be in stdout.
            # ------------------------------------------------

            combined_output = stdout + "\n" + stderr

            if result.returncode != 0:

                print(
                    f"       ⚠️ llama.cpp returned "
                    f"code {result.returncode}",
                    flush=True
                )

                print(
                    f"       STDERR: {stderr[:1000]}",
                    flush=True
                )

                if attempt < max_retries:
                    time.sleep(2)
                    continue

                return None

            # Successful process
            return combined_output

        except subprocess.TimeoutExpired:

            print(
                f"       ⚠️ Timeout after 300 seconds",
                flush=True
            )

            if attempt < max_retries:
                print(
                    "       Retrying...",
                    flush=True
                )
                time.sleep(2)
                continue

            return None

        except Exception as e:

            print(
                f"       ⚠️ Subprocess error: {e}",
                flush=True
            )

            if attempt < max_retries:
                time.sleep(2)
                continue

            return None

    return None


# ============================================================
# EXTRACT JSON
# ============================================================

def parse_and_normalize_json(raw_text, task_name):

    # --------------------------------------------------------
    # No output
    # --------------------------------------------------------

    if not raw_text:

        return {
            "objects": [],
            "affordances": [],
            "status": "empty_output"
        }

    # --------------------------------------------------------
    # Remove Qwen thinking section
    # --------------------------------------------------------

    cleaned_text = re.sub(
        r"<think>.*?</think>",
        "",
        raw_text,
        flags=re.DOTALL | re.IGNORECASE
    )

    # Handle unfinished <think>
    cleaned_text = re.sub(
        r"<think>.*$",
        "",
        cleaned_text,
        flags=re.DOTALL | re.IGNORECASE
    )

    cleaned_text = cleaned_text.strip()

    # --------------------------------------------------------
    # Remove markdown JSON blocks
    # --------------------------------------------------------

    cleaned_text = re.sub(
        r"```json",
        "",
        cleaned_text,
        flags=re.IGNORECASE
    )

    cleaned_text = cleaned_text.replace("```", "")

    cleaned_text = cleaned_text.strip()

    # --------------------------------------------------------
    # Find JSON object
    # --------------------------------------------------------

    start_idx = cleaned_text.find("{")
    end_idx = cleaned_text.rfind("}") + 1

    if start_idx == -1 or end_idx <= start_idx:

        return {
            "objects": [],
            "affordances": [],
            "status": "no_json_found",
            "raw_output": raw_text[-3000:]
        }

    json_text = cleaned_text[start_idx:end_idx]

    # --------------------------------------------------------
    # Parse JSON
    # --------------------------------------------------------

    try:

        data = json.loads(json_text)

        if isinstance(data, dict):

            data.setdefault("objects", [])
            data.setdefault("affordances", [])

            data["status"] = "success"

            return data

        return {
            "objects": [],
            "affordances": [],
            "status": "json_not_dict",
            "raw_output": raw_text[-3000:]
        }

    except json.JSONDecodeError as e:

        return {
            "objects": [],
            "affordances": [],
            "status": "json_parse_error",
            "error": str(e),
            "raw_output": raw_text[-3000:]
        }


# ============================================================
# SAVE CHECKPOINT
# ============================================================

def save_results(all_results):

    try:

        temp_file = OUTPUT_JSON + ".tmp"

        with open(
            temp_file,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                all_results,
                f,
                indent=4,
                ensure_ascii=False
            )

        # Replace old file only after successful write
        os.replace(
            temp_file,
            OUTPUT_JSON
        )

    except Exception as e:

        print(
            f"       ⚠️ Failed to save checkpoint: {e}",
            flush=True
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("\n==============================================")
    print("       QWEN3-VL DATASET EVALUATION")
    print("==============================================\n")

    # --------------------------------------------------------
    # Check required files
    # --------------------------------------------------------

    if not os.path.exists(CLI_PATH):

        print(
            f"[FATAL] CLI executable missing:\n{CLI_PATH}"
        )

        sys.exit(1)

    if not os.path.exists(MODEL_PATH):

        print(
            f"[FATAL] Model missing:\n{MODEL_PATH}"
        )

        sys.exit(1)

    if not os.path.exists(MMPROJ_PATH):

        print(
            f"[FATAL] MM projector missing:\n{MMPROJ_PATH}"
        )

        sys.exit(1)

    if not os.path.exists(DATASET_DIR):

        print(
            f"[FATAL] Dataset directory missing:\n{DATASET_DIR}"
        )

        sys.exit(1)

    print("CLI      :", CLI_PATH)
    print("MODEL    :", MODEL_PATH)
    print("MMPROJ   :", MMPROJ_PATH)
    print("DATASET  :", DATASET_DIR)
    print("OUTPUT   :", OUTPUT_JSON)
    print()

    # --------------------------------------------------------
    # Load previous checkpoint
    # --------------------------------------------------------

    all_results = {}

    if os.path.exists(OUTPUT_JSON):

        try:

            with open(
                OUTPUT_JSON,
                "r",
                encoding="utf-8"
            ) as f:

                all_results = json.load(f)

            total_saved = sum(
                len(images)
                for images in all_results.values()
                if isinstance(images, dict)
            )

            print(
                f"Loaded checkpoint: "
                f"{total_saved} images already processed.\n"
            )

        except Exception as e:

            print(
                f"⚠️ Could not load previous checkpoint: {e}"
            )

            print(
                "Starting with empty results.\n"
            )

            all_results = {}

    # --------------------------------------------------------
    # Get task folders
    # --------------------------------------------------------

    task_folders = sorted(
        [
            f
            for f in os.listdir(DATASET_DIR)
            if os.path.isdir(
                os.path.join(DATASET_DIR, f)
            )
        ]
    )

    total_folders = len(task_folders)

    print(
        f"Found {total_folders} task folders.\n"
    )

    # ========================================================
    # DATASET LOOP
    # ========================================================

    for folder_idx, task_folder in enumerate(
        task_folders,
        1
    ):

        task_path = os.path.join(
            DATASET_DIR,
            task_folder
        )

        # ----------------------------------------------------
        # Task name
        # ----------------------------------------------------

        task_name = task_folder.replace(
            "_",
            " "
        ).strip()

        if (
            "extnigh" in task_name.lower()
            or "exhaust" in task_name.lower()
        ):
            task_name = "extinguish fire"

        print("\n==================================================")
        print(
            f"[{folder_idx}/{total_folders}] "
            f"Task: '{task_name}'"
        )
        print("==================================================")

        # ----------------------------------------------------
        # Initialize folder
        # ----------------------------------------------------

        if task_folder not in all_results:

            all_results[task_folder] = {}

        # ----------------------------------------------------
        # Find images
        # ----------------------------------------------------

        images = sorted(
            [
                i
                for i in os.listdir(task_path)
                if i.lower().endswith(
                    (
                        ".jpg",
                        ".jpeg",
                        ".png",
                        ".webp"
                    )
                )
            ]
        )

        print(
            f"Images in folder: {len(images)}\n"
        )

        # ====================================================
        # IMAGE LOOP
        # ====================================================

        for img_idx, img_name in enumerate(
            images,
            1
        ):

            # ------------------------------------------------
            # Skip completed images
            # ------------------------------------------------

            if img_name in all_results[task_folder]:

                previous_status = all_results[
                    task_folder
                ][img_name].get(
                    "status",
                    "unknown"
                )

                print(
                    f"  [{img_idx}/{len(images)}] "
                    f"Skipping {img_name} "
                    f"({previous_status})",
                    flush=True
                )

                continue

            img_path = os.path.join(
                task_path,
                img_name
            )

            print(
                f"  [{img_idx}/{len(images)}] "
                f"Evaluating {img_name}...",
                flush=True
            )

            # ------------------------------------------------
            # Check image exists
            # ------------------------------------------------

            if not os.path.isfile(img_path):

                print(
                    "       ⚠️ Image does not exist",
                    flush=True
                )

                all_results[
                    task_folder
                ][img_name] = {

                    "objects": [],
                    "affordances": [],
                    "status": "image_missing"
                }

                save_results(all_results)

                continue

            # ------------------------------------------------
            # Prompt
            # ------------------------------------------------

            prompt = f"""
You are an expert in visual affordance detection.

Analyze the ENTIRE image.

Task: {task_name.upper()}

Identify the visible objects relevant to this task.

Return ONLY valid JSON.
Do not write explanations.
Do not use markdown.
Do not include <think>.

Use exactly this structure:

{{
    "objects": [
        "object 1",
        "object 2"
    ],
    "affordances": [
        {{
            "object": "target object",
            "affordance": "{task_name}",
            "region": "specific part of object used",
            "action": "action to perform"
        }}
    ]
}}

If there is no suitable object, return:

{{
    "objects": [],
    "affordances": []
}}
"""

            # ------------------------------------------------
            # Run model
            # ------------------------------------------------

            raw_output = None

            try:

                raw_output = run_qwen_vl_cli(
                    img_path,
                    prompt,
                    max_retries=2
                )

            except Exception as e:

                print(
                    f"       ⚠️ Unexpected model error: {e}",
                    flush=True
                )

            # ------------------------------------------------
            # Parse result
            # ------------------------------------------------

            try:

                parsed_json = parse_and_normalize_json(
                    raw_output,
                    task_name
                )

            except Exception as e:

                print(
                    f"       ⚠️ Parser error: {e}",
                    flush=True
                )

                parsed_json = {

                    "objects": [],
                    "affordances": [],

                    "status": "parser_exception",

                    "error": str(e)
                }

            # ------------------------------------------------
            # Save result
            # ------------------------------------------------

            all_results[
                task_folder
            ][img_name] = parsed_json

            # ------------------------------------------------
            # SAVE IMMEDIATELY
            # ------------------------------------------------

            save_results(all_results)

            # ------------------------------------------------
            # Print status
            # ------------------------------------------------

            status = parsed_json.get(
                "status",
                "unknown"
            )

            if status == "success":

                print(
                    "       ✅ Prediction saved",
                    flush=True
                )

            else:

                print(
                    f"       ⚠️ Saved with status: {status}",
                    flush=True
                )

            # ------------------------------------------------
            # Small delay
            # ------------------------------------------------

            time.sleep(0.2)

    # ========================================================
    # COMPLETE
    # ========================================================

    total_images = sum(
        len(images)
        for images in all_results.values()
        if isinstance(images, dict)
    )

    successful = 0
    failed = 0

    for folder_results in all_results.values():

        if not isinstance(folder_results, dict):
            continue

        for result in folder_results.values():

            if isinstance(result, dict):

                if result.get("status") == "success":
                    successful += 1
                else:
                    failed += 1

    print("\n==============================================")
    print("       DATASET TRAVERSAL COMPLETE")
    print("==============================================")

    print(f"Total images : {total_images}")
    print(f"Successful   : {successful}")
    print(f"Failed       : {failed}")

    print(
        f"\nOutput saved to:\n{OUTPUT_JSON}"
    )