
import json
import os
import re
import subprocess
import sys


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

# ------------------------------------------------------------
# FINAL CLEAN RESULTS
# ------------------------------------------------------------

OUTPUT_JSON = os.path.join(
    SCRIPT_DIR,
    "full_dataset_evaluation_results.json"
)

# ------------------------------------------------------------
# QWEN THINKING / DEBUG LOG
# ------------------------------------------------------------

THINKING_LOG_JSON = os.path.join(
    SCRIPT_DIR,
    "qwen3vl_thinking_log.json"
)


# ============================================================
# RUN QWEN3-VL
# ============================================================

def run_qwen_vl_cli(image_path, prompt):

    """
    Run Qwen3-VL through llama-mtmd-cli.

    Maximum time per image = 180 seconds.

    Returns:
        raw_output
        execution_status
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
        "8192",

        "-n",
        "1024"
    ]

    try:

        print(
            "       Running Qwen3-VL...",
            flush=True
        )

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=180
        )

        stdout = result.stdout or ""
        stderr = result.stderr or ""

        # ----------------------------------------------------
        # llama.cpp failed
        # ----------------------------------------------------

        if result.returncode != 0:

            print(
                f"       ⚠️ llama.cpp failed "
                f"(return code {result.returncode})",
                flush=True
            )

            print(
                f"       STDERR:\n{stderr[-2000:]}",
                flush=True
            )

            return stdout, "execution_error"

        # ----------------------------------------------------
        # Normally model response is in stdout
        # ----------------------------------------------------

        return stdout, "completed"

    except subprocess.TimeoutExpired as e:

        print(
            "       ⏱️ Timeout after 180 seconds.",
            flush=True
        )

        print(
            "       Moving to next image.",
            flush=True
        )

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # Python may have captured partial stdout/stderr
        # before the timeout.
        # ----------------------------------------------------

        partial_stdout = e.stdout or ""
        partial_stderr = e.stderr or ""

        if isinstance(partial_stdout, bytes):
            partial_stdout = partial_stdout.decode(
                "utf-8",
                errors="replace"
            )

        if isinstance(partial_stderr, bytes):
            partial_stderr = partial_stderr.decode(
                "utf-8",
                errors="replace"
            )

        partial_output = (
            partial_stdout
            + "\n"
            + partial_stderr
        )

        return partial_output, "timeout"

    except Exception as e:

        print(
            f"       ⚠️ Unexpected execution error: {e}",
            flush=True
        )

        return None, "execution_error"


# ============================================================
# EXTRACT THINKING
# ============================================================

def extract_thinking(raw_text):

    """
    Extract everything generated inside <think>...</think>.

    If the model was interrupted before </think>, we also
    capture the unfinished thinking block.
    """

    if not raw_text:
        return ""

    # --------------------------------------------------------
    # Normal completed thinking
    # --------------------------------------------------------

    matches = re.findall(
        r"<think>(.*?)</think>",
        raw_text,
        flags=re.DOTALL | re.IGNORECASE
    )

    if matches:

        return "\n\n".join(
            matches
        ).strip()

    # --------------------------------------------------------
    # Incomplete thinking block
    # --------------------------------------------------------

    match = re.search(
        r"<think>(.*)$",
        raw_text,
        flags=re.DOTALL | re.IGNORECASE
    )

    if match:

        return match.group(1).strip()

    return ""


# ============================================================
# PARSE MODEL OUTPUT
# ============================================================

def parse_and_normalize_json(raw_text):

    if not raw_text:

        return {
            "objects": [],
            "affordances": [],
            "status": "empty_output"
        }

    # --------------------------------------------------------
    # Remove thinking
    # --------------------------------------------------------

    cleaned_text = re.sub(
        r"<think>.*?</think>",
        "",
        raw_text,
        flags=re.DOTALL | re.IGNORECASE
    )

    # --------------------------------------------------------
    # Remove unfinished thinking
    # --------------------------------------------------------

    cleaned_text = re.sub(
        r"<think>.*$",
        "",
        cleaned_text,
        flags=re.DOTALL | re.IGNORECASE
    )

    cleaned_text = cleaned_text.strip()

    # --------------------------------------------------------
    # Remove markdown code fences
    # --------------------------------------------------------

    cleaned_text = re.sub(
        r"```json",
        "",
        cleaned_text,
        flags=re.IGNORECASE
    )

    cleaned_text = cleaned_text.replace(
        "```",
        ""
    )

    cleaned_text = cleaned_text.strip()

    # --------------------------------------------------------
    # Find JSON
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

    json_text = cleaned_text[
        start_idx:end_idx
    ]

    # --------------------------------------------------------
    # Parse JSON
    # --------------------------------------------------------

    try:

        data = json.loads(
            json_text
        )

        if isinstance(data, dict):

            data.setdefault(
                "objects",
                []
            )

            data.setdefault(
                "affordances",
                []
            )

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
# SAVE JSON SAFELY
# ============================================================

def save_json(data, output_path):

    try:

        temp_file = output_path + ".tmp"

        with open(
            temp_file,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                data,
                f,
                indent=4,
                ensure_ascii=False
            )

        os.replace(
            temp_file,
            output_path
        )

    except Exception as e:

        print(
            f"       ⚠️ Failed to save "
            f"{output_path}: {e}",
            flush=True
        )


# ============================================================
# LOAD JSON IF IT EXISTS
# ============================================================

def load_json(output_path):

    if not os.path.exists(
        output_path
    ):
        return {}

    try:

        with open(
            output_path,
            "r",
            encoding="utf-8"
        ) as f:

            return json.load(f)

    except Exception as e:

        print(
            f"⚠️ Could not load {output_path}: {e}"
        )

        print(
            "Starting with empty file."
        )

        return {}


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print()
    print("==============================================")
    print("       QWEN3-VL DATASET EVALUATION")
    print("==============================================")
    print()

    # ========================================================
    # CHECK REQUIRED FILES
    # ========================================================

    if not os.path.exists(CLI_PATH):

        print("[FATAL] CLI executable missing:")
        print(CLI_PATH)
        sys.exit(1)

    if not os.path.exists(MODEL_PATH):

        print("[FATAL] Model missing:")
        print(MODEL_PATH)
        sys.exit(1)

    if not os.path.exists(MMPROJ_PATH):

        print("[FATAL] MM projector missing:")
        print(MMPROJ_PATH)
        sys.exit(1)

    if not os.path.exists(DATASET_DIR):

        print("[FATAL] Dataset directory missing:")
        print(DATASET_DIR)
        sys.exit(1)

    print("CLI          :", CLI_PATH)
    print("MODEL        :", MODEL_PATH)
    print("MMPROJ       :", MMPROJ_PATH)
    print("DATASET      :", DATASET_DIR)
    print("RESULTS      :", OUTPUT_JSON)
    print("THINKING LOG :", THINKING_LOG_JSON)
    print()

    # ========================================================
    # LOAD FINAL RESULTS
    # ========================================================

    all_results = load_json(
        OUTPUT_JSON
    )

    total_saved = sum(
        len(images)
        for images in all_results.values()
        if isinstance(images, dict)
    )

    print(
        f"Loaded results checkpoint."
    )

    print(
        f"Previously processed images: {total_saved}"
    )

    print()

    # ========================================================
    # LOAD THINKING LOG
    # ========================================================

    thinking_logs = load_json(
        THINKING_LOG_JSON
    )

    total_logs = sum(
        len(images)
        for images in thinking_logs.values()
        if isinstance(images, dict)
    )

    print(
        f"Loaded thinking log."
    )

    print(
        f"Previously logged images: {total_logs}"
    )

    print()

    # ========================================================
    # FIND TASK FOLDERS
    # ========================================================

    task_folders = sorted(
        [
            f
            for f in os.listdir(DATASET_DIR)
            if os.path.isdir(
                os.path.join(
                    DATASET_DIR,
                    f
                )
            )
        ]
    )

    total_folders = len(
        task_folders
    )

    print(
        f"Found {total_folders} task folders."
    )

    print()

    # ========================================================
    # LOOP THROUGH TASK FOLDERS
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
        # Convert folder name to task
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

        print()
        print("==================================================")

        print(
            f"[{folder_idx}/{total_folders}] "
            f"Task: '{task_name}'"
        )

        print("==================================================")

        # ----------------------------------------------------
        # Create entries
        # ----------------------------------------------------

        if task_folder not in all_results:

            all_results[
                task_folder
            ] = {}

        if task_folder not in thinking_logs:

            thinking_logs[
                task_folder
            ] = {}

        # ----------------------------------------------------
        # Find images
        # ----------------------------------------------------

        images = sorted(
            [
                i
                for i in os.listdir(
                    task_path
                )
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
            f"Images in folder: {len(images)}"
        )

        # ====================================================
        # LOOP THROUGH IMAGES
        # ====================================================

        for img_idx, img_name in enumerate(
            images,
            1
        ):

            # =================================================
            # IMPORTANT:
            #
            # If image already exists in RESULTS JSON,
            # NEVER RUN IT AGAIN.
            #
            # STATUS DOES NOT MATTER.
            # =================================================

            if img_name in all_results[
                task_folder
            ]:

                previous_result = all_results[
                    task_folder
                ][img_name]

                if isinstance(
                    previous_result,
                    dict
                ):

                    previous_status = (
                        previous_result.get(
                            "status",
                            "unknown"
                        )
                    )

                else:

                    previous_status = "unknown"

                print(
                    f"  [{img_idx}/{len(images)}] "
                    f"Skipping {img_name} "
                    f"(already processed, "
                    f"status: {previous_status})",
                    flush=True
                )

                continue

            # =================================================
            # BRAND NEW IMAGE
            # =================================================

            print(
                f"  [{img_idx}/{len(images)}] "
                f"Evaluating {img_name}...",
                flush=True
            )

            img_path = os.path.join(
                task_path,
                img_name
            )

            # ------------------------------------------------
            # Verify image
            # ------------------------------------------------

            if not os.path.isfile(
                img_path
            ):

                parsed_json = {
                    "objects": [],
                    "affordances": [],
                    "status": "image_missing"
                }

                all_results[
                    task_folder
                ][img_name] = parsed_json

                thinking_logs[
                    task_folder
                ][img_name] = {

                    "thinking": "",
                    "raw_output": "",
                    "status": "image_missing"
                }

                save_json(
                    all_results,
                    OUTPUT_JSON
                )

                save_json(
                    thinking_logs,
                    THINKING_LOG_JSON
                )

                continue

            # =================================================
            # PROMPT
            # =================================================

            prompt = f"""
Look at the actual image and answer the task: "{task_name}".

Identify only objects that are visibly present and relevant to the task.
Do not assume an object exists just because it is related to the task.

Return ONLY valid JSON in exactly this format:

{{
    "objects": ["object1", "object2"],
    "affordances": [
        {{
            "object": "object",
            "affordance": "{task_name}",
            "region": "specific part of the object used",
            "action": "action to perform"
        }}
    ]
}}

If no visible object can perform the task, return:

{{
    "objects": ["visible object1", "visible object2"],
    "affordances": []
}}

Do not explain your reasoning.
Do not reason step-by-step.
Do not repeat yourself.
Return the JSON immediately.
"""
            # =================================================
            # RUN MODEL
            # =================================================

            raw_output = None
            execution_status = None

            try:

                raw_output, execution_status = (
                    run_qwen_vl_cli(
                        img_path,
                        prompt
                    )
                )

            except Exception as e:

                print(
                    f"       ⚠️ Model error: {e}",
                    flush=True
                )

                execution_status = (
                    "execution_error"
                )

            # =================================================
            # EXTRACT THINKING
            # =================================================

            thinking = extract_thinking(
                raw_output
            )

            # =================================================
            # SAVE THINKING LOG
            # =================================================

            thinking_logs[
                task_folder
            ][img_name] = {

                "thinking": thinking,

                "raw_output": (
                    raw_output[-5000:]
                    if raw_output
                    else ""
                ),

                "status": execution_status
            }

            # ------------------------------------------------
            # Save thinking immediately
            # ------------------------------------------------

            save_json(
                thinking_logs,
                THINKING_LOG_JSON
            )

            # =================================================
            # HANDLE EXECUTION STATUS
            # =================================================

            if execution_status == "timeout":

                parsed_json = {

                    "objects": [],
                    "affordances": [],
                    "status": "timeout"
                }

            elif execution_status == "execution_error":

                parsed_json = {

                    "objects": [],
                    "affordances": [],
                    "status": "execution_error"
                }

            else:

                # =============================================
                # PARSE MODEL OUTPUT
                # =============================================

                try:

                    parsed_json = (
                        parse_and_normalize_json(
                            raw_output
                        )
                    )

                except Exception as e:

                    parsed_json = {

                        "objects": [],
                        "affordances": [],

                        "status": "parser_exception",

                        "error": str(e)
                    }

            # =================================================
            # SAVE FINAL RESULT
            # =================================================

            all_results[
                task_folder
            ][img_name] = parsed_json

            # =================================================
            # SAVE FINAL RESULTS IMMEDIATELY
            # =================================================

            save_json(
                all_results,
                OUTPUT_JSON
            )

            # =================================================
            # PRINT STATUS
            # =================================================

            status = parsed_json.get(
                "status",
                "unknown"
            )

            if status == "success":

                print(
                    "       ✅ Prediction + thinking saved",
                    flush=True
                )

            elif status == "timeout":

                print(
                    "       ⏱️ Timeout saved.",
                    flush=True
                )

            else:

                print(
                    f"       ⚠️ Saved with status: "
                    f"{status}",
                    flush=True
                )

    # ========================================================
    # FINAL STATISTICS
    # ========================================================

    total_images = 0
    successful = 0
    failed = 0

    status_counts = {}

    for folder_results in all_results.values():

        if not isinstance(
            folder_results,
            dict
        ):
            continue

        for result in folder_results.values():

            total_images += 1

            if isinstance(
                result,
                dict
            ):

                status = result.get(
                    "status",
                    "unknown"
                )

                status_counts[status] = (
                    status_counts.get(
                        status,
                        0
                    ) + 1
                )

                if status == "success":

                    successful += 1

                else:

                    failed += 1

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print("==============================================")
    print("       DATASET TRAVERSAL COMPLETE")
    print("==============================================")

    print(
        f"Total images : {total_images}"
    )

    print(
        f"Successful   : {successful}"
    )

    print(
        f"Failed       : {failed}"
    )

    print()
    print("Status breakdown:")

    for status, count in sorted(
        status_counts.items()
    ):

        print(
            f"  {status}: {count}"
        )

    print()
    print(
        f"Results saved to:\n{OUTPUT_JSON}"
    )

    print()
    print(
        f"Thinking log saved to:\n{THINKING_LOG_JSON}"
    )

    print()


# ### You'll now have two files

# **Final predictions:**

# ```text
# E:\RM_sem7\full_dataset_evaluation_results.json
# ```

# **Qwen debugging/thinking:**

# ```text
# E:\RM_sem7\qwen3vl_thinking_log.json
# ```

# For example, the thinking log could contain:

# ```json
# {
#     "open_parcel": {
#         "000000123071.jpg": {
#             "thinking": "I need to inspect the image carefully. I can see a package...",
#             "raw_output": "<think>I need to inspect...",
#             "status": "completed"
#         }
#     }
# }
# ```

# And if an image times out, you'll potentially get something like:

# ```json
# {
#     "thinking": "I need to inspect the image. The object appears to...",
#     "raw_output": "<think>I need to inspect the image...",
#     "status": "timeout"
# }
# ```

# That is particularly useful because **we can later look at the timeout images and see whether Qwen was stuck in its thinking, whether it had already processed the image, or whether something else was happening.**

# One important point: **the main results JSON remains your source of truth for whether an image has been processed.** So even if the thinking log somehow gets out of sync, an image present in `full_dataset_evaluation_results.json` will never be rerun.
