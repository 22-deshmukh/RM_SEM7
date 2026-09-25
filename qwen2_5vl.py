
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

# ============================================================
# QWEN3-VL INSTRUCT MODEL
# ============================================================

MODEL_PATH = os.path.join(
    SCRIPT_DIR,
    "E:\RM_sem7\Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf"
)

MMPROJ_PATH = os.path.join(
    SCRIPT_DIR,
    "mmproj-Qwen2.5-VL-3B-Instruct-f16.gguf"
)

DATASET_DIR = os.path.join(
    SCRIPT_DIR,
    "data"
)


# ============================================================
# INSTRUCT RESULTS
# ============================================================

OUTPUT_JSON = os.path.join(
    SCRIPT_DIR,
    "qwen2_5vl_instruct_results.json"
)


# ============================================================
# RUN QWEN3-VL INSTRUCT
# ============================================================

def run_qwen_vl_cli(image_path, prompt):

    """
    Run Qwen3-VL-Instruct through llama-mtmd-cli.

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
            "       Running Qwen2_5-VL-Instruct...",
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
    # Remove markdown code fences
    # --------------------------------------------------------

    cleaned_text = re.sub(
        r"```json",
        "",
        raw_text,
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

    if not os.path.exists(output_path):

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
    print("   QWEN3-VL INSTRUCT DATASET EVALUATION")
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


    print("CLI     :", CLI_PATH)
    print("MODEL   :", MODEL_PATH)
    print("MMPROJ  :", MMPROJ_PATH)
    print("DATASET :", DATASET_DIR)
    print("RESULTS :", OUTPUT_JSON)

    print()


    # ========================================================
    # LOAD PREVIOUS RESULTS
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
        "Loaded results checkpoint."
    )

    print(
        f"Previously processed images: {total_saved}"
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
        # Create result entry
        # ----------------------------------------------------

        if task_folder not in all_results:

            all_results[
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
            # SKIP ALREADY PROCESSED IMAGE
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


                save_json(
                    all_results,
                    OUTPUT_JSON
                )

                continue


            # =================================================
            # PROMPT
            # ====================================================

            prompt = f"""
Look at the image and try to answer the task: "{task_name}".

Find the object or objects in the image that can be used for this task.

Use your best judgment based on what you can see.
The object should be reasonably useful for doing the task, but it does not need to be perfectly obvious.

Do not worry about background objects unless they can help with the task.
Do not include the target object just because it is mentioned in the task.

For example:
- "cut paper" → scissors are useful, paper is the target.
- "sit on a chair" → chair is useful.
- "drink from a bottle" → bottle is useful.
- "open a parcel" → a visible tool such as scissors, knife, or the parcel itself may be relevant depending on the image.

If more than one object could help, include them.

For each useful object, give the relevant part and a simple action.

Return ONLY JSON in this format:

{{
    "objects": ["object1", "object2"],
    "affordances": [
        {{
            "object": "object1",
            "affordance": "{task_name}",
            "region": "part of object used",
            "action": "simple action"
        }}
    ]
}}

If you are not completely sure, make your best reasonable guess from the image rather than returning empty lists.

Return ONLY the JSON.
"""


            # =================================================
            # RUN INSTRUCT MODEL
            # ====================================================

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
            # HANDLE EXECUTION STATUS
            # ====================================================

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
            # SAVE RESULT
            # ====================================================

            all_results[
                task_folder
            ][img_name] = parsed_json


            save_json(
                all_results,
                OUTPUT_JSON
            )


            # =================================================
            # PRINT STATUS
            # ====================================================

            status = parsed_json.get(
                "status",
                "unknown"
            )


            if status == "success":

                print(
                    "       ✅ Prediction saved",
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
    print("       INSTRUCT DATASET TRAVERSAL COMPLETE")
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

    print(
        "Status breakdown:"
    )


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

