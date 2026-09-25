# Visual Affordance Detection with Qwen Vision-Language Models

This repository studies whether multimodal language models can detect which objects in an image are useful for a given human task, such as:

- pour_sugar
- sit_comfortably
- step_on
- smear_butter
- serve_wine
- extinguish_fire
- open_parcel
- place_flowers

The project uses local Qwen vision-language models through llama.cpp, evaluates them on image datasets, and then converts the outputs into structured JSON, CSV, and Excel formats for analysis and later training.

## Topic and goal

The core research problem is visual affordance detection: given an image, identify objects that afford an action and describe the relevant region and action. For example, in an image of a chair and table, the model should recognize that the chair affords sitting, while scissors may afford cutting or opening a parcel.

This repo is built for benchmarking and experimentation with different multimodal model variants:

- Qwen3-VL thinking model
- Qwen3-VL instruct model
- Qwen2.5-VL instruct model
- a lightweight single-image inference script for debugging and testing

The workflow combines model inference, output parsing, result aggregation, and dataset conversion into a full evaluation pipeline.

---

## Project structure

```text
.
├── README.md
├── data/
│   ├── extinguish_fire/
│   ├── open_parcel/
│   ├── place_flowers/
│   ├── pour_sugar/
│   ├── serve_wine/
│   ├── sit_comfortably/
│   ├── smear_butter/
│   ├── step_on/
│   └── ...
├── llama.cpp/
│   └── local llama.cpp build used for multimodal inference
├── qwen3vl.py
├── qwen3vl_original.py
├── qwen3vl_instuct.py
├── qwen2_5vl.py
├── qwen3vl_wine.py
├── tenImage.py
├── fortrainingjson.py
├── json_to_csv.py
├── converting.py
├── qwen3vl_thinking_log.json
├── qwen2_5vl_instruct_results.json
├── qwen3vl_instruct_results.json
├── full_dataset_evaluation_results.json
├── affordance_results_*.csv
├── train_affordance.jsonl
├── baseline_evaluation_results.json
├── env/
└── .gitignore
```

---

## Main files and their roles

### Model evaluation scripts

#### qwen3vl.py
Primary evaluation script for the Qwen3-VL thinking model. It:

- locates the local llama.cpp multimodal CLI
- loads the GGUF model and mmproj file
- iterates through every image in the dataset folders
- sends a structured affordance prompt to the model
- parses the answer and extracts JSON
- writes aggregated results to full_dataset_evaluation_results.json
- stores a reasoning log in qwen3vl_thinking_log.json

#### qwen3vl_original.py
An earlier or alternate version of the Qwen3-VL evaluation pipeline. It implements the same general flow, but uses a retry-based subprocess runner and explicit thinking-strip logic before JSON extraction.

#### qwen3vl_instuct.py
Runs the instruct-tuned Qwen3-VL model. This variant is meant for comparison against the thinking model and typically produces cleaner structured outputs for affordance extraction.

#### qwen2_5vl.py
Runs the Qwen2.5-VL-3B instruct model. This script is used to compare a different model family and size against the Qwen3-VL variants.

#### qwen3vl_wine.py
Task-specific or variant evaluation script for the wine-related affordance scenario.

#### tenImage.py
A smaller test pipeline for quick single-image or limited-batch inference. It initializes the model using llama_cpp.Llama and can be used for debugging prompts, output format, and parsing behavior without running the full dataset.

### Data conversion and post-processing

#### fortrainingjson.py
Converts a CSV annotation table into a JSONL training dataset. It keeps only rows with verdict == YES, groups by task and image, and formats each example as a prompt + completion pair for training or fine-tuning.

#### json_to_csv.py
Converts JSON result files into flat CSV rows. It expands each image result into row-level object affordance records, with columns such as:

- task
- image
- object
- affordance
- region
- action
- has_affordance
- status

#### converting.py
Takes a CSV result file and writes an Excel workbook with embedded images for easier review and manual labeling. It automatically resolves the task and image columns and inserts the corresponding image into the spreadsheet.

---

## Dataset format

The dataset is organized under the data folder. Each task has its own directory, for example:

```text
data/
├── pour_sugar/
│   ├── image_001.jpg
│   ├── image_002.jpg
│   └── ...
├── sit_comfortably/
│   └── ...
└── ...
```

Each folder represents one affordance task. The model is prompted to determine which visible object in the image can be used for that task and to return JSON containing:

```json
{
  "objects": ["object1", "object2"],
  "affordances": [
    {
      "object": "object1",
      "affordance": "pour sugar",
      "region": "handle",
      "action": "grip and tilt"
    }
  ]
}
```

This structure is used throughout the pipeline for evaluation and later training conversion.

---

## Typical workflow

### 1. Prepare the environment

This project relies on a local Windows setup and a prebuilt llama.cpp multimodal stack.

Recommended steps:

```powershell
cd E:\RM_sem7
.\env\Scripts\activate
```

Then ensure that the local model files and llama.cpp build exist, especially:

- Qwen3VL-2B-Thinking-Q4_K_M.gguf
- Qwen3VL-2B-Instruct-Q4_K_M.gguf
- Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf
- mmproj-Qwen3VL-2B-Thinking-F16.gguf
- mmproj-Qwen3VL-2B-Instruct-F16.gguf
- mmproj-Qwen2.5-VL-3B-Instruct-f16.gguf
- llama.cpp/build/bin/Release/llama-mtmd-cli.exe

### 2. Run a model over the dataset

Example:

```powershell
python qwen3vl.py
```

or:

```powershell
python qwen2_5vl.py
```

These scripts iterate through task folders, process image-by-image, and save the extracted results to JSON files.

### 3. Convert JSON results into CSV

```powershell
python json_to_csv.py
```

This creates a tabular dataset for comparison or analysis.

### 4. Create training JSONL data

```powershell
python fortrainingjson.py
```

This produces a JSONL file for model training or fine-tuning.

### 5. Build Excel reports

```powershell
python converting.py
```

This generates a spreadsheet with the image embedded next to each result row.

---

## What the project is measuring

The benchmark focuses on whether a model can answer questions like:

- Which object in the image affords the task?
- What part of the object is relevant?
- What action is appropriate?
- Is the object actually useful for the target action?

This is more than object detection. The model must reason about object functionality and human interaction rather than just detect a category label.

---

## Observed output pattern

The scripts are designed to be robust to common inference issues:

- they strip out reasoning tags such as <think>...</think>
- they remove markdown fences around JSON answers
- they look for JSON objects inside raw model output
- they retry failed or timed-out executions
- they keep partial logs for debugging when inference fails unexpectedly

This makes the project practical for local experimentation with multimodal models on limited hardware.

---

## Research value

This repository is useful for:

- benchmarking affordance understanding in VLMs
- comparing model families and sizes
- analyzing how different prompt structures change object selection
- converting model predictions into structured training data
- preparing downstream classification, evaluation, and annotation workflows

---

## Notes

- The project is highly dependent on local model files and the llama.cpp multimodal build.
- The scripts assume a Windows directory layout and may need path adjustments if moved to Linux or macOS.
- Some scripts are experiment variants rather than the canonical pipeline, so the recommended starting point is qwen3vl.py or qwen2_5vl.py.

---

## Suggested starting points

If you are new to the project, start with these files in order:

1. README.md
2. qwen3vl.py
3. qwen2_5vl.py
4. json_to_csv.py
5. fortrainingjson.py
6. data/

These files capture the main evaluation loop, model comparison, and data preparation path.

---

## Summary

This repository is a research and evaluation pipeline for visual affordance detection using Qwen-based multimodal models. It connects dataset images, local model inference, structured JSON extraction, and downstream CSV/JSONL preparation in a single workflow. The project is designed for experimentation, comparison, and analysis of how well current vision-language models understand object affordances in real-world scenes.
