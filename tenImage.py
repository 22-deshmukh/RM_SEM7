import json
import os
import base64
import re
from llama_cpp import Llama

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Path configuration based on your local setup
MODEL_PATH = os.path.join(SCRIPT_DIR, "Qwen3VL-2B-Thinking-Q4_K_M.gguf")
MMPROJ_PATH = os.path.join(SCRIPT_DIR, "mmproj-Qwen3VL-2B-Thinking-F16.gguf")

# Point this to the root folder that contains your task subfolders (e.g., 'pour_sugar', 'sit_comfortably')
DATASET_DIR = os.path.join(SCRIPT_DIR, "data") 
OUTPUT_JSON = os.path.join(SCRIPT_DIR, "baseline_evaluation_results.json")

def initialize_model():
    """Initialize the Qwen-VL model with multimodal support via llama.cpp"""
    try:
        llm = Llama(
            model_path=MODEL_PATH,
            mmproj=MMPROJ_PATH,
            n_ctx=4096,
            n_threads=4,
            verbose=False,
            chat_format="chatml"
        )
        return llm
    except Exception as e:
        print(f"Error initializing model: {e}")
        raise

def run_qwen_vl(image_path, prompt):
    """Run zero-shot inference with Qwen-VL on the given image and prompt"""
    try:
        with open(image_path, "rb") as img_file:
            base64_data = base64.b64encode(img_file.read()).decode('utf-8')
            data_uri = f"data:image/jpeg;base64,{base64_data}"

        response = llm.create_chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": "You are a visual affordance detector. Respond concisely with valid JSON only."
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_uri}}
                    ]
                }
            ],
            temperature=0.2,
            top_p=0.9,
            repeat_penalty=1.1,
            max_tokens=2048
        )
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"Error during inference: {e}")
        return None

def parse_json_output(output):
    """Clean think tags and extract JSON from model output"""
    if not output:
        return None
    # Strip out reasoning tokens from thinking models
    cleaned_output = re.sub(r'<think>.*?(?:</think>|$)', '', output, flags=re.DOTALL).strip()
    text_to_parse = cleaned_output if cleaned_output else output

    json_start = text_to_parse.find("{")
    if json_start == -1:
        return {"raw_output": output}
        
    json_str = text_to_parse[json_start:]
    json_end = json_str.rfind("}") + 1
    json_str = json_str[:json_end]
    
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return {"raw_output": output}

if __name__ == "__main__":
    print("Initializing model...")
    llm = initialize_model()
    
    if not os.path.exists(DATASET_DIR):
        print(f"Error: Dataset directory '{DATASET_DIR}' not found. Update DATASET_DIR path to match your folder structure.")
        exit(1)

    all_results = {}

    # Get all task folders
    task_folders = [f for f in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, f))]
    total_folders = len(task_folders)
    
    print(f"\nFound {total_folders} task folders to process.\n")

    # Automatically iterate through each task folder structure
    for folder_idx, task_folder in enumerate(task_folders, 1):
        task_path = os.path.join(DATASET_DIR, task_folder)
        task_name = task_folder.replace("_", " ")
        
        print(f"\n==================================================")
        print(f"[{folder_idx}/{total_folders}] Processing Task Folder: '{task_folder}'")
        print(f"Goal/Affordance: {task_name}")
        print(f"==================================================")
        
        all_results[task_folder] = {}
        
        # Grab up to 10 images from the folder
        images = [img for img in os.listdir(task_path) if img.lower().endswith(('.jpg', '.jpeg', '.png'))][:10]
        total_images = len(images)
        
        if total_images == 0:
            print(f"⚠️ No valid images found in {task_folder}. Skipping.")
            continue

        print(f"Found {total_images} images to evaluate for this task.\n")

        for img_idx, img_name in enumerate(images, 1):
            img_path = os.path.join(task_path, img_name)
            print(f"  [{img_idx}/{total_images}] Evaluating image: {img_name}...")
            
            prompt = f"""You are an expert in visual affordance detection.
Analyze the image and identify:
1. Objects present
2. Which objects afford {task_name.upper()}
3. Exact region or part used for this task
4. Suggested action

Return JSON in format:
{{
  "objects": [],
  "affordances": [
    {{
      "object": "",
      "affordance": "{task_name}",
      "region": "",
      "action": ""
    }}
  ]
}}
"""
            raw_output = run_qwen_vl(img_path, prompt)
            parsed_json = parse_json_output(raw_output)
            
            all_results[task_folder][img_name] = parsed_json
            
            # Quick status print for individual image
            if "raw_output" in parsed_json:
                print(f"       -> Status: Completed (Parsed with fallback/raw text)")
            else:
                print(f"       -> Status: Success (Valid JSON extracted)")

    # Save results to disk
    with open(OUTPUT_JSON, "w") as f:
        json.dump(all_results, f, indent=4)
        
    print(f"\n==================================================")
    print(f"🎉 Execution complete! All results saved to {OUTPUT_JSON}")
    print(f"==================================================")