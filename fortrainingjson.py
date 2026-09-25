import pandas as pd
import json
import os

def convert_csv_to_jsonl(csv_path, output_jsonl_path="train_affordance.jsonl"):
    print(f"Loading CSV from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Filter for valid rows where verdict is YES
    df_valid = df[df['verdict'].astype(str).str.strip().str.upper() == 'YES'].copy()
    print(f"Found {len(df_valid)} valid object-affordance annotations.")
    
    records = []
    # Group by task and image to collect all objects and affordances per image
    grouped = df_valid.groupby(['task', 'image'])
    
    for (task, image), group in grouped:
        task_name = str(task).replace('_', ' ').strip()
        
        # Extract unique object names
        objects = group['object'].dropna().unique().tolist()
        
        # Build affordances list matching your prompt format
        affordances = []
        for _, row in group.iterrows():
            affordances.append({
                "object": str(row['object']),
                "affordance": task_name,
                "region": str(row['region']) if pd.notna(row['region']) else "entire object",
                "action": str(row['action']) if pd.notna(row['action']) else "use"
            })
        
        # Reconstruct the exact prompt used during inference
        prompt = f"""Look at the image and try to answer the task: "{task_name}".

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

Return ONLY the JSON."""

        target_dict = {
            "objects": objects,
            "affordances": affordances
        }
        
        # Format completion with reasoning and answer tags for VLM-R1 / GRPO training
        completion = f"<think>\nBased on visual evidence in the image, the tools capable of performing '{task_name}' are identified along with their regions and actions.\n</think>\n<answer>\n{json.dumps(target_dict, indent=4)}\n</answer>"

        records.append({
            "image": image,
            "task": task_name,
            "prompt": prompt,
            "completion": completion,
            "ground_truth": target_dict
        })
        
    # Write out to JSONL
    with open(output_jsonl_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            
    print(f"Successfully wrote {len(records)} training records to {output_jsonl_path}!")

if __name__ == "__main__":
    convert_csv_to_jsonl("affordance_results_3vl__for_tuning.csv")