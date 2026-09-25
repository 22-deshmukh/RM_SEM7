import os
import pandas as pd
import xlsxwriter

BASE_DIR = r"E:\RM_sem7"
input_file = os.path.join(BASE_DIR, "affordance_all_results.csv")
output_file = os.path.join(BASE_DIR, "affordance_all_results_with_images.xlsx")

df = pd.read_csv(input_file)

# Automatically find task and image column names regardless of exact casing
task_col = next((c for c in df.columns if 'task' in c.lower()), df.columns[0])
img_col = next((c for c in df.columns if any(k in c.lower() for k in ['image', 'img', 'file', 'photo', 'name'])), df.columns[1])

print(f"Detected Columns -> Task Column: '{task_col}' | Image Column: '{img_col}'")

writer = pd.ExcelWriter(output_file, engine='xlsxwriter')
df.to_excel(writer, sheet_name='Results', index=False)

worksheet = writer.sheets['Results']
image_column = len(df.columns)
worksheet.write(0, image_column, 'actual_image')
worksheet.set_column(image_column, image_column, 25)

success_count = 0

for row_num in range(len(df)):
    task_val = str(df.iloc[row_num][task_col]).strip()
    img_val = str(df.iloc[row_num][img_col]).strip()
    
    # Append .jpg if missing in the CSV value
    img_val_ext = img_val if any(img_val.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png']) else f"{img_val}.jpg"

    # Possible path variations
    candidates = [
        os.path.join(BASE_DIR, "data", task_val, img_val),
        os.path.join(BASE_DIR, "data", task_val, img_val_ext),
        os.path.join(BASE_DIR, "data", img_val),
        os.path.join(BASE_DIR, "data", img_val_ext),
    ]
    
    img_path = next((cand for cand in candidates if os.path.isfile(cand)), None)
    
    cell_row = row_num + 1
    worksheet.set_row(cell_row, 100)
    
    if img_path:
        worksheet.insert_image(cell_row, image_column, img_path, 
                               {'x_scale': 0.4, 'y_scale': 0.4, 'object_position': 1})
        success_count += 1
    else:
        if row_num == 0:
            print("\n--- DEBUG (Row 0 Failed) ---")
            print(f"CSV Task Cell: '{task_val}'")
            print(f"CSV Image Cell: '{img_val}'")
            print(f"First Path Checked: '{candidates[0]}'")
            print("----------------------------\n")
        worksheet.write(cell_row, image_column, 'Image not found')

writer.close()
print(f"Done! Successfully inserted {success_count}/{len(df)} images into {output_file}")