import os
import shutil
import pandas as pd

def main():
    dest_dir = 'test_images_to_try'
    os.makedirs(dest_dir, exist_ok=True)

    df_test = pd.read_csv('data/splits/test.csv')

    # Select 2 distinct test samples per class from the test split
    selected_samples = []
    for cls in sorted(df_test['class'].unique()):
        cls_df = df_test[df_test['class'] == cls]
        for idx, (_, row) in enumerate(cls_df.head(2).iterrows(), 1):
            src_path = row['filepath']
            target_name = f"{cls}_sample_{idx}.jpg"
            target_path = os.path.join(dest_dir, target_name)
            shutil.copy2(src_path, target_path)
            selected_samples.append({
                'class': cls,
                'file': target_name,
                'path': os.path.abspath(target_path)
            })

    print("Successfully prepared sample test images in test_images_to_try/:")
    for s in selected_samples:
        print(f"  - {s['file']} [Class: {s['class']}]")

if __name__ == '__main__':
    main()
