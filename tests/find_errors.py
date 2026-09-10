import pandas as pd
import numpy as np
from pathlib import Path
import tensorflow as tf
from src.data.preprocessing import load_and_preprocess_image
from src.config import CLASSES, MODELS_DIR

def main():
    model = tf.keras.models.load_model(MODELS_DIR / 'efficientnet_finetuned.keras')
    df = pd.read_csv('data/splits/test.csv')
    scratches_df = df[df['class'] == 'scratches'].copy()

    batch = []
    for _, row in scratches_df.iterrows():
        img_rgb = load_and_preprocess_image(Path(row['filepath']).read_bytes())
        batch.append(tf.keras.applications.efficientnet.preprocess_input(img_rgb))

    preds = model.predict(np.array(batch), batch_size=32, verbose=0)
    pred_classes = [CLASSES[i] for i in preds.argmax(axis=1)]
    scratches_df['pred'] = pred_classes
    scratches_df['conf'] = preds.max(axis=1)

    errors = scratches_df[scratches_df['pred'] != 'scratches']
    print(f"Total Scratches in test split: {len(scratches_df)}")
    print(f"Total Misclassified: {len(errors)}")
    for _, r in errors.iterrows():
        conf_pct = r['conf'] * 100
        print(f"  - File: {r['filename']} | True: {r['class']} | Pred: {r['pred']} | Conf: {conf_pct:.2f}%")

if __name__ == '__main__':
    main()
