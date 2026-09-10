import requests
import pandas as pd
import json
import os
import sys

def main():
    df_test = pd.read_csv('data/splits/test.csv')
    print(f'Total test images: {len(df_test)}')

    classes = df_test['class'].unique()
    results = []

    print('\nTesting 1 sample image per defect class against /api/v1/predict and /api/v1/explain...')

    for cls in classes:
        sample_row = df_test[df_test['class'] == cls].iloc[0]
        img_path = sample_row['filepath']
        true_label = sample_row['class']
        filename = sample_row['filename']
        
        with open(img_path, 'rb') as f:
            img_bytes = f.read()
            
        # 1. Test /api/v1/predict
        r_pred = requests.post(
            'http://127.0.0.1:8000/api/v1/predict',
            files={'file': (filename, img_bytes, 'image/bmp')}
        )
        assert r_pred.status_code == 200, f'Prediction failed: {r_pred.text}'
        data = r_pred.json()
        
        # 2. Test /api/v1/explain
        r_cam = requests.post(
            'http://127.0.0.1:8000/api/v1/explain',
            files={'file': (filename, img_bytes, 'image/bmp')},
            data={'model_name': 'efficientnet_finetuned'}
        )
        assert r_cam.status_code == 200, f'GradCAM failed: {r_cam.text}'
        cam_data = r_cam.json()
        
        pred_cls = data['primary_prediction']
        is_correct = (pred_cls == true_label)
        
        results.append({
            'true_label': true_label,
            'primary_pred': pred_cls,
            'correct': is_correct,
            'consensus': f"{data['consensus_count']}/{data['total_models']} ({data['consensus_status']})",
            'confidence_tier': data['confidence_tier'],
            'recommendation': data['recommendation'],
            'effnet_pred': data['models']['efficientnet_finetuned']['predicted_class'],
            'effnet_conf': f"{data['models']['efficientnet_finetuned']['confidence']*100:.2f}%",
            'custom_pred': data['models']['custom_cnn']['predicted_class'],
            'custom_conf': f"{data['models']['custom_cnn']['confidence']*100:.2f}%",
            'heatmap_valid': bool(cam_data.get('heatmap_base64')),
            'overlay_valid': bool(cam_data.get('overlay_base64')),
            'inspection_id': data.get('inspection_id')
        })

    # Check audit history
    r_hist = requests.get('http://127.0.0.1:8000/api/v1/history?limit=10')
    assert r_hist.status_code == 200
    history = r_hist.json()

    print('\n=== LIVE PREDICTION VERIFICATION ACROSS ALL 6 DEFECT CLASSES ===')
    print(json.dumps(results, indent=2))
    print(f'\nTotal inspections logged in SQLite database: {len(history)}')
    
    # Check all are correct
    all_correct = all(r['correct'] for r in results)
    print(f'\nAll 6 test samples predicted correctly: {all_correct}')

if __name__ == '__main__':
    main()
