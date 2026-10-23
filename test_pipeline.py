import os
from services.data_service import get_all_plants, get_latest_observation
from services.cnn_service import analyze_image_with_gradcam
from services.lstm_service import predict_growth_4days
from services.anomaly_service import detect_anomalies_for_plant
from ai.openai_service import run_chat_query

print("=== 1. TEST DATA SERVICE ===")
plants = get_all_plants()
print("Plants:", plants['plant_id'].tolist())

obs = get_latest_observation("PCK-001")
print("Latest obs PCK-001:", obs)

print("\n=== 2. TEST CNN + GRAD-CAM SERVICE ===")
sample_img = r"C:\Users\Asus\.gemini\antigravity\scratch\smart-farming-pakcoy-new\data\images\Siap_Panen\Siap_Panen_073.jpg"
if os.path.exists(sample_img):
    cnn_res = analyze_image_with_gradcam(sample_img)
    print("CNN Result:", cnn_res['predicted_label'], "Confidence:", cnn_res['confidence'])
    print("Probabilities:", cnn_res['probabilities'])
    print("Gradcam Image Size:", cnn_res['gradcam_image'].size)
else:
    print("Sample img not found")

print("\n=== 3. TEST LSTM PREDICTION SERVICE ===")
pred_res = predict_growth_4days("PCK-001")
if pred_res:
    print("Base Day:", pred_res['base_day'], "Latest Mat:", pred_res['latest_maturity'])
    print("Daily Summary:")
    print(pred_res['daily_summary_df'])
else:
    print("Prediction failed")

print("\n=== 4. TEST ANOMALY DETECTION ===")
anomalies = detect_anomalies_for_plant("PCK-001")
print(f"Detected {len(anomalies)} anomalies for PCK-001.")
for a in anomalies:
    print(f" - [{a['severity']}] {a['title']}: {a['value']}")

print("\n=== 5. TEST AI ASSISTANT QUERY ===")
reply = run_chat_query(
    messages=[{"role": "user", "content": "Bagaimana perkembangan PCK-001?"}],
    current_plant_id="PCK-001"
)
print("AI Reply:")
print(reply['content'][:300] + "...")
