import tensorflow as tf
import tf2onnx
import onnx
import os

# 1. تحديد المسارات
model_path = 'best_model.keras'
output_onnx_path = "lstm_model_v4.onnx"

# 2. تحميل النموذج مع تعطيل الـ compile لتجنب خطأ loss_fn
print("[1/3] Loading Keras model...")
model = tf.keras.models.load_model(model_path, compile=False)

# 3. تحديد الأبعاد الصحيحة بناءً على كود التدريب v4
# التدريب كان على (50, 6)
# None تعني أن الـ Batch size متغير
input_shape = (None, 50, 6) 
spec = (tf.TensorSpec(input_shape, tf.float32, name="input"),)

# 4. البدء في التحويل
print("[2/3] Converting to ONNX...")
model_proto, _ = tf2onnx.convert.from_keras(
    model, 
    input_signature=spec, 
    opset=17 # إصدار 17 يدعم عمليات الـ LSTM و CNN المعقدة في موديلك
)

# 5. حفظ الملف النهائي
onnx.save(model_proto, output_onnx_path)
print(f"[3/3] ✅ Done! Model saved as: {output_onnx_path}")