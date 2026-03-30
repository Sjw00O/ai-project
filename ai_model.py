# 前沿AI图像分类模型
import tensorflow as tf
from tensorflow.keras import layers

# 构建CNN模型（可迁移、调参）
model = tf.keras.Sequential([
    layers.Rescaling(1./255, input_shape=(64,64,3)),
    layers.Conv2D(64, 3, activation='relu'),  # 参数可调
    layers.MaxPooling2D(),
    layers.Conv2D(128, 3, activation='relu'), # 参数可调
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),    # 参数可调
    layers.Dense(10, activation='softmax')    # 10分类
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("=== AI模型构建完成 ===")
model.summary()
print("\n可进行：模型迁移学习、参数优化、训练测试")
