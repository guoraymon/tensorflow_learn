import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)
print(tf.keras.__version__)

# 生成模拟数据
train_x = np.linspace(-1, 1, 100)  # 使用linspace均分函数在-1,1之间平均取100个点
noise = np.random.randn(*train_x.shape) * 0.3  # 噪音
train_y = 2 * train_x + noise  # y=2x+noise

# # 构建模型
# model = tf.keras.Sequential()
# model.add(tf.keras.layers.Dense(units=1, input_shape=(1,)))
#
# print(model.predict(train_x))
#
# # 编译模型
# model.compile(
#     optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
#     loss='mse'  # 优化器 loss_mse = tf.keras.losses.MSE(y,model.predict(x))
# )
#
# # 训练模型
# history = model.fit(train_x, train_y, batch_size=2, epochs=100)
#
# # 保存模型
# model.save('models/y_2x')

# 加载模型
model = tf.keras.models.load_model('models/y_2x')

w, b = model.layers[0].get_weights()
print('w=', w, 'b=', b)  # ('w=', array([[0.2669799]], dtype=float32), 'b=', array([0.09615658], dtype=float32))

# print(model.predict(pd.Series([2, 5])))  # 序列预测    [[0.6301164][1.4310561]]

plt.scatter(train_x, train_y)
plt.plot(train_x, model.predict(train_x))
plt.show()
