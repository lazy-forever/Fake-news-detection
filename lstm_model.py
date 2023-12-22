import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import json

# 训练轮数
epoch = 15

# 加载文章是否被删除的标记
# file=open('./data/url_check.txt','r')
# train_web=json.loads(file.readline())
# test_web=json.loads(file.readline())
# file.close()

X_test=np.load("./data/X_test.npy")
X_train=np.load("./data/X_train.npy")
train_data = pd.read_csv('./data/train.news.csv')
Y_train = train_data['label']

# 是否采用情感分析
X_test=X_test[:,1:,:]
X_train=X_train[:,1:,:]

# 是否加入网页信息
# train_web_array=np.zeros((10587,1,1024))
# test_web_array=np.zeros((10141,1,1024))
# for i in range(10587):
#     train_web_array[i,0,0]=float(train_web[str(i+1)])
# for i in range(10141):
#     test_web_array[i,0,0]=float(test_web[str(i+1)])
# X_test=np.concatenate((X_test,test_web_array),axis=1)
# X_train=np.concatenate((X_train,train_web_array),axis=1)

np.random.seed(1)
np.random.shuffle(X_train)
np.random.seed(1)
np.random.shuffle(Y_train)

split = len(X_train) // 7
X_validation = X_train[:split]
X_train_split = X_train[split:]
Y_validation = Y_train[:split]
Y_train_split = Y_train[split:]
print("finish loading data")

def lstm(X_train):
    model=tf.keras.Sequential([
        tf.keras.layers.LSTM(128,input_shape=(X_train.shape[1],X_train.shape[2]),return_sequences=True),
        tf.keras.layers.Dropout(rate=0.5),
        tf.keras.layers.LSTM(128,return_sequences=True),
        tf.keras.layers.Dropout(rate=0.5),
        tf.keras.layers.LSTM(64),
        tf.keras.layers.Dropout(rate=0.5),
        tf.keras.layers.Dense(2,activation='softmax')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['sparse_categorical_accuracy'])
    print(model.summary())
    return model

model=lstm(X_train_split)

print("finish creating model")

history=model.fit(X_train_split,Y_train_split,epochs=epoch,
                  batch_size=64,validation_data=(X_validation,Y_validation))
model.save("lstm")
predictions = model.predict(X_test)

np.savetxt('predict.csv', predictions, delimiter=',', fmt='%f')

print(history.history.keys())
plt.plot(history.history['sparse_categorical_accuracy'])
plt.plot(history.history['val_sparse_categorical_accuracy'])
plt.legend(['training', 'validation'], loc='upper left')
plt.savefig('1.png')