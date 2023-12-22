import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import xmnlp
xmnlp.set_model('./model/xmnlp-onnx-models')

# 训练轮数
epoch = 17

X_test=np.load("./data/X_test.npy")
X_train=np.load("./data/X_train.npy")
train_data = pd.read_csv('./data/train.news.csv')
test_data = pd.read_csv('./data/test.feature.csv')
Y_train = train_data['label']


## 加载Title情感分析
# def SentimentAnalysis(text):
#     x = list(xmnlp.sentiment(text))
#     x.extend([0.]*1022)
#     return x
# train_sen=np.zeros((10587,1,1024))
# test_sen=np.zeros((10141,1,1024))
# for i in range(10587):
#     train_sen[i,0,:]=SentimentAnalysis(train_data['Title'][i])
# for i in range(10141):
#     test_sen[i,0,:]=SentimentAnalysis(test_data['Title'][i])
# X_test=np.concatenate((X_test,test_sen),axis=1)
# X_train=np.concatenate((X_train,train_sen),axis=1)


## 是否排除Content情感分析
# X_test=X_test[:,1:,:]
# X_train=X_train[:,1:,:]

## 加载文章是否被删除的标记
# file=open('./data/url_check.txt','r')
# import json
# train_web=json.loads(file.readline())
# test_web=json.loads(file.readline())
# file.close()
# train_web_array=np.zeros((10587,1,1024))
# test_web_array=np.zeros((10141,1,1024))
# for i in range(10587):
#     train_web_array[i,0,0]=float(train_web[str(i+1)])
# for i in range(10141):
#     test_web_array[i,0,0]=float(test_web[str(i+1)])
# X_test=np.concatenate((X_test,test_web_array),axis=1)
# X_train=np.concatenate((X_train,train_web_array),axis=1)

# for i in range(10587):
#     X_train[i,0,-1]=float(train_web[str(i+1)])
# for i in range(10141):
#     X_test[i,0,-1]=float(test_web[str(i+1)])

np.random.seed(1)
np.random.shuffle(X_train)
np.random.seed(1)
np.random.shuffle(Y_train)
print("finish shuffling data")

split = len(X_train) // 7
X_validation = X_train[:split]
X_train_split = X_train[split:]
Y_validation = Y_train[:split]
Y_train_split = Y_train[split:]
print("finish loading data")

def cnn(X_train):
    model = tf.keras.Sequential([
        tf.keras.layers.Convolution1D(input_shape=(X_train.shape[1], X_train.shape[2]),
                                      filters=128, kernel_size=3, activation='relu'),
        tf.keras.layers.Dropout(rate=0.5),
        tf.keras.layers.MaxPool1D(),
        tf.keras.layers.Convolution1D(128, 4, activation='relu'),
        tf.keras.layers.Dropout(rate=0.5),
        tf.keras.layers.MaxPool1D(),
        tf.keras.layers.Convolution1D(64, 5),
        tf.keras.layers.Dropout(rate=0.5),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(rate=0.5),
        tf.keras.layers.Dense(2, activation='softmax'),
])
    model.compile(loss='sparse_categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['accuracy'])
    print(model.summary())
    return model


model = cnn(X_train)

print("finish creating model")

history = model.fit(X_train_split, Y_train_split, epochs=epoch,
                    batch_size=128, verbose=1,
                    validation_data=(X_validation, Y_validation))

model.save('fakenews_model')
print("finish training model")

predictions = model.predict(X_test)
np.savetxt('predict.csv', predictions, delimiter=',', fmt='%f')
print("finish predicting")


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training', 'validation'], loc='upper left')
plt.savefig('1.png')