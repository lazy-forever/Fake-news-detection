import pandas as pd
import jieba
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import xmnlp

# 获取数据
train_data = pd.read_csv('./data/train.news.csv')
test_data = pd.read_csv('./data/test.feature.csv')
print("finish read csv")



# 获取预训练 word2vec 并构建词表
word2vec = open("./data/sgns.sogounews.bigram-char", "r", encoding='UTF-8')
t = word2vec.readline().split()
n, dimension = int(t[0]), int(t[1])
print(n)
print(dimension)
wordAndVec = word2vec.readlines()
wordAndVec = [i.split() for i in wordAndVec]
vectorsMap = []
word2index = {}
index2word = {}
for i in range(n):
    vectorsMap.append(list(map(float, wordAndVec[i][len(wordAndVec[i]) - dimension:])))
    word2index[wordAndVec[i][0]] = i
    index2word[i] = wordAndVec[i][0]

word2vec.close()
print("finish reading")

# 情感判断
def SentimentAnalysis(text):
    xmnlp.set_model('./model/xmnlp-onnx-models')
    x = list(xmnlp.sentiment(text))
    x.extend([0.]*298)
    return x

# jieba 分词与词向量构建
features_train = []
features_test = []
for text,comment in zip(train_data['Title'],train_data['Report Content']):
    # print(len(SentimentAnalysis(comment)),len([0]))
    word_feature = [SentimentAnalysis(comment)]
    for word in jieba.cut(text):
        if word in word2index:
            word_feature.append(vectorsMap[word2index[word]])
    features_train.append(word_feature)

for text,comment in zip(test_data['Title'],test_data['Report Content']):
    word_feature = [SentimentAnalysis(comment)]
    # word_feature=[]
    for word in jieba.cut(text):
        if word in word2index:
            word_feature.append(vectorsMap[word2index[word]])
    features_test.append(word_feature)


print("finish creating features")


# 模型输入构建
max_len1 = max([len(i) for i in features_train])
max_len2 = max([len(i) for i in features_test])
max_len = max(max_len1, max_len2)
X_train = []
X_test = []
for sen in features_train:
    tl = sen
    tl += [[0] * 300] * (max_len - len(tl))
    X_train.append(tl)
for sen in features_test:
    tl = sen
    tl += [[0] * 300] * (max_len - len(tl))
    X_test.append(tl)

print("finish creating X_train X_test")

Y_train = train_data['label']

X_train = np.array(X_train)
X_test = np.array(X_test)
Y_train = np.array(Y_train)

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



# 模型构建
def cnn(X_train):
    model = tf.keras.Sequential([
        tf.keras.layers.Convolution1D(input_shape=(X_train.shape[1], X_train.shape[2]),
                                      filters=128, kernel_size=3, activation='relu'),
        tf.keras.layers.Dropout(rate=0.5),  # Added dropout layer after the first convolutional layer
        tf.keras.layers.MaxPool1D(),
        tf.keras.layers.Convolution1D(128, 4, activation='relu'),
        tf.keras.layers.Dropout(rate=0.5),  # Added dropout layer after the second convolutional layer
        tf.keras.layers.MaxPool1D(),
        tf.keras.layers.Convolution1D(64, 5),
        tf.keras.layers.Dropout(rate=0.5),  # Added dropout layer after the third convolutional layer
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(rate=0.5),  # Dropout layer before the dense layer (unchanged)
        tf.keras.layers.Dense(2, activation='softmax'),
    ])
    # lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    #     initial_learning_rate=1e-2,
    #     decay_steps=10000,
    #     decay_rate=0.9)
    # model.compile(loss='sparse_categorical_crossentropy',
    #           optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
    #           metrics=['accuracy'])
    model.compile(loss='sparse_categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['accuracy'])
    print(model.summary())
    return model


# 模型训练
model = cnn(X_train)

print("finish creating model")

history = model.fit(X_train_split, Y_train_split, epochs=20,
                    batch_size=128, verbose=1,
                    validation_data=(X_validation, Y_validation))

model.save('fakenews_model')

predictions = model.predict(X_test)

np.savetxt('predict.csv', predictions, delimiter=',', fmt='%f')


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training', 'validation'], loc='upper left')
plt.savefig('1.png')
