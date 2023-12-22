import numpy as np
import pandas as pd
from threading import Thread
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC, SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings("ignore")

data=pd.read_csv("./data/train.news.csv")
data_validate=pd.read_csv("./data/test.feature.csv")
x,y,x_validate=data['Title'],data['label'],data_validate['Title']
print(x.shape,y.shape,x_validate.shape)
print("读取数据完成")

vectorizer=TfidfVectorizer()
x=vectorizer.fit_transform(x)
x_validate=vectorizer.transform(x_validate)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,stratify=y,random_state = 0)
x_validate1,x_validate2=train_test_split(x_validate,test_size=0.5,random_state=0)

# 线性支持向量机
model1 = LinearSVC()
# 朴素贝叶斯
model2 = MultinomialNB()
# K近邻
model3 = KNeighborsClassifier(n_neighbors=50)
# 决策树
model4 = DecisionTreeClassifier(random_state=77)
# 随机森林
model5 = RandomForestClassifier(n_estimators=500, max_features='sqrt', random_state=10)
# 梯度提升
model6 = GradientBoostingClassifier(random_state=123)
# 支持向量机
model7 = SVC(kernel="rbf", random_state=77)
# 神经网络
model8 = MLPClassifier(hidden_layer_sizes=(16, 8), random_state=77, max_iter=10000)
# AdaBoostClassifier
model9 = AdaBoostClassifier(n_estimators=100, random_state=77)
# 逻辑回归
model10 = LogisticRegression(random_state=77)

model_list = [model1, model2, model3, model4, model5, model6, model7, model8, model9, model10]
model_name = ['线性支持向量机', '朴素贝叶斯', 'K近邻', '决策树', '随机森林', '梯度提升', '支持向量机', '神经网络', 'AdaBoostClassifier', '逻辑回归']
threads = []

def fit(model, name:str):
    model.fit(x_train, y_train)
    s=model.score(x_test, y_test)
    pre=model.predict(x_validate)
    np.savetxt(f"./data/机器学习{name}方法prediction.csv",pre,delimiter=',',fmt='%f')
    print(f'{name}方法在测试集的准确率为{s}')

for i in range(len(model_list)):
    model=model_list[i]
    name=model_name[i]
    threads.append(Thread(target=fit, args=(model, name)))

for thread in threads:
    thread.start()
for thread in threads:
    thread.join()
print("done!")