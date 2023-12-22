from transformers import BertTokenizer, BertModel
import torch
import numpy as np
import pandas as pd
import xmnlp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.get_device_name())
train_data = pd.read_csv('./data/train.news.csv')
print("finish read csv")

model_name = './model/chinese_roberta_wwm_large_ext_pytorch'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name, config='./model/chinese_roberta_wwm_large_ext_pytorch/bert_config.json')
xmnlp.set_model('./model/xmnlp-onnx-models')

# 情感判断
def SentimentAnalysis(text):
    x = list(xmnlp.sentiment(text))
    x.extend([0.]*1022)
    return x

features_train = []
data_len=len(train_data['label'])
data_p=int(data_len/100)
index=0

model.to(device)
model.eval()

with torch.no_grad():
    for text,comment in zip(train_data['Title'],train_data['Report Content']):
        index += 1
        if index % data_p == 0:
            if index % (data_p*10) <data_p:
                print(str(int(index/data_p))+'%')
            else:
                print('#',end='')

        word_feature = [SentimentAnalysis(comment)]
        input_ids = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        input_ids.to(device)
        outputs = model(**input_ids)
        sequence_output = outputs.last_hidden_state.squeeze(0).cpu().numpy()
        word_feature.extend(sequence_output)
        features_train.append(word_feature)

print("finish creating features_train")

max_len = max([len(i) for i in features_train])
X_train = []
for sen in features_train:
    tl = sen
    tl += [[0.] * 1024] * (max_len - len(tl))
    X_train.append(tl)


X_train = np.array(X_train)
np.save('./data/X_train',X_train)