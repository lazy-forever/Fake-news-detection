from transformers import BertTokenizer, BertModel
import torch
import numpy as np
import pandas as pd
import xmnlp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.get_device_name())
test_data = pd.read_csv('./data/test.feature.csv')
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

features_test = []
data_len=len(test_data['Title'])
data_p=int(data_len/100)
index=0

model.to(device)
model.eval()

with torch.no_grad():
    for text,comment in zip(test_data['Title'],test_data['Report Content']):
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
        # sequence_output = outputs[0].numpy().tolist()[0] #(1,n,1024)
        # pooled_output = outputs[1].numpy().tolist()[0]  #(1024)
        word_feature.extend(sequence_output)
        features_test.append(word_feature)

print("finish creating features_test")

max_len = max([len(i) for i in features_test])
X_test = []
for sen in features_test:
    tl = sen
    tl += [[0.] * 1024] * (max_len - len(tl))
    X_test.append(tl)

X_test = np.array(X_test)
np.save('./data/X_test',X_test)