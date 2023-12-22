import pandas as pd
import csv

# 分隔数值
split=0.97

with open('predict.csv', 'r') as f:
    predict = csv.reader(f)
pre = [i[0] for i in predict]
prediction = [int(float(i) < split) for i in pre]
id = [str(i) for i in range(1,10141)]

series1 = pd.Series(prediction)
series2 = pd.Series(id)
print(1, sum(prediction), 0, len(prediction) - sum(prediction))

pd.DataFrame({'id': series2, 'label': series1}).to_csv('submit.csv', index=False)