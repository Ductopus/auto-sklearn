from pprint import pprint
import pandas as pd
import sklearn.datasets
import sklearn.metrics

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score, accuracy_score
from sklearn.metrics import mean_squared_error

import autosklearn.classification
df = pd.read_csv('/mnt/e/Workspace/Moonlighting/ASD Dignositc model/AutoML/Data/Datacleaned_XY_ALL.csv')
# df = pd.read_csv('E:\\Workspace\\Moonlighting\\ASD Dignositc model\\AutoML\\Data\\Datacleaned_XY_ALL.csv')
# 自变量（该数据集的前13项）
X = df.iloc[:, :-1].values
# 因变量（该数据集的最后1项，即第14项）
y = df.iloc[:, -1].values

f = open('XY_ALL.txt','w')
# print(X)
# print(y)

#关注从5：5到9：1的寻训练集与测试集的划分
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=123)

automl = autosklearn.classification.AutoSklearnClassifier(
    time_left_for_this_task=120,
    per_run_time_limit=30,
   # tmp_folder="/tmp/",
)
automl.fit(X_train, y_train, dataset_name="df")
print(automl.leaderboard(), file=f)
pprint(automl.show_models(), indent=4)
pprint(automl.sprint_statistics())
predictions = automl.predict(X_test)
print("Accuracy score:", sklearn.metrics.balanced_accuracy_score(y_test, predictions), file=f)

