from pprint import pprint
import pandas as pd
import sklearn.datasets
import sklearn.metrics


from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score, accuracy_score
from sklearn.metrics import mean_squared_error

import autosklearn.classification
# my_list = [0.5,0.4,0.3,0.2,0.1]
# Meta_scale = pd.read_csv('/mnt/e/Workspace/Moonlighting/ASD Dignositc model/AutoML/Data/Meta_scale.csv')
# # df = pd.read_csv('E:\\Workspace\\Moonlighting\\ASD Dignositc model\\AutoML\\Data\\Datacleaned_XY_ALL.csv')
# # 自变量（该数据集的前13项）
# X = Meta_scale.iloc[:, :-1].values
# # 因变量（该数据集的最后1项，即第14项）
# y = Meta_scale.iloc[:, -1].values
# f = open('Meta_scale.csv.txt','w')
# print("生成Meta_scale.csv为结局的AutoML,训练集与测试集比例由7:3", file=f)
#
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=.3, random_state=123)
#
# automl = autosklearn.classification.AutoSklearnClassifier(
#     time_left_for_this_task=240,
#     per_run_time_limit=30,
#     # tmp_folder="/tmp/",
# )
# automl.fit(X_train, y_train, dataset_name="df")
# print(automl.leaderboard(), file=f)
# pprint(automl.show_models(), indent=4)
# pprint(automl.sprint_statistics())
# predictions = automl.predict(X_test)
# print("Accuracy score:", sklearn.metrics.balanced_accuracy_score(y_test, predictions), file=f)
# f.close()

Micro_16s_scale = pd.read_csv('/mnt/e/Workspace/Moonlighting/ASD Dignositc model/AutoML/Data/Dataset_ASD_ID.csv')
# df = pd.read_csv('E:\\Workspace\\Moonlighting\\ASD Dignositc model\\AutoML\\Data\\Datacleaned_XY_ALL.csv')
# 自变量（该数据集的前13项）
X = Micro_16s_scale.iloc[:, :-1].values
# 因变量（该数据集的最后1项，即第14项）
y = Micro_16s_scale.iloc[:, -1].values
f = open('Dataset_ASD_ID.csv.txt','w')
print("Dataset_ASD_ID.csv为结局的AutoML,训练集与测试集比例由.3", file=f)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=.3, random_state=123)

automl = autosklearn.classification.AutoSklearnClassifier(
    time_left_for_this_task=1200,
    per_run_time_limit=30,
    # tmp_folder="/tmp/",
)
automl.fit(X_train, y_train, dataset_name="df")
print(automl.leaderboard(), file=f)
pprint(automl.show_models(), indent=4)
pprint(automl.sprint_statistics())
predictions = automl.predict(X_test)
print("Accuracy score:", sklearn.metrics.accuracy_score(y_test, predictions), file=f)
f.close()
