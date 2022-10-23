from pprint import pprint
import pandas as pd
import sklearn.datasets
import sklearn.metrics


from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score, accuracy_score
from sklearn.metrics import mean_squared_error

import autosklearn.classification
my_list = [0.5,0.4,0.3,0.2,0.1]
# # 生成襄阳的以全部为结局的AutoML
# df = pd.read_csv('/mnt/e/Workspace/Moonlighting/ASD Dignositc model/AutoML/Data/Datacleaned_XY_ALL.csv')
# # df = pd.read_csv('E:\\Workspace\\Moonlighting\\ASD Dignositc model\\AutoML\\Data\\Datacleaned_XY_ALL.csv')
# # 自变量（该数据集的前13项）
# X = df.iloc[:, :-1].values
# # 因变量（该数据集的最后1项，即第14项）
# y = df.iloc[:, -1].values
#
# f = open('XY_ALL.txt','w')
# print("生成襄阳的以全部为结局的AutoML,训练集与测试集比例由5：5-9：1", file=f)
#
# my_list = [0.5,0.4,0.3,0.2,0.1]
# for splitsize in my_list:
#     print(splitsize, file=f)
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=splitsize, random_state=123)
#
#     automl = autosklearn.classification.AutoSklearnClassifier(
#         time_left_for_this_task=120,
#         per_run_time_limit=30,
#         # tmp_folder="/tmp/",
#     )
#     automl.fit(X_train, y_train, dataset_name="df")
#     print(automl.leaderboard(), file=f)
#     pprint(automl.show_models(), indent=4)
#     pprint(automl.sprint_statistics())
#     predictions = automl.predict(X_test)
#     print("Accuracy score:", sklearn.metrics.balanced_accuracy_score(y_test, predictions), file=f)
#
# f.close()
# # 生成襄阳的以NBNA为结局的AutoML
# XY_NBNA = pd.read_csv('/mnt/e/Workspace/Moonlighting/ASD Dignositc model/AutoML/Data/Datacleaned_XY_NBNA.csv')
#
# # 自变量（该数据集的前13项）
# X = XY_NBNA.iloc[:, :-1].values
# # 因变量（该数据集的最后1项，即第14项）
# y = XY_NBNA.iloc[:, -1].values
# f = open('XY_NBNA.txt','w')
# print("生成襄阳的以NBNA为结局的AutoML,训练集与测试集比例由5：5-9：1", file=f)
# for splitsize in my_list:
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=splitsize, random_state=123)
#
#     automl = autosklearn.classification.AutoSklearnClassifier(
#         time_left_for_this_task=120,
#         per_run_time_limit=30,
#         # tmp_folder="/tmp/",
#     )
#     automl.fit(X_train, y_train, dataset_name="df")
#     print(automl.leaderboard(), file=f)
#     pprint(automl.show_models(), indent=4)
#     pprint(automl.sprint_statistics())
#     predictions = automl.predict(X_test)
#     print("Accuracy score:", sklearn.metrics.balanced_accuracy_score(y_test, predictions), file=f)
# f.close()
#
# # 生成襄阳的以Hear为结局的AutoML
# XY_Hear = pd.read_csv('/mnt/e/Workspace/Moonlighting/ASD Dignositc model/AutoML/Data/Datacleaned_XY_Hear.csv')
# # df = pd.read_csv('E:\\Workspace\\Moonlighting\\ASD Dignositc model\\AutoML\\Data\\Datacleaned_XY_ALL.csv')
# # 自变量（该数据集的前13项）
# X = XY_Hear.iloc[:, :-1].values
# # 因变量（该数据集的最后1项，即第14项）
# y = XY_Hear.iloc[:, -1].values
# f = open('XY_Hear.txt','w')
# print("生成襄阳的以Hear为结局的AutoML,训练集与测试集比例由5：5-9：1", file=f)
# for splitsize in my_list:
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=splitsize, random_state=123)
#
#     automl = autosklearn.classification.AutoSklearnClassifier(
#         time_left_for_this_task=120,
#         per_run_time_limit=30,
#         # tmp_folder="/tmp/",
#     )
#     automl.fit(X_train, y_train, dataset_name="df")
#     print(automl.leaderboard(), file=f)
#     pprint(automl.show_models(), indent=4)
#     pprint(automl.sprint_statistics())
#     predictions = automl.predict(X_test)
#     print("Accuracy score:", sklearn.metrics.balanced_accuracy_score(y_test, predictions), file=f)
# f.close()
#
# # 生成襄阳的以RP为结局的AutoML
# XY_RP = pd.read_csv('/mnt/e/Workspace/Moonlighting/ASD Dignositc model/AutoML/Data/Datacleaned_XY_RP.csv')
# # df = pd.read_csv('E:\\Workspace\\Moonlighting\\ASD Dignositc model\\AutoML\\Data\\Datacleaned_XY_ALL.csv')
# # 自变量（该数据集的前13项）
# X = XY_RP.iloc[:, :-1].values
# # 因变量（该数据集的最后1项，即第14项）
# y = XY_RP.iloc[:, -1].values
# f = open('XY_RP.txt','w')
# print("生成襄阳的以RP为结局的AutoML,训练集与测试集比例由5：5-9：1", file=f)
# for splitsize in my_list:
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=splitsize, random_state=123)
#
#     automl = autosklearn.classification.AutoSklearnClassifier(
#         time_left_for_this_task=120,
#         per_run_time_limit=30,
#         # tmp_folder="/tmp/",
#     )
#     automl.fit(X_train, y_train, dataset_name="df")
#     print(automl.leaderboard(), file=f)
#     pprint(automl.show_models(), indent=4)
#     pprint(automl.sprint_statistics())
#     predictions = automl.predict(X_test)
#     print("Accuracy score:", sklearn.metrics.balanced_accuracy_score(y_test, predictions), file=f)
# f.close()
#
# 生成襄阳的以Hear为结局的AutoML
SX_Hear = pd.read_csv('/mnt/e/Workspace/Moonlighting/ASD Dignositc model/AutoML/Data/Datacleaned_SX_Hear.csv')
# df = pd.read_csv('E:\\Workspace\\Moonlighting\\ASD Dignositc model\\AutoML\\Data\\Datacleaned_XY_ALL.csv')
# 自变量（该数据集的前13项）
X = SX_Hear.iloc[:, :-1].values
# 因变量（该数据集的最后1项，即第14项）
y = SX_Hear.iloc[:, -1].values
f = open('SX_Hear.txt','w')
print("生成SX的以Hear为结局的AutoML,训练集与测试集比例由5：5-9：1", file=f)
for splitsize in my_list:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=splitsize, random_state=123)

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
f.close()
# 生成襄阳的以RP为结局的AutoML
