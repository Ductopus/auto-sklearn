from pprint import pprint
import pandas as pd
import sklearn.datasets
import sklearn.metrics

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score, accuracy_score
from sklearn.metrics import mean_squared_error

import autosklearn.classification

#
#
#
#
# def run_core(current_user):
#     # You Should change the below csv file path (It's different from Sherman between Quan)
#     # df = pd.read_csv('/mnt/e/Workspace/Moonlighting/ASD Dignositc model/AutoML/Data/Datacleaned_XY_ALL.csv')
#     df = pd.read_csv('/mnt/e/Development/AutoSklearn-Python/Data/Datacleaned_XY_ALL.csv')
#
#     # 自变量
#     X = df.iloc[:, :-1].values
#     # 因变量
#     y = df.iloc[:, -1].values
#
#     f = open('XY_ALL.txt', 'w')
#     # print(X)
#     # print(y)
#
#     # 关注从5：5到9：1的寻训练集与测试集的划分
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.5, random_state=123)
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
#
# def write_file(filename_key, data):
#     file = open('/mnt/e/Development/AutoSklearn-Python/' + filename_key + '.txt', 'a')
#     for i in range(len(data)):
#         # s = str(data[i]).replace('[','').replace(']','')#去除[],这两行按数据不同，可以选择
#         # s = s.replace("'",'').replace(',','') +'\n'   #去除单引号，逗号，每行末尾追加换行符
#         file.write(i)
#     file.close()
#     print("保存文件成功:" + filename_key)
#
#
# run_core('sherman')
