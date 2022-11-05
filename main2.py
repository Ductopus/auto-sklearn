import pandas as pd
# Importing dataset
from pycaret import classification
from pycaret.classification import compare_models, setup

data_NBNA = pd.read_csv('/mnt/e/Workspace/Moonlighting/ASD Dignositc model/AutoML/Data/diabetes.csv')
clf1 = setup(data = data_NBNA, target = 'Class variable',fold=5,session_id = 123)
# 选取AUC最优5个模型
best_top5 = compare_models(sort = 'AUC',n_select = 5)
