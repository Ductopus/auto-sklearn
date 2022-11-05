import pandas as pd
from pycaret import classification
# f = open('XY.txt','w')
# print("NBNA为结局的多模型,训练集.8", file=f)
# data_ALL = pd.read_csv('/mnt/e/Workspace/Moonlighting/ASD Dignositc model/AutoML/Data/Datacleaned_XY_ALL_train.csv')
# data_Hear = pd.read_csv('/mnt/e/Workspace/Moonlighting/ASD Dignositc model/AutoML/Data/Datacleaned_XY_Hear_train.csv')
data_NBNA = pd.read_csv('/mnt/e/Workspace/Moonlighting/ASD Dignositc model/AutoML/Data/Datacleaned_XY_NBNA_train.csv')
# data_ALL.head()
# data_Hear.head()
data_NBNA.head()

classification_setup = classification.setup(data= data_NBNA, target='Outcome_NBAB')#以NBNA为结局进行训练
classification_dt = classification.create_model('dt')
classification.compare_models()
# print(classification_dt)
classification.plot_model(classification_dt, plot = 'auc')
# classification_xgb = classification.create_model('xgboost')
# top10 = classification.compare_models(n_select = 3)
# top3 = classification.compare_models(n_select = 3) # Top 3
# Modelsbest = classification.compare_models(sort = 'AUC') # Sorted by AUC
# best_specific = classification.compare_models(whitelist = ['lr','knn','dt']) # only these three models
# comparedbest_specific = classification.compare_models(blacklist = ['catboost', 'xgboost']) # compare all models except for categorical boost and XGBoost.

# print(top10, file=f)
# print(top3, file=f)
# print(Modelsbest, file=f)
# print(best_specific, file=f)
# f.close()