import pandas as pd
import xgboost as xgb
import copy
import itertools
from sklearn import ensemble
from sklearn import metrics
from sklearn import preprocessing


def mean_target_encoding(data):

    df = copy.deepcopy(data)

    num_cols = [
        "fnlwgt",
        "age",
        "capital.gain",
        "capital.loss",
        "hours.per.week"
    ]

    target_mapping = {
        '<=50K': 0,
        '>50K': 1
    }

    df.loc[:,'income'] = df.income.map(target_mapping)
     
    features = [
    f for f in df.columns if f not in ("kfold", "income") and f not in num_cols
    ]

    for col in features:
        if col not in num_cols:
            df.loc[:,col] = df[col].astype(str).fillna('NONE')
            
            lbl = preprocessing.LabelEncoder()
            lbl.fit(df[col])
            df.loc[:,col] = lbl.transform(df[col])

    #a list to store 5 validation dataframes    
    encoded_df = []

    for fold in range(5):

        df_train = df[df.kfold != fold].reset_index(drop = True)
        df_valid = df[df.kfold == fold].reset_index(drop = True)

        for column in features:

            mapping_dict = dict(df_train.groupby(column)['income'].mean())

            df_valid.loc[:,column + 'enc'] = df_valid[column].map(mapping_dict)
        encoded_df.append(df_valid)

    encoded_df = pd.concat(encoded_df,axis = 0)

    return encoded_df


def run(df,fold):

    df_train = df[df.kfold != fold].reset_index(drop = True)

    df_valid = df[df.kfold == fold].reset_index(drop = True)

    features = [
    f for f in df.columns if f not in ("kfold", "income")
    ]

    x_train = df_train[features].values

    x_valid = df_valid[features].values

    model = xgb.XGBClassifier(
        n_jobs = -1,
        max_depth = 7
    )

    model.fit(x_train,df_train.income.values)

    valid_preds = model.predict_proba(x_valid)[:,1]

    auc = metrics.roc_auc_score(df_valid.income.values,valid_preds)

    print(f"fold  = {fold}, auc = {auc}")


if __name__ == "__main__":

    df = pd.read_csv('./data/adult_folds.csv')

    df = mean_target_encoding(df)


    for fold_ in range(5):
        run(df,fold_)