from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif,f_regression
from sklearn.feature_selection import mutual_info_classif,mutual_info_regression
from sklearn.feature_selection import SelectKBest,SelectPercentile


class UnivariateFeatureSelection:
    def __init__(self,n_features,problem_type,scoring):

        if problem_type == 'classification':
            valid_scoring = {
                'f_classif':f_classif,
                'chi2':chi2,
                'mutual_info_classif':mutual_info_classif
            }
        else:
            valid_scoring = {
                'f_regression':f_regression,
                'mutual_info_regression':mutual_info_regression
            }

        if scoring not in valid_scoring:
            raise Exception('invalid scoring function')

        if isinstance(n_features,int):
            self.selection = SelectKBest(valid_scoring[scoring],k=n_features)
        elif isinstance(n_features,float):
            self.selection = SelectPercentile(valid_scoring[scoring],percentile=int(n_features)*100)
        else:
            raise Exception('invalid type of features')

    def fit(self,X,y):
        return self.selection.fit(X,y)

    def transform(self,X):
        return self.selection.transform(X)

    def fit_transform(Oself,X,y):
        return self.selection.fit_transform(X,y)
    
