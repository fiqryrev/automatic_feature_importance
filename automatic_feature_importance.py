import pandas as pd
import numpy as np
import time
from prettytable import PrettyTable

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import xgboost as xg
import catboost as cb

class FindSomethingImportant():
    """
    Function to find and create an ensemble feature importance score within one or more classifier models. 
    Suggestion, you may insert the optimized hyperparameter of each model to this function.
    
    Particularly, in this class, you can:
    1. Fit(X_train,y_train)
       Train all inserted models
    2. summarize()
       Give the feature importance information along with the overall best feature table
       
    Changelog:
    v1.0
    - First initialization
    - You should have prettytable, lightgbm, xgboost, and catboost installed in your notebook first
    - This function works well to all scikitlearn classifier, lightgbm, xgboost, and catboost
    - Not support for the automatic hyperparameter tuning
    - Not support for all deep learning models
    - Due to limited generalization, I used these abbrevations within my code. So, if you want to use it, please 
      set these libraries as what I named below:
        - Numpy as np
        - Pandas as pd
        - Catboost as cb
    - Require pretty table, seaborn, and matplotlib to be activated in the python environment
    
    """
    def __init__(self,random_state=123,limit=10,verbose=False,classifier=[RandomForestClassifier()],\
                show_plot = False, show_table = True):
        """
        Create the class, set the parameters.
        
        random_state = Set your seed [Default = 123]
        limit = Set your maximum features shown in the feature importance [Default = 10]
        verbose = Show the track information along the process [Default = False]
        classifier = Use only the classifier, otherwise automatically set to RandomForestClassifier. You can add
                     some models in a list. [Default = RandomForestClassifier()]
        show_plot = Show the plot on the summarize result [Default = False]
        show_table = Show the importance table on the summarize result [Default = True]
        
        Restrictions:
        
        
        """

        self.verbose = verbose
        self.random_state = random_state
        self.show_plot = show_plot
        self.show_table = show_table
        self.__train_table = False
        self.importance = {}
        self.__fi_score = {}
        self.__importance_sign = {}
		if limit > len(col_name):
            limit = len(col_name)
        if limit < 0 or limit >100:
            print("Limit is limited between 1 - 100. Automatically set to 10")
            self.limit = 10
        else:
            self.limit = limit
        try:
            cache_logic = len(classifier)
            clf_list = []
            for i in range(len(classifier)):
                if "Classifier" not in str(classifier[i].__class__):
                    print(f"{classifier[i]} is not classifier class, automatically remove from the list.")
                else:
                    clf_list.append(classifier[i])
            if len(clf_list) == 0:
                print("Since there is no eligible classifier, automatically set to Random Forest Classifier")
                self.classifier = [RandomForestClassifier(random_state=self.random_state)]
            else:
                self.classifier = clf_list
        except:
            if "Classifier" not in str(classifier.__class__):
                print(f"{classifier.__class__.__name__} is not classifier class, automatically set to Random Forest Classifier")
                self.classifier = [RandomForestClassifier(random_state=self.random_state)]
            else:
                self.classifier = [classifier]
        self.__classifier_name_list= []
        for i in range(len(self.classifier)):
            self.__classifier_name_list.append(self.classifier[i].__class__.__name__)
        print("=================================================================")
        print("Show Importance Function v1.0")
        print(f"Trying to find feature importances from {self.__classifier_name_list}")
        print("=================================================================")
        
    def __importance_table_func(self,classifier_at):
        """
        
        Create the table of feature importance, sort descently
        
        """
        class_name = classifier_at.__class__.__name__
        index = np.argsort(self.importance[class_name])
        t = PrettyTable(["Category name","Feature Importances","Score"])
        for i in range(self.limit):
            t.add_row([self.__colnames[index][len(self.importance[class_name])-self.limit:][self.limit-i-1],
                      self.importance[class_name][index][len(self.importance[class_name])-self.limit:][self.limit-i-1],
                      self.limit-i])
        print(t)
        return t
    
    def fit(self,X,y,colnames):
        """
        Train the models you've input in this class. 
        
        X = Your X_train
        y = Your y_train
        colnames = The feature names in a list. Column names. If it is empty, the feature names will be set to a
                 sequence of number from 1 to N (maximum length of your feature)
        
        """
        try:
            cache_test = len(colnames)
            self.__colnames = np.array(colnames)
        except:
            print("Column name is empty, automatically set to sequence number from 1 to N")
            self.__colnames = np.array([x+1 for x in range(np.array(X).shape[1])])
        self.X = (X if isinstance(X,np.ndarray) else np.array(X))
        self.y = (y if isinstance(y,np.ndarray) else np.array(y))
        
        if self.__train_table == False:
            start_time_all = time.time()
            for i in range(len(self.classifier)):
                class_name = self.classifier[i].__class__.__name__
                if self.verbose:
                    print(f"Training Process on {class_name}")
                start_time = time.time()
                self.classifier[i].fit(self.X,self.y)
                elapsed_time = time.time() - start_time
                if self.verbose:
                    print("=================================================================")
                    print(f"Training Complete")
                    print(f"Elapsed: {time.strftime('%H:%M:%S', time.gmtime(elapsed_time))}")
                    print("=================================================================")
                try:  
                    try:
                        self.importance[class_name] = self.classifier[i].feature_importances_
                    except:
                        self.importance[class_name] = np.array(self.classifier[i].get_feature_importance(\
                                                                                 cb.Pool(self.X,label=self.y)))
                    calc_index = np.argsort(self.importance[class_name])
                    calc_fi_score = 0
                    col_name_loop = []
                    for j in range(self.limit):
                        calc_fi_score += self.importance[class_name][calc_index][len(self.importance[class_name]\
                                                                                 )-self.limit:][self.limit-j-1]
                        col_name_loop.append(self.__colnames[calc_index][len(self.importance[class_name])-self.limit:][\
                                      self.limit-j-1])
                    self.__importance_sign[class_name] = col_name_loop
                    self.__fi_score[class_name] = calc_fi_score
                except AttributeError:
                    print(f"{class_name} has no attribute Feature Importance. Stored nothing.")
                    self.importance[class_name] = np.array([0 for x in range(len(col_name))])
                    self.__importance_sign[class_name] = col_name
                    self.__fi_score[class_name] = 0
                
            clear_output()
            elapsed_time_all = time.time() - start_time_all
            print("=================================================================")
            print(f"Training All Classifiers Completed")
            print(f"Elapsed: {time.strftime('%H:%M:%S', time.gmtime(elapsed_time_all))}")
            print("=================================================================")
            self.__train_table = True
        else:
            print("Data have already been trained. Try summarize()")

    def summarize(self):
        """
        
        Summarize the feature importance by table and or bar chart. Also, return a data frame, as a
        result from majority voting of feature importance from all classifiers.
        
        """
        all_class = []
        for i in range(len(self.classifier)):
            class_name = self.classifier[i].__class__.__name__
            calc_index = np.argsort(self.importance[class_name])
            all_class.append(self.__importance_sign[class_name])
            if self.show_table:
                print(f"Top {self.limit} of {class_name} Feature Importances:")
                self.__importance_table_func(self.classifier[i])          
            print(f"FI Cummulative Score of {class_name}: {self.__fi_score[class_name]}")
            print("")
        top_features = {}
        for i in all_class:
            for index,value in enumerate(i):
                score = (len(i)-index)
                if value not in top_features:
                    top_features[value] = 1*score
                else:
                    top_features[value] += score
        total_feature_include = self.limit*len(self.classifier)
        best_features = sorted(top_features.items(), key=lambda x: x[1],reverse=True)
        self.most_important_features = pd.DataFrame(best_features[:self.limit])
        self.most_important_features.columns = ["Feature","Score"]
        self.most_important_features["Percentage"] = self.most_important_features["Score"].map(lambda x: 100*round(x/total_feature_include,2))
        if self.show_plot:
            self.most_important_features.set_index("Feature")["Percentage"].plot(kind="bar",figsize=(14,7),\
            title="Top Feature Overall")
        return self.most_important_features

def __init__():
    clf_A = RandomForestClassifier()
    clf_B = xg.XGBClassifier()
    clf_C = lgb.LGBMClassifier(silent=True)
    clf_D = cb.CatBoostClassifier(verbose=False)
    classifier_stack = [clf_A,clf_B,clf_C,clf_D]
    find_importance_all = FindSomethingImportant(random_state=123,limit=10,verbose=True,classifier=classifier_stack,show_plot = True, show_table = True)
    find_importance_all.fit(X_train,y_train,col_name)
    find_importance_all.summarize()