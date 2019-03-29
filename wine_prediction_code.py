import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import statsmodels.api as sm
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.metrics import roc_curve, auc
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from imblearn.combine import SMOTEENN
from collections import Counter

## CREATION OF FUNCTIONS

# Read the data

def read_data(path):
    
    data = pd.read_csv(path, header = 0, sep = ';')
    
    return data

#  'alcohol'

def treat_alcohol(data):
    
    mask = data['alcohol'].str.len() <= 5
    data = data.loc[mask]
    data['alcohol'] = data['alcohol'].astype('float64')
    
    return data

# Separate dataframe with variable 'type'

def separate_df(data):
    
    mask_white = data['type'] == 'White'
    mask_red = data['type'] == 'Red'

    data_white = data.loc[mask_white]
    del data_white['type']
    data_red = data.loc[mask_red]
    del data_red['type']
    
    return data_red, data_white

# Transform variable response into binary:

def resp_binary(data):
    # any wine with quality above 6 will be classified as 'good'.
    data['quality'] = data['quality'].astype(int)
    bins = (2, 6, 9)
    labels = ['good', 'bad']
    data['quality'] = pd.cut(data['quality'], bins = bins, labels = labels, include_lowest = False)

    label = LabelEncoder()
    data['quality'] = label.fit_transform(data['quality'])
    
    return data

# SMOTEENN algorithm for class balancing:

def balance_classes(X,Y):
    
    sme = SMOTEENN(random_state=42)
    X_res, y_res = sme.fit_resample(X,Y)

    return X_res, y_res

# Standardize the base. Use RobustScaler because it treats the presence of outliers well:

def padronize(x_train):
    
    rob = RobustScaler()
    rob.fit(x_train)
    
    x_train_transf = rob.fit_transform(x_train)

    return x_train_transf

def model(df):
    
    df_bin = resp_binary(df)
    y_train = df_bin['quality']
    x_train = df_bin.drop(['free sulfur dioxide','quality'], axis = 1)
    
    
    # model objects
    rf = RandomForestClassifier(n_estimators=200)
    dtree = DecisionTreeClassifier(criterion = 'entropy',
                               min_weight_fraction_leaf = .06,
                               min_samples_leaf = .06)
    logreg = LogisticRegression(solver='lbfgs')
    gaus = GaussianNB()
    knn = KNeighborsClassifier()
    neur = MLPClassifier()
    svc = SVC(kernel = 'rbf', class_weight = 'balanced', probability = True)

    # list with models and names
    mods = [dtree,
        logreg,
        gaus,
        rf,
        knn,
        neur,
        svc
        ]
    name_model = {type(dtree): 'decision_tree',
               type(logreg): 'logistic_regression',
               type(gaus): 'naive_bayes',
               type(rf): 'random_forest',
               type(knn): 'knn',
               type(neur): 'neural_nets',
               type(svc): 'svc'
              }
    mean = 0
    std = 0
    kfold = KFold(n_splits=10)
    mod_final = None

    for mod in mods:
        x_train_balanc, y_train_balanc = balance_classes(x_train,y_train)
        x_train_padr = padronize(x_train)
        score_new = cross_val_score(mod, x_train_padr, y_train, cv=kfold)
        mean_new = score_new.mean()
        std_new = score_new.std()

        if mean_new > mean:
            mean = mean_new
            std = std_new
            mod_final = mod
        else:
            pass

    print('Original dataset distribution %s' % Counter(y_train))
    print('Redesigned dataset distribution %s' % Counter(y_train_balanc)) 
    print('\n')
    print('Best model: %s' % name_model[type(mod_final)])
    print("Score of the final model: %0.2f (+/- %0.2f)" % (mean, std * 2))
    
    return None
    
# READING THE DATASET:

path = r'winequality.csv'
df = read_data(path)

# EXPLORATORY ANALYSIS

# Plotting the first 10 lines of the dataframe for data visualization
df.head(10)

# Information of the dataset 
df.info()

#Variable 'alcohol'
df = treat_alcohol(df)

# Creating two dataframes for each type of wine
df_red, df_white = separate_df(df)

# Descriptive statistics for each dataframe
df_white.describe()
df_red.describe()

# Checking the average of each independent for each level of the dependent
df_red.groupby('quality').mean()
df_white.groupby('quality').mean()

# Correlation matrices
df_white.corr()
df_red.corr()

# DEPENDENT VARIABLE

# Count of each level of the variable
df_red['quality'].value_counts()
df_white['quality'].value_counts()

# Bar graphs of the dependent variable
df_red['quality'].value_counts().plot.bar()
df_white['quality'].value_counts().plot.bar()

# Boxplot
sns.boxplot(df_red['quality'])
sns.boxplot(df_white['quality'])

# MODELING

model(df_red)
model(df_white)
