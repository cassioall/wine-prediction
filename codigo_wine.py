#IMPORTAÇÕES
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

## CRIAÇÃO DE FUNÇÕES

# Ler os dados

def ler_dados(caminho):
    
    dados = pd.read_csv(caminho, header = 0, sep = ';')
    
    return dados

# Tratar 'alcohol'

def tratar_alcohol(dados):
    
    mask = dados['alcohol'].str.len() <= 5
    dados = dados.loc[mask]
    dados['alcohol'] = dados['alcohol'].astype('float64')
    
    return dados

# Separar dataframe pela variável 'type'

def separar_df(dados):
    
    mask_white = dados['type'] == 'White'
    mask_red = dados['type'] == 'Red'

    dados_white = dados.loc[mask_white]
    del dados_white['type']
    dados_red = dados.loc[mask_red]
    del dados_red['type']
    
    return dados_red, dados_white

# Transformar variável resposta em binária:

def resp_binario(dados):
    #todo vinho com qualidade acima de 6 será classificado como 'bom'.
    dados['quality'] = dados['quality'].astype(int)
    bins = (2, 6, 9)
    labels = ['bom', 'ruim']
    dados['quality'] = pd.cut(dados['quality'], bins = bins, labels = labels, include_lowest = False)

    label = LabelEncoder()
    dados['quality'] = label.fit_transform(dados['quality'])
    
    return dados

# Algoritmo SMOTEENN para balanceamento de classes:

def balancear_classes(X,Y):
    
    sme = SMOTEENN(random_state=42)
    X_res, y_res = sme.fit_resample(X,Y)

    return X_res, y_res

#Padronizar a base. Utilizar o RobustScaler por ele tratar bem a presença de outliers:

def padronizar(x_treino):
    
    rob = RobustScaler()
    rob.fit(x_treino)
    
    x_treino_transf = rob.fit_transform(x_treino)

    return x_treino_transf

def modelar(df):
    
    df_bin = resp_binario(df)
    y_treino = df_bin['quality']
    x_treino = df_bin.drop(['free sulfur dioxide','quality'], axis = 1)
    
    
    #objetos dos modelos
    rf = RandomForestClassifier(n_estimators=200)
    dtree = DecisionTreeClassifier(criterion = 'entropy',
                               min_weight_fraction_leaf = .06,
                               min_samples_leaf = .06)
    logreg = LogisticRegression(solver='lbfgs')
    gaus = GaussianNB()
    knn = KNeighborsClassifier()
    neur = MLPClassifier()
    svc = SVC(kernel = 'rbf', class_weight = 'balanced', probability = True)

    #lista com modelos e nomes
    mods = [dtree,
        logreg,
        gaus,
        rf,
        knn,
        neur,
        svc
        ]
    nome_modelo = {type(dtree): 'arvore_decisao',
               type(logreg): 'regressao_logistica',
               type(gaus): 'naive_bayes',
               type(rf): 'random_forest',
               type(knn): 'knn',
               type(neur): 'rede_neural',
               type(svc): 'svc'
              }
    media = 0
    std = 0
    kfold = KFold(n_splits=10)
    mod_final = None

    for mod in mods:
        x_treino_balanc, y_treino_balanc = balancear_classes(x_treino,y_treino)
        x_treino_padr = padronizar(x_treino)
        score_novo = cross_val_score(mod, x_treino_padr, y_treino, cv=kfold)
        media_novo = score_novo.mean()
        std_novo = score_novo.std()

        if media_novo > media:
            media = media_novo
            std = std_novo
            mod_final = mod
        else:
            pass

    print('Distribuição do Dataset original %s' % Counter(y_treino))
    print('Distribuição do Dataset reamostrado %s' % Counter(y_treino_balanc)) 
    print('\n')
    print('Melhor modelo: %s' % nome_modelo[type(mod_final)])
    print("Acurácia do modelo final: %0.2f (+/- %0.2f)" % (media, std * 2))
    
    return None
    
#LEIRUTA DO DATASET:

caminho = r'winequality.csv'
df = ler_dados(caminho)

#ANÁLISE EXPLORATÓRIA

#Plotando as 10 primeiras linhas do dataframe para visualização dos dados
df.head(10)

#Informação das variáveis
df.info()

#Variável 'alcohol'
df = tratar_alcohol(df)

#Criando dois dataframes para cada tipo de vinho
df_red, df_white = separar_df(df)

#Estatísticas descritivas de cada dataframe
df_white.describe()
df_red.describe()

# Verificando a média de cada independente para cada nível da dependente
df_red.groupby('quality').mean()
df_white.groupby('quality').mean()

#Matrizes de correlação
df_white.corr()
df_red.corr()

#VARIÁVEL DEPENDENTE

#Contagem de cada nível da variável
df_red['quality'].value_counts()
df_white['quality'].value_counts()

# Gráficos de barra da variável dependente
df_red['quality'].value_counts().plot.bar()
df_white['quality'].value_counts().plot.bar()

#Boxplot
sns.boxplot(df_red['quality'])
sns.boxplot(df_white['quality'])

#MODELAGEM

modelar(df_red)
modelar(df_white)
