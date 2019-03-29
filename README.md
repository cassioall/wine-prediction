# wine-prediction

######################## CONCEPT ########################


Aim: Create a model to estimate the quality of a bottle of wine.

Dependent Variables:

1. type
2. fixed acidity
3. volatile acidity
4. citric acid
5. residual sugar
6. chlorides
7. free sulfur dioxide
8. total sulfur dioxide
9. density
10. pH
11. sulphates
12. alcohol

Independent Variable:

13. quality

################### STRATEGY ADOPTED ###################


First, a descriptive analysis of the data was carried out in order to identify possible patterns among the independent (multicollinearity),
presence of missing values and/or invalid information. Also perform possible data processing. In this were identified some points:

  1 - The variable 'type' is categorical, the other independent variables are continuous. So it's more interesting to separate the dataframe in two, one for each level of the variable.

  2 - The variable 'alcohol' presented some non-numeric registers, these were excluded from the dataset to continue with the modeling.
 
  3 - At first it should be noted that the variables are not on the same scale.

  4 - No null values are present.

  5 - The variables 'free sulfur dioxide' and 'total sulfur dioxide' had a strong correlation (0.75 aprox)
   in the dataframe. After research, I found that 'total sulfur dioxide' is' the set of different forms of
   sulfur dioxide present in the free state or combined with the components of the wine ".
   "Adding the combined sulfur dioxide to the free sulfur dioxide gives the total sulfur dioxide."
   This means that, in addition to the correlation value, there is a chance of a problem
   of multicollinearity. Therefore, the variable 'free sulfur dioxide' was excluded in the modeling.

  6 - No independent variable has a high correlation with the dependent variable.

  7 - The dependent variable is discrete. Therefore, three possible strategies were considered / tested:
      
      * Transform into a binary variable, where 1 would mean good and 0 bad. Classifying good as 'quality'> = 7
      
      * Transform the variable into three levels: bad (0 to 3), medium (4 to 6) and good (7 to 10).
      
      * Treat the variable as it is and model it.
      
   
  The chosen strategy was to 'binarize' the dependent variable, since there are few records smaller than 4 and larger than 6,
   which would disrupt the modeling in other cases.
    
After defining the two datasets, as well as the dependent and independent variables, the modeling was performed as follows:

- Even after 'binarizing' the database, the classes were still unbalanced. Therefore, the SMOTEENN class balancing algorithm, which uses combined oversampling and undersampling techniques, was applied.
- To standardize the data was used the algorithm RobustScaler, because it behaves well with outliers, unlike StandardScaler,
using the median, instead of the mean, to perform the standardization.
- To select the final model, the technique of cross-validation of the data was used, by doing several tests with the database,
thus reducing the risk of over-adjustment. The model that obtained the best average accuracy was the one selected.
- The models tested were: random forest, árvores de decisão, regressão logística, naive bayes gaussiano, KNN, redes neurais (MLP) and support vector machine. Not choosing a specific one to work with the data, but letting the result tell you which one fits best.

####################### RESULTS #######################

At the end of the modeling the chosen models and their respective results were:

For the red wines:
  Model: regressão logística
  Score: 0.88 (+/- 0.12)

For the white wines:
  Model: random forest
  Score: 0.81 (+/- 0.08)

This means that they are good models that correctly predict.

