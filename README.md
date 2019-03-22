# wine-prediction
Teste cognitivo.ai

######################## CONCEITO ########################


Objetivo: Criar um modelo para estimar a qualidade do vinho.

Variáveis Dependentes:

1. Tipo
2. Acidez fixa
3. Volatilidade da acidez
4. Ácido cítrico
5. Açúcar residual
6. Cloretos
7. Dióxido de enxofre livre
8. Dióxido de enxofre total
9. Densidade
10. pH
11. Sulfatos
12. Álcool

Variável Independente:

13. Quality

################### ESTRATÉGIA ADOTADA ###################

Primeiramente foi realizada uma análise descritiva dos dados, afim de identificar possíveis padrões entre as independentes (multicolinearidade),
presença de valores ausentes e/ou informações inválidas. Também realizar possíveis tratamentos dos dados. Nesta foram identificados alguns pontos:

  1 - A variável 'type' é categórica, as demais independentes são contínuas. Então é mais interessante separar o dataframe em dois,
  um para cada nível da variável.

  2 - A variável 'alcohol' aprensentou alguns registros não numéricos, estes foram excluídos do dataset para continuar com a modelagem.

  3 - A princípio dá para notar que as variáveis não estão na mesma escala.

  4 - Não há presença de valores nulos.

  5 - As variáveis 'free sulfur dioxide' e 'total sulfur dioxide' possuem correlação moderada (0.6 aprox)
   em ambos os dataframes. Após pesquisa, encontrei que o 'total sulfur dioxide' é "o conjunto das diferentes formas de 
   dióxido de enxofre presentes, no estado livre ou combinadas com os componentes do vinho".
   "Somando o dióxido de enxofre combinado ao dióxido de enxofre livre, obtém-se o dióxido de enxofre total."
   Isto quer dizer que, somando com o valor de correlação, existe a chance de resultar e problema
   de multicolinearidade. Portanto a variável 'free sulfur dioxide' foi excluída na modelagem.

  6 - Nenhuma variável independente tem correlação alta com a dependente.

  7 - A variável dependente é discreta. Portanto, três possíveis estratégias foram consideradas/testadas:
      * Transformar em uma variável binária, sendo que 1 significaria bom e 0 ruim. Classificando bom como 'quality' >= 7
      * Transformar a variável em três níveis: ruim (0 a 3), médio (4 a 6) e bom (7 a 10).
      * Tratar a variável como está e modelá-la.
    A estratégia escolhida foi de 'binarizar' a variável dependente, pois existem poucos registros menores do que 4 e maiores do que 6,
    o que atrapalharia um pouco a modelagem nos outros casos.
    
Depois de definir os dois datasets, assim como as variáveis dependentes e independente, a modelagem foi realizada da seguinte maneira:

- Mesmo depois de 'binarizar'a base de dados, as classes ainda ficaram desbalanceadas. Portanto foi aplicado o algorítmo de balanceamento de classes SMOTEENN, que utiliza técnicas combinadas de oversampling e undersampling.
- Para padronizar os dados foi utilizado o algorítmo RobustScaler, pois ele comporta bem com outliers, diferentemente do StandardScaler,
utilizando a mediana, ao invés da média, para realizar a padronização.
- Para selecionar o modelo final foi utilizada a técnica de validação cruzada dos dados, por fazer vários testes com o banco de dados,
assim diminuindo o risco de acontecer sobreajuste . O modelo que obteve a melhor acurácia média foi o selecionado.
- Os modelos testados foram: random forest, árvores de decisão, regressão logística, naive bayes gaussiano, KNN, redes neurais (MLP) e support vector machine.
 Não escolhendo um em específico para trabalhar os dados, mas deixando com que o resultado diga qual é o que melhor se ajusta.

####################### RESULTADOS #######################

No final da modelagem os modelos escolhidos e seus respectivos resultados foram:

Para os vinhos vermelhos:
  Modelo: regressão logística
  Acurácia: 0.88 (+/- 0.12)

Para os vinhos brancos:
  Modelo: random forest
  Acurácia: 0.81 (+/- 0.08)

Isto quer dizer que são bons modelos que acertam bem as predições.

