1. introduction
==============

와인 quality 맞추는 문제입니다.
와인을 나타내는 여러 특징들이 주어집니다.

<https://dacon.io/competitions/open/235610/overview/description>

2. Model
=======

i. Dataset
--------

데이터는 

label(quality)로써 0부터 9까지 정수가 주어지나 0부터 5까지는 0 6부터 9까지는 1로 예측합니다.
classification을 필요하기에 logistic regression모델을 사용하겠습니다.

quality까지 11개의 features, 1439개의 데이터가 실수와 정수타입으로 콤마를 구분자로하여 주어집니다.

* fixed acidity
고정산도를 나타냅니다.
* volatile acidity
휘발산도를 나타냅니다.
* citric acid       
시트르 산의 정도를 나타냅니다.

와인의 신맛을 결정합니다.

* residual sugar
잔류 설탕을 나타냅니다. 

와인의 단맛을 결정합니다.


* chlorides
염화물 함량을 나타냅니다. 와인의 짠맛과 신맛             
* free sulfur dioxide
유리 이산화 황
* total sulfur dioxide
총 이산화
* density               
* pH                   
* sulphates            
* alcohol              
* quality 

    
ii. preprocessing
------------------
input data는 1439

