---
title: "캐글 타이타닉2 - 모델 튜닝을 통한 점수 높이기"
excerpt: "scikit-learn을 통해 생성한 생존자 예측 모델 튜닝 방법을 정리한다."
toc: true
toc_sticky: true
header:
  teaser: /assets/images/3280_1552550473.jpg

categories:
  - 캐글
tags:
  - 타이타닉

use_math: true

last_modified_at: 2020-05-23
---

지난 포스팅에서 타이타닉의 생존자 예측 모델을 생성해보았다. 데이터 전처리에는 꽤 많은 공을 들였지만 학습 과정에서 모델 파라미터를 하나도 조절하지 않은 점이 아쉬웠다. 이번에는 모델을 조금 더 정교하게 만드는 방법들을 적용해보고자 한다.

# 1. feature scaling

머신러닝 알고리즘은 입력 숫자들의 스케일이 많이 다르면 잘 작동하지 않는다. 타이타닉 문제의 특성들을 다시 살펴보자.

|Variable|	Definition|	Key|
|---|---|---|
|survival|	생존여부|	0 = No, 1 = Yes|
|pclass|	사회-경제적 지위 |	1 = 1st, 2 = 2nd, 3 = 3rd|
|sex|	성별	|
|Age|	나이	|
|sibsp|	타이타닉호에 탑승한 형제-자매 수	|
|parch|	타이타닉호에 탑승한 부모-자녀 수	|
|ticket|	티켓 번호	|
|fare|	탑승 요금	|
|cabin|	방 번호	|
|embarked|	탑승 지역(항구 위치)|	C = Cherbourg, Q = Queenstown, S = Southampton|

특성들 중 연속 변수에 해당하는 것은 Age, Fare이고 실제로 이 둘이 가장 큰 범위를 갖는다. 저번 포스팅에서는 이 둘을 구간과 개수를 기준으로 그룹화하여 범주형 데이터로 만들어 스케일을 줄였다. 이번에는 다른 방법을 써보려 한다.  

우선 저번 포스팅의 코드를 압축하여 `Age` 전까지의 전처리를 진행한 뒤에 시작하자.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 데이터 불러오기
train = pd.read_csv('./train.csv')
test = pd.read_csv('./test.csv')

train_test_data = [train, test]

# Sex 전처리
sex_mapping = {"male":0, "female":1}
for dataset in train_test_data:
    dataset['Sex'] = dataset['Sex'].map(sex_mapping)

# SibSp & Parch 전처리
for dataset in train_test_data:
    # 가족수 = 형제자매 + 부모님 + 자녀 + 본인
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
    dataset['IsAlone'] = 1
    
    # 가족수 > 1이면 동승자 있음
    dataset.loc[dataset['FamilySize'] > 1, 'IsAlone'] = 0
    
# Embarked 전처리
for dataset in train_test_data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')

embarked_mapping = {'S':0, 'C':1, 'Q':2}
for dataset in train_test_data:
    dataset['Embarked'] = dataset['Embarked'].map(embarked_mapping)
    
# Name 전처리
for dataset in train_test_data:
    dataset['Title'] = dataset['Name'].str.extract('([\w]+)\.', expand=False)

for dataset in train_test_data:
    dataset['Title'] = dataset['Title'].apply(lambda x: 0 if x=="Mr" else 1 if x=="Miss" else 2 if x=="Mrs" else 3 if x=="Master" else 4)
```

## 1.1 Age : min-max 스케일링

각 데이터에서 최소값을 뺀 뒤 범위(max - min)로 나누어 모든 데이터를 0~1 사이에 오도록 만드는 방법이다. 상한과 하한이 확실히 정해지는 장점이 있으나 이상치에 민감한 단점이 있다. 비교적 정규분포에 가까운 Age에 `min-max 스케일링`을 적용하자.  
  
직접 코드를 작성하여 스케일링을 할 수도 있으나 사이킷런에서 이에 해당하는 `MinMaxScaler` 변환기를 제공한다. 사용 방법은 다음과 같다.

```python
# 결측치 제거
for dataset in train_test_data:
    dataset['Age'].fillna(dataset.groupby("Title")["Age"].transform("median"), inplace=True)
    
# min-max 스케일링
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
for dataset in train_test_data:
    array = dataset['Age'].values.reshape(-1,1) # 2D array로 변환
    scaler.fit(array) # 스케일링에 필요한 값(최소값, range 등) 계산
    dataset['AgeScale'] = pd.Series(scaler.transform(array).reshape(-1)) # 스케일링 후 series로 추가
    
train.head()
```

![](https://github.com/romanticq/romanticq.github.io/blob/master/assets/images/kaggle-titanic2/1.1_min_max_head.png?raw=true)
{: .notice--warning}

우선 `MinMaxScaler`는 2D array에 대해서 작동한다. 그러나 `dataset['Age']`은 Dataframe의 series로써 1D array이다. 따라서 이를 numpy의 `reshape(-1,1)` 메서드를 통해 2D array로 바꾼다. 참고로 `reshape` 메서드는 pandas의 series에는 사용할 수 없으므로 `values` 메서드를 통해 series를 numpy array로 바꿔주어야 한다.  
  
`scaler.fit(array)` 는 스케일링에 필요한 값들을 계산해서 저장한다. 실제 스케일링은 `scaler.transform(array)`으로 해야한다. 변환이 끝난 결과물은 아직 `2D numpy array` 이므로 `reshape(-1)`을 통해 1D numpy array로 바꾼 뒤 `Series` 메서드를 통해 series로 바꾼 다음 train과 test dataframe의 새로운 column으로 추가시킨다.

## 1.2 Fare : 표준화(standardization)

정규분포의 Z변수$\left(X-mu \over \sigma \right)$로 바꾸는 작업이다. min-max 스케일링과 마찬가지로 사이킷런에서 변환기를 제공한다. `Fare` 특성은 분포의 왜곡이 심하므로 표준화가 적절해 보인다. 사용법은 min-max 스케일링과 동일하다.

```python
# 결측치 제거
for dataset in train_test_data:
    dataset["Fare"].fillna(dataset.groupby("Pclass")["Fare"].transform("median"), inplace=True)
    
# 표준화
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
for dataset in train_test_data:
    array = dataset['Fare'].values.reshape(-1,1)
    scaler.fit(array)
    dataset['FareScale'] = pd.Series(scaler.transform(array).reshape(-1))
    
train.head()
```

![](https://github.com/romanticq/romanticq.github.io/blob/master/assets/images/kaggle-titanic2/1.2_std_head.png?raw=true)
{: .notice--warning}


# 2. 상관계수 : 필요없는 특성 제거

상관계수를 통해 target과 관련이 없는 특성은 제거하도록 하자. 불필요한 특성이 많을 경우 모델이 과대적합될 위험이 있으므로 이 작업은 중요하다. 데이터셋이 크지 않은 경우 모든 특성 간의 표준 상관계수(피어슨 r)를 `corr()` 메서드를 통해 쉽게 계산할 수 있다.

```python
# 전처리가 끝난 특성들 제거
drop_column = ['Name', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin']

for dataset in train_test_data:
    dataset = dataset.drop(drop_column, axis=1, inplace=True)

# 상관계수 계산
corr_matrix = train.corr()
corr_matrix["Survived"].sort_values(ascending=False)
```

Survived       1.000000  
Sex            0.543351  
Title          0.414088  
FareScale      0.257307  
Embarked       0.106811  
FamilySize     0.016639  
PassengerId   -0.005007  
AgeScale      -0.078698  
IsAlone       -0.203367  
Pclass        -0.338481  
Name: Survived, dtype: float64
{: .notice--warning}

`FamilySize`의 경우 상관계수가 거의 0에 가까우므로 `Survived`와는 큰 관련이 없는 것으로 볼 수 있다. 따라서 삭제하도록 한다.

```python
drop_column = ['FamilySize']

for dataset in train_test_data:
    dataset = dataset.drop(drop_column, axis=1, inplace=True)
    
train.head()
```

![](https://github.com/romanticq/romanticq.github.io/blob/master/assets/images/kaggle-titanic2/2_corr_head.png?raw=true)
{: .notice--warning}

***

# 3. k-겹 교차 검증(k-fold cross-validation) : 결정트리

지금까지 특성들에 대한 정제방법에 대해 다루었다면 이제부터는 모델을 튜닝해보도록 하자. 사이킷런의 k-fold 교차검증은 훈련 데이터를 폴드(fold)라 불리는 n개의 서브셋으로 무작위 분할한다. 그 다음 매번 다른 폴드를 선택해 평가에 사용하고 나머지 n-1개의 폴드는 훈련에 사용한다. 즉 모델을 n번 훈련하고 평가하여 n개의 평가 점수가 담긴 배열을 출력한다.  

저번 포스팅에서 가장 점수가 좋았던 결정트리 모델에 k-fold 교차 검증을 적용하자.

```python
# 훈련을 위한 train, target 분할
drop_column2 = ['PassengerId', 'Survived']
train_data = train.drop(drop_column2, axis=1)
target = train['Survived']

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score # 교차 검증 패키지
from sklearn.metrics import mean_squared_error

clf = DecisionTreeClassifier()
clf.fit(train_data, target) # 모델 훈련
prediction = clf.predict(train_data) # 예측
tree_mse = mean_squared_error(target, prediction) # 오차 계산
tree_rmse = np.sqrt(tree_mse) 
print("훈련 데이터 점수 :", tree_rmse)

# 교차 검증
scores = cross_val_score(clf, train_data, target,
                        scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)

# 점수 표시 함수 작성
def display_scores(scores):
    print("교차검증 점수:", scores)
    print("평균:", scores.mean())
    print("표준편차:", scores.std())

display_scores(tree_rmse_scores)
```

훈련 데이터 점수 : 0.138129235668458  
교차검증 점수: [0.52704628 0.43704832 0.56089816 0.46204236 0.41053541 0.47404546 
 0.38218767 0.51929079 0.44971901 0.43704832]  
평균: 0.46598617890957217  
표준편차: 0.05262594705758865  
{: .notice--warning}

점수는 평균제곱오차에 해당하므로 0에 가까울수록 좋다. 훈련데이터 전체에 대해서는 점수가 매우 높은 반면 교차 검증에 대해서는 그 결과를 잘 반영하지 못한다. 즉 모델이 과대적합 되었다는 뜻이다. 실제로 캐글에 제출했을 때에도 스케일링 전보다 점수가 더 낮아졌다.

***

# 4. 모델 파라미터 조작 : 규제

규제는 과대적합된 모델이 다른 데이터에 대해서도 일반성을 갖도록 하는 방법중 하나이다. 훈련 세트 전체에 대한 설명력은 낮아지지만 교차검증에 대한 점수는 높일 수 있도록 모델을 튜닝해야한다.

## 4.1 max_depth
  
결정트리에 대해 규제를 가하는 대표적인 방법은 가지치기이다. 즉 트리의 깊이를 조절하는 방법이다. 사이킷런에서는 `max_depth` 매개변수로 이를 조절할 수 있다. 트리의 깊이에 상한을 정하여 과도하게 복잡한 트리가 만들어지는 것을 방지한다. 결정트리 모델에 아무런 매개변수도 입력하지 않았을 경우(기본 매개변수) 규제가 없는 모델이 생성된다.

```python
score_list=[]
kfold_score_list=[]
for i in range(1,11):
    clf = DecisionTreeClassifier(max_depth = i)
    clf.fit(train_data, target)
    prediction = clf.predict(train_data)
    tree_mse = mean_squared_error(target, prediction)
    tree_rmse = np.sqrt(tree_mse)
    scores = cross_val_score(clf, train_data, target,
                        scoring="neg_mean_squared_error", cv=10)
    tree_rmse_scores = np.sqrt(-scores)
    score_list.append(tree_rmse)
    kfold_score_list.append(tree_rmse_scores.mean())
```

``` python
score_data = {'훈련 세트 점수' : score_list, '교차 검증 점수' : kfold_score_list}
score_data_frame = pd.DataFrame(score_data, index=range(1,11))
score_data_frame
```

![](https://github.com/romanticq/romanticq.github.io/blob/master/assets/images/kaggle-titanic2/4.1_max_depth.png?raw=true)
{: .notice--warning}

최대 가지 깊이가 높아지면서 규제가 풀릴수록 훈련 세트에 대해서는 높은 성능을 보인다. 교차 검증 점수는 3~7 정도에서 높다. 이들 중 훈련세트와의 오차는 `max_depth=3`일때 가장 적지만 교차 검증 점수는 `max_depth=4`일때 가장 낮으므로 해당 파라미터는 `max_depth=4`를 택한다.

## 4.2 min_samples_leaf

리프노드는 더이상 자식노드를 가지지 않는 불순도 0의 노드이다. 사이킷런의 `min_samples_leaf` 파라미터는 리프 노드가 가져야할 최소 샘플 수를 말한다. 결정트리에서 `min_` 으로 시작하는 매개변수를 증가시키거나 `max_`로 시작하는 매개변수를 감소시킬수록 모델의 규제는 커지게 된다. `max_depth=4`로 고정하고 `min_samples_leaf`에 따른 k-fold 교차검증을 시도해보자.

```python
score_list=[]
kfold_score_list=[]
for i in range(1,11):
    clf = DecisionTreeClassifier(max_depth = 4, min_samples_leaf = i)
    clf.fit(train_data, target)
    prediction = clf.predict(train_data)
    tree_mse = mean_squared_error(target, prediction)
    tree_rmse = np.sqrt(tree_mse)
    scores = cross_val_score(clf, train_data, target,
                        scoring="neg_mean_squared_error", cv=10)
    tree_rmse_scores = np.sqrt(-scores)
    score_list.append(tree_rmse)
    kfold_score_list.append(tree_rmse_scores.mean())
```

```python
score_data = {'훈련 세트 점수' : score_list, '교차 검증 점수' : kfold_score_list}
score_data_frame = pd.DataFrame(score_data, index=range(1,11))
score_data_frame
```

![](https://github.com/romanticq/romanticq.github.io/blob/master/assets/images/kaggle-titanic2/4.2_min_samples_leaf.png?raw=true)
{: .notice--warning}

`min_samples_leaf=6` 일때 교차 검증 점수가 가장 높으므로 해당 파라미터를 채택한다.  
  
채택한 파라미터로 모델을 훈련하여 결과를 저장한다.  
  

결정트리에 어떤 파라미터들이 더 있는지 확인 하고싶다면 다음 링크를 들어가보자.

[scikit-learn : DecisionTreeClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html?highlight=decision#sklearn.tree.DecisionTreeClassifier)
{: .notice--info}

```python
# 모델 훈련
clf = DecisionTreeClassifier(max_depth=4, min_samples_leaf=6)
clf.fit(train_data, target)
test_data = test.drop("PassengerId", axis=1)
predict = clf.predict(test_data)

# 예측 결과 저장
submission = pd.DataFrame({
    'PassengerId' : test['PassengerId'],
    'Survived' : predict})

submission.to_csv('submission_tree_reg.csv', index=False)
```

캐글에 제출한 결과이다.

![](https://github.com/romanticq/romanticq.github.io/blob/master/assets/images/kaggle-titanic2/submission_score.png?raw=true)

전보다 꽤 많이 올랐다!

![](https://github.com/romanticq/romanticq.github.io/blob/master/assets/images/kaggle-titanic2/submission_rank.png?raw=true)

랭크도 상위 29%에서 9%로 상승.

***

# 참고

핸즈온 머신러닝 2판 (오렐리앙 제롱 지음. 박해선 옮김)