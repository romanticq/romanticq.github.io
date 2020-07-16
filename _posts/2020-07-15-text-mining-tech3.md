---
title: "[텍스트 마이닝] 텍스트 마이닝 기법3"
excerpt: "텍스트 마이닝 기법 중 하나인 연관어 분석에 대해 정리한다."
toc: true
toc_sticky: true
header:
  teaser: /assets/images/3280_1552550473.jpg

categories:
  - 머신러닝
tags:
  - 텍스트 마이닝

use_math: true

last_modified_at: 2020-07-15
---

이 포스팅의 jupyter notebook 파일과 예제에 사용되는 데이터셋은 아래 링크에서 확인할 수 있습니다.

[텍스트 마이닝 기법3 주피터 노트북](https://github.com/romanticq/TextMining/blob/master/04.%20TextMiningTechnique3.ipynb)  
<br>
[영화 리뷰 데이터](https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
{: .notice--info}

## 4.5 연관어 분석

### 개요
연관어는 두 단어가 문맥 내에서 서로 얼마나 연관되어 있는지를 말한다. 두 단어 사이의 관련 정도는 다음과 같은 방법들을 통해 측정한다.

1) 두 단어가 같은 문서에서 함께 출현하는 횟수 세기  
<br>
2) 통계적 방법으로 두 단어간의 유사도 산출  
<br>
3) 딥러닝(word2vec) 유사도 이용
{: .notice--info}

연관어 추출을 위해서는 다음 두 가지를 선행해야한다.

1) 대상어 선정 : 대상어는 연구의 주된 대상으로 삼는 단어이다. 대상어는 분석 목적에 따라 결정된다. 연구자가 관심을 갖는 단어 또는 고빈도 출현 단어로 선정할 수도 있다.  
<br>
2) 문맥 결정 : 연관어를 어떠한 문맥 대상으로 추출할지 결정한다. 연관어 분석에서 문맥이란 글의 범위를 말한다. 하나의 문맥은 문서, 문단, 문장, 문장 내 단어 등이 될 수 있다. 하나의 문서를 한 문맥으로 보는 게 가장 보편적이다.
{: .notice--info}

연관어 분석은 다른 텍스트 마이닝 기법들보다 시각화에 대한 의존도가 높다. 단어 간의 연관도를 살펴야하기 때문이다. 보통 파이썬 이외의 시각화 툴을 사용하며 종류는 다음과 같은 것들이 있다.

1) Gephi  
<br>
2) Centriufuge  
<br>
3) Commetrix  
{: .notice--info}

***

### 4.5.1 동시 출현 기반 연관어 분석

동시출현 기반 연관어 분석은 대상어와 다른 단어들이 같은 문맥 내에서 동시에 출현한 횟수를 세는 방법이다. 동시 출현 빈도가 높을 수록 연관성이 강하다고 가정한다. 이 가정하에 동시출현 횟수에 대한 임계값을 정하여 임계값을 넘는 단어 간의 페어만 남기고 나머지는 필터링한다.

***

**< 영화 리뷰 데이터 분석 >** 

필요한 패키지를 임포트한 뒤 리뷰 데이터를 불러온다.


```python
# 필요한 패키지 임포트
import pandas as pd
import glob
from afinn import Afinn
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
import numpy as np
import matplotlib.pyplot as plt
```


```python
# 리뷰 데이터 로드
review = pd.read_csv('E:\\text-mining\\IMDB\IMDB-Dataset.csv', engine="python")
review.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>review</th>
      <th>sentiment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>One of the other reviewers has mentioned that ...</td>
      <td>positive</td>
    </tr>
    <tr>
      <th>1</th>
      <td>A wonderful little production. &lt;br /&gt;&lt;br /&gt;The...</td>
      <td>positive</td>
    </tr>
    <tr>
      <th>2</th>
      <td>I thought this was a wonderful way to spend ti...</td>
      <td>positive</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Basically there's a family where a little boy ...</td>
      <td>negative</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Petter Mattei's "Love in the Time of Money" is...</td>
      <td>positive</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Probably my all-time favorite movie, a story o...</td>
      <td>positive</td>
    </tr>
    <tr>
      <th>6</th>
      <td>I sure would like to see a resurrection of a u...</td>
      <td>positive</td>
    </tr>
    <tr>
      <th>7</th>
      <td>This show was an amazing, fresh &amp; innovative i...</td>
      <td>negative</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Encouraged by the positive comments about this...</td>
      <td>negative</td>
    </tr>
    <tr>
      <th>9</th>
      <td>If you like original gut wrenching laughter yo...</td>
      <td>positive</td>
    </tr>
  </tbody>
</table>
</div>



리뷰 데이터들 중 긍정 리뷰만을 필터링 한다.


```python
# 긍정 리뷰 필터링
is_pos_review = review['sentiment'] == 'positive'
pos_review = review[is_pos_review]['review'][0:100] # 100개만 추출
pos_review.reset_index(inplace=True, drop=True) # 인덱스 초기화
print(pos_review)
type(pos_review)
```

    0     One of the other reviewers has mentioned that ...
    1     A wonderful little production. <br /><br />The...
    2     I thought this was a wonderful way to spend ti...
    3     Petter Mattei's "Love in the Time of Money" is...
    4     Probably my all-time favorite movie, a story o...
                                ...                        
    95    I think this movie has got it all. It has real...
    96    Howard (Kevin Kline) teaches English at the hi...
    97    We usually think of the British as the experts...
    98    One of Starewicz's longest and strangest short...
    99    Nice character development in a pretty cool mi...
    Name: review, Length: 100, dtype: object
    




    pandas.core.series.Series




```python
tokenizer = RegexpTokenizer('[\w]+')
stop_words = stopwords.words('english')

count = {} # 동시출현 빈도가 저장될 dict
for line in pos_review:
    words = line.lower() # 각 리뷰를 소문자로 변환
    tokens = tokenizer.tokenize(words) # 각 리뷰를 토큰화한 뒤 리스트에 저장
    stopped_tokens = [i for i in list(set(tokens)) if not i in stop_words+["br"]]
    stopped_tokens2 = [i for i in stopped_tokens if len(i)>1]
    for i,a in enumerate(stopped_tokens2):
        for b in stopped_tokens2[i+1:]:
            if a>b:
                count[b,a] = count.get((b,a),0) + 1
            else:
                count[a,b] = count.get((a,b),0) + 1
```

위 코드는 빈 딕셔너리 `count`에 동시출현 페어를 적재하는 과정이다.  
  
첫 번째 for문에서 `tokens` 변수에는 각 리뷰에 대해 토큰화된 단어들을 리스트로 묶어 저장한다. 첫 번째 리뷰를 예로 들면 다음과 같다. 리뷰의 길이가 길기 때문에 앞에서부터 10개 단어만 살펴보자.


```python
tokenizer.tokenize(pos_review[0].lower())[0:10]
```




    ['one',
     'of',
     'the',
     'other',
     'reviewers',
     'has',
     'mentioned',
     'that',
     'after',
     'watching']



다음으로 `set`을 통해 `tokens`의 중복되는 단어들을 제거한 뒤 다시 리스트로 만든다. 그 다음에 불용어와 "br"이라는 문자를 제거 후 `stopped_tokens`에 저장한다.  
  
동시출현 페어를 적재하는 아이디어는 조합의 계산이다. `set`을 통해 중복되는 토큰들을 제거했으므로 combination의 개념을 적용할 수 있다. 즉 $n \choose 2$를 계산하는 과정을 코드로 구현한다. 
  
구현 과정은 중복 for문을 이용한다. 예를 들어, 50개의 단어로 이루어진 리뷰 100개가 있다고 하자. 알고리즘은 다음과 같다.

1) 첫 리뷰, 첫 번째 단어에 대해 나머지 49개의 단어들과 튜플을 구성한다. 각 튜플은 딕셔너리의 key에 해당한다.  
<br>
2) 튜플의 성분은 오름차순으로 구성한다. (one, of)와 (of, one)이 다르게 취급되는 것을 막기 위함이다.  
<br>
3) 두 번째 단어에 대해 나머지 48개의 단어들과 튜플을 구성하여 위 과정을 반복한다.  
<br>
4) 마지막 단어까지 위 과정을 반복한다.
<br>
5) 마지막 리뷰까지 2~4번 과정을 반복하며 같은 튜플이 나올때마다 value를 1씩 늘려 counting한다.  
{: .notice--info}
  
**< 중복 for문 코드 설명 >**  
* `for i,a enumerate(stopped_tokens2)` : `i,a`는 리스트(`stopped_tokens2`)의 인덱스(i)와 그에 대응하는 값(a)을 의미한다.
* `if a>b:` : a,b는 각각 하나의 토큰들이다. 즉 문자열의 비교이다. 파이썬에서 문자열 비교는 가장 앞 글자의 ascii 코드 값을 비교한다. 가장 앞 글자의 ascii 코드 값이 같다면 그 다음 글자의 값을 비교한다. ascii 코드 값이 클수록 순서가 뒤로 밀린다. 즉 사전식 배열에서 뒤에 위치한다. 이 조건문은 튜플을 오름차순으로 배열하기 위해 사용한다.
* `count[b,a] = count.get((b,a),0) + 1` : count 딕셔너리에 key : `(b,a)`, value : `count.get((b,a),0) + 1` 인 성분을 추가한다.  
  
**< 딕셔너리 용법 : get >**  
1) `count.get((b,a))` : count에서 key `(b,a)`에 대응하는 value를 얻는다. 딕셔너리에 `(b,a)`라는 key가 없을 경우 None을 반환한다.  
<br>
2) `count.get((b,a),0)` : get은 최대 2개의 인수를 받을 수 있다. 두 번째 인수는 기본값에 해당한다. key 리스트에 `(b,a)`가 있을 경우 value를 반환하며, 없을 경우 기본값에 해당하는 0을 반환한다.
{: .notice--info}


***

위 과정을 통해 만든 딕셔너리를 dataframe으로 만들면 다음과 같다.


```python
df = pd.DataFrame.from_dict(count, orient='index')

print(df.info())
df
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 694983 entries, ('far', 'many') to ('pound', 'pretty')
    Data columns (total 1 columns):
     #   Column  Non-Null Count   Dtype
    ---  ------  --------------   -----
     0   0       694983 non-null  int64
    dtypes: int64(1)
    memory usage: 10.6+ MB
    None
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>(far, many)</th>
      <td>5</td>
    </tr>
    <tr>
      <th>(cells, far)</th>
      <td>2</td>
    </tr>
    <tr>
      <th>(away, far)</th>
      <td>3</td>
    </tr>
    <tr>
      <th>(far, guards)</th>
      <td>1</td>
    </tr>
    <tr>
      <th>(far, order)</th>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>(acting, spider)</th>
      <td>1</td>
    </tr>
    <tr>
      <th>(acting, pound)</th>
      <td>1</td>
    </tr>
    <tr>
      <th>(pretty, spider)</th>
      <td>1</td>
    </tr>
    <tr>
      <th>(pound, spider)</th>
      <td>1</td>
    </tr>
    <tr>
      <th>(pound, pretty)</th>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>694983 rows × 1 columns</p>
</div>



dataframe은 한 줄짜리 column으로 구성된다. index는 딕셔너리의 key에 해당하는 튜플들이고 entry는 딕셔너리의 value이다.  
  
다음은 적재한 동시출현 페어(term1, term2)의 성분(freq)을 각 series로 구성하여 새로운 dataframe을 만든다.


```python
list1=[]
for i in range(len(df)):
    list1.append([df.index[i][0], df.index[i][1], df[0][i]])

df2 = pd.DataFrame(list1, columns=['term1','term2','freq'])
df3 = df2.sort_values(by=['freq'], ascending=False) # freq 기준으로 내림차순 정렬
df3 = df3.reset_index(drop=True)
df3.head(20)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>term1</th>
      <th>term2</th>
      <th>freq</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>film</td>
      <td>one</td>
      <td>31</td>
    </tr>
    <tr>
      <th>1</th>
      <td>like</td>
      <td>movie</td>
      <td>24</td>
    </tr>
    <tr>
      <th>2</th>
      <td>movie</td>
      <td>one</td>
      <td>24</td>
    </tr>
    <tr>
      <th>3</th>
      <td>film</td>
      <td>like</td>
      <td>23</td>
    </tr>
    <tr>
      <th>4</th>
      <td>film</td>
      <td>story</td>
      <td>22</td>
    </tr>
    <tr>
      <th>5</th>
      <td>movie</td>
      <td>time</td>
      <td>22</td>
    </tr>
    <tr>
      <th>6</th>
      <td>movie</td>
      <td>see</td>
      <td>22</td>
    </tr>
    <tr>
      <th>7</th>
      <td>one</td>
      <td>time</td>
      <td>21</td>
    </tr>
    <tr>
      <th>8</th>
      <td>film</td>
      <td>movie</td>
      <td>20</td>
    </tr>
    <tr>
      <th>9</th>
      <td>film</td>
      <td>way</td>
      <td>20</td>
    </tr>
    <tr>
      <th>10</th>
      <td>movie</td>
      <td>really</td>
      <td>20</td>
    </tr>
    <tr>
      <th>11</th>
      <td>like</td>
      <td>one</td>
      <td>20</td>
    </tr>
    <tr>
      <th>12</th>
      <td>good</td>
      <td>movie</td>
      <td>20</td>
    </tr>
    <tr>
      <th>13</th>
      <td>film</td>
      <td>see</td>
      <td>20</td>
    </tr>
    <tr>
      <th>14</th>
      <td>one</td>
      <td>see</td>
      <td>20</td>
    </tr>
    <tr>
      <th>15</th>
      <td>film</td>
      <td>really</td>
      <td>19</td>
    </tr>
    <tr>
      <th>16</th>
      <td>film</td>
      <td>much</td>
      <td>19</td>
    </tr>
    <tr>
      <th>17</th>
      <td>film</td>
      <td>time</td>
      <td>19</td>
    </tr>
    <tr>
      <th>18</th>
      <td>film</td>
      <td>well</td>
      <td>19</td>
    </tr>
    <tr>
      <th>19</th>
      <td>best</td>
      <td>one</td>
      <td>18</td>
    </tr>
  </tbody>
</table>
</div>



부정리뷰에 대해서도 같은 과정을 적용한다.


```python
# 부정 리뷰 필터링
is_neg_review = review['sentiment'] == 'negative'
neg_review = review[is_neg_review]['review'][0:100] # 100개만 추출
neg_review.reset_index(inplace=True, drop=True) # 인덱스 초기화

tokenizer = RegexpTokenizer('[\w]+')
stop_words = stopwords.words('english')

count = {} # 동시출현 빈도가 저장될 dict
for line in neg_review:
    words = line.lower() # 각 리뷰를 소문자로 변환
    tokens = tokenizer.tokenize(words) # 각 리뷰를 토큰화한 뒤 리스트에 저장
    stopped_tokens = [i for i in list(set(tokens)) if not i in stop_words+["br"]]
    stopped_tokens2 = [i for i in stopped_tokens if len(i)>1]
    for i,a in enumerate(stopped_tokens2):
        for b in stopped_tokens2[i+1:]:
            if a>b:
                count[b,a] = count.get((b,a),0) + 1
            else:
                count[a,b] = count.get((a,b),0) + 1
```


```python
df = pd.DataFrame.from_dict(count, orient='index')

list1=[]
for i in range(len(df)):
    list1.append([df.index[i][0], df.index[i][1], df[0][i]])

df2 = pd.DataFrame(list1, columns=['term1','term2','freq'])
df3 = df2.sort_values(by=['freq'], ascending=False) # freq 기준으로 내림차순 정렬
df3 = df3.reset_index(drop=True)
df3.head(20)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>term1</th>
      <th>term2</th>
      <th>freq</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>like</td>
      <td>movie</td>
      <td>47</td>
    </tr>
    <tr>
      <th>1</th>
      <td>film</td>
      <td>movie</td>
      <td>37</td>
    </tr>
    <tr>
      <th>2</th>
      <td>movie</td>
      <td>one</td>
      <td>34</td>
    </tr>
    <tr>
      <th>3</th>
      <td>film</td>
      <td>like</td>
      <td>33</td>
    </tr>
    <tr>
      <th>4</th>
      <td>good</td>
      <td>movie</td>
      <td>32</td>
    </tr>
    <tr>
      <th>5</th>
      <td>film</td>
      <td>one</td>
      <td>31</td>
    </tr>
    <tr>
      <th>6</th>
      <td>like</td>
      <td>one</td>
      <td>31</td>
    </tr>
    <tr>
      <th>7</th>
      <td>even</td>
      <td>movie</td>
      <td>29</td>
    </tr>
    <tr>
      <th>8</th>
      <td>good</td>
      <td>like</td>
      <td>29</td>
    </tr>
    <tr>
      <th>9</th>
      <td>movie</td>
      <td>would</td>
      <td>28</td>
    </tr>
    <tr>
      <th>10</th>
      <td>even</td>
      <td>like</td>
      <td>27</td>
    </tr>
    <tr>
      <th>11</th>
      <td>movie</td>
      <td>see</td>
      <td>27</td>
    </tr>
    <tr>
      <th>12</th>
      <td>film</td>
      <td>would</td>
      <td>27</td>
    </tr>
    <tr>
      <th>13</th>
      <td>film</td>
      <td>see</td>
      <td>26</td>
    </tr>
    <tr>
      <th>14</th>
      <td>could</td>
      <td>movie</td>
      <td>26</td>
    </tr>
    <tr>
      <th>15</th>
      <td>movie</td>
      <td>movies</td>
      <td>25</td>
    </tr>
    <tr>
      <th>16</th>
      <td>film</td>
      <td>good</td>
      <td>25</td>
    </tr>
    <tr>
      <th>17</th>
      <td>movie</td>
      <td>time</td>
      <td>25</td>
    </tr>
    <tr>
      <th>18</th>
      <td>one</td>
      <td>see</td>
      <td>24</td>
    </tr>
    <tr>
      <th>19</th>
      <td>bad</td>
      <td>movie</td>
      <td>24</td>
    </tr>
  </tbody>
</table>
</div>



***

### 4.5.2 통계적 가중치 기반 연관어 분석

두 단어가 연관된 정도를 '유사도'를 통해 나타내는 방법이다. 텍스트 마이닝에서 가장 많이 사용하는 유사도는 `cosine similarity`이다. 그 외에도 `jaccard similarity`, `overlap similarity` 등이 있다.  
  
유사도는 정량적인 통계값이다. 따라서 텍스트에 대해 유사도를 계산하려면 단어마다 가중치를 할당해야 한다. 즉 단어마다 적절한 숫자를 부여한다. 가중치로써 자주 이용하는 것은 출현빈도, TF-IDF 등이 있다.


$$\mathrm{cosine\ similarity}\ S_{ij} = {A \cdot B \over ||A||\ ||B||} = {\sum_{k} x_{ik} \times x_{jk} \over \sqrt{\sum_{k} (x_{ik})^2} \times \sqrt{\sum_{k} (x_{jk})^2}}$$

$$\mathrm{jaccard\ similarity}\ S_{ij} = {\sum_k \mathrm{min}(x_{ik}, x_{jk}) \over \sum_k \mathrm{max}(x_{ik}, x_{jk})}$$

$$\mathrm{overlap\ similarity}\ S_{ij} = {\sum_k \mathrm{min}(x_{ik}, x_{jk}) \over \mathrm{min}(\sum_k x_{ik}, \sum_k x_{jk})}$$

x는 출현빈도를 의미하며 i, j는 단어 인덱스, k는 문서 인덱스를 의미한다.  
  
영화 리뷰 데이터에 해당 개념을 적용해보자.

***

**< 영화 리뷰 데이터 분석 >**


```python
# 필요한 패키지 임포트
import pandas as pd
import glob
from afinn import Afinn
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
```


```python
# 리뷰 데이터 로드
review = pd.read_csv('E:\\text-mining\\IMDB\IMDB-Dataset.csv', engine="python")

# 긍정 리뷰 필터링
is_pos_review = review['sentiment'] == 'positive'
pos_review = review[is_pos_review]['review'][0:100] # 100개만 추출
pos_review.reset_index(inplace=True, drop=True) # 인덱스 초기화
```


```python
stop_words = stopwords.words('english')
vec = TfidfVectorizer(stop_words=stop_words)
vector_pos_review = vec.fit_transform(pos_review)
vector_pos_review
```




    <100x4995 sparse matrix of type '<class 'numpy.float64'>'
    	with 10713 stored elements in Compressed Sparse Row format>



TFIDF 가중치를 할당한 결과는 희소행렬에 저장된다. 이를 일반행렬로 바꾼다.


```python
A = vector_pos_review.toarray()
pd.DataFrame(A)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>4985</th>
      <th>4986</th>
      <th>4987</th>
      <th>4988</th>
      <th>4989</th>
      <th>4990</th>
      <th>4991</th>
      <th>4992</th>
      <th>4993</th>
      <th>4994</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.146447</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>95</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>96</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>97</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.085182</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>98</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.072555</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>99</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>100 rows × 4995 columns</p>
</div>



각 리뷰를 하나의 문서로 보면 총 100개 문서, 4995개 단어에 대해 가중치가 할당된 결과이다. 이대로 유사도를 구하게 되면 문서 간의 유사도를 구하게 되므로 transpose를 통해 단어-문서 매트릭스로 바꾼다.


```python
A=A.transpose()
pd.DataFrame(A)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>90</th>
      <th>91</th>
      <th>92</th>
      <th>93</th>
      <th>94</th>
      <th>95</th>
      <th>96</th>
      <th>97</th>
      <th>98</th>
      <th>99</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.074655</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.074307</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>4990</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4991</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4992</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.072555</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4993</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.085182</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4994</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>4995 rows × 100 columns</p>
</div>




```python
A_sparse = sparse.csr_matrix(A) # A를 다시 희소행렬로 변환
similarities_sparse = cosine_similarity(A_sparse, dense_output=False)
list(similarities_sparse.todok().items())[35000:35010]
```




    [((1098, 133), 0.3484457223054042),
     ((1099, 133), 0.4726439154636967),
     ((1104, 133), 0.3382677891921974),
     ((1133, 133), 0.2738023270540216),
     ((1134, 133), 0.17092631488618495),
     ((1154, 133), 0.4726439154636967),
     ((1173, 133), 0.4726439154636967),
     ((1184, 133), 0.4726439154636967),
     ((1209, 133), 0.1523951677290328),
     ((1211, 133), 0.8812534988158323)]



`todok()` 메서드는 행렬을 딕셔너리로 변환하는 기능을 한다. `items()` 는 딕셔너리의 key와 value를 튜플로 묶어 dict_items 객체로 반환한다. 이를 다시 리스트로 바꾸기 위해 `list()` 메서드를 이용한다.  
  
출력 결과 단어의 페어는 인덱스 형태로 주어진다. 인덱스 (1098, 133)에 해당하는 각 단어는 `get_features_names()` 메서드로 볼 수 있다.


```python
print(vec.get_feature_names()[1098])
print(vec.get_feature_names()[133])
```

    dead
    affected
    


```python
vec.get_feature_names()[100:105]
```




    ['active', 'activities', 'actor', 'actors', 'actress']



이제 단어 페어간의 유사도가 높은 순으로 정렬해서 dataframe으로 나타내보자.


```python
df = pd.DataFrame(list(similarities_sparse.todok().items()), columns=['words', 'weight'])
df2 = df.sort_values(by=['weight'], ascending=False)
df2 = df2.reset_index(drop=True)
df3 = df2.loc[np.round(df2['weight']) < 1]
df3 = df3.reset_index(drop=True)

df3.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>words</th>
      <th>weight</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>(616, 1511)</td>
      <td>0.500000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>(1511, 616)</td>
      <td>0.500000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>(2929, 2082)</td>
      <td>0.499995</td>
    </tr>
    <tr>
      <th>3</th>
      <td>(2082, 2929)</td>
      <td>0.499995</td>
    </tr>
    <tr>
      <th>4</th>
      <td>(3483, 69)</td>
      <td>0.499987</td>
    </tr>
    <tr>
      <th>5</th>
      <td>(4701, 3483)</td>
      <td>0.499987</td>
    </tr>
    <tr>
      <th>6</th>
      <td>(3483, 1886)</td>
      <td>0.499987</td>
    </tr>
    <tr>
      <th>7</th>
      <td>(2033, 3483)</td>
      <td>0.499987</td>
    </tr>
    <tr>
      <th>8</th>
      <td>(4680, 3483)</td>
      <td>0.499987</td>
    </tr>
    <tr>
      <th>9</th>
      <td>(3483, 4987)</td>
      <td>0.499987</td>
    </tr>
  </tbody>
</table>
</div>



자기 자신끼리의 페어는 유사도가 무조건 1이므로 1 미만의 단어만 추출하도록 한다.

***

### 4.5.3 word2vec 기반 연관어 분석

word2vec은 두 가지의 가정을 기반으로 한다.  
  
1) 단어의 의미는 그 단어 주변 단어의 분포로 이해된다.  
<br>
2) 단어의 의미는 단어 박터 안에 인코딩 될 수 있다.  
{: .notice--info}

앞서 소개한 두 방법과 word2vec의 차이는 가중치의 계산 방식이다. 문맥내 출현 빈도가 같은 단어는 같은 연관도(혹은 유사도)를 갖는다. 그러나 word2vec에서는 문맥에 출현 횟수가 같다고 해서 가중치가 할당되지 않는다. 첫 번째 가정에 의해 단어의 위치, 순서에 따라서도 가중치가 달라지게 된다. 가중치의 산출 후에는 앞서 소개한 유사도 공식에 따라 두 단어 간의 유사도를 산출한다.  
  
word2vec은 다음과 같이 크게 두 가지로 나뉜다.  
  
1) `CBOW` : 주변 단어로 중심 단어를 예측하도록 모델 구축  
<br>
2) `Skip-gram` : 중심 단어로 주변 단어를 예측하도록 모델 구축  
{: .notice--info}

`Skip-gram`이 `window size`(주변에 포함할 단어 수)에 따라 더 많은 반복학습을 하게 된다. 때문에 CBOW보다 더 정확한 예측을 하는 경우가 많아 널리 쓰인다.

***

**< 영화 리뷰 데이터 분석 >**


```python
# 필요한 패키지 임포트
import pandas as pd
import glob
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
import numpy as np
from gensim.models.word2vec import Word2Vec
```


```python
# 리뷰 데이터 로드
review = pd.read_csv('E:\\text-mining\\IMDB\IMDB-Dataset.csv', engine="python")

# 긍정 리뷰 필터링
is_pos_review = review['sentiment'] == 'positive'
pos_review = review[is_pos_review]['review'][0:100] # 100개만 추출
pos_review.reset_index(inplace=True, drop=True) # 인덱스 초기화
```


```python
tokenizer = RegexpTokenizer('[\w]+')
stop_words = stopwords.words('english')

text = [] # word2vec을 적용할 list
for line in pos_review:
    words = line.lower() # 각 리뷰를 소문자로 변환
    tokens = tokenizer.tokenize(words) # 각 리뷰를 토큰화한 뒤 리스트에 저장
    stopped_tokens = [i for i in list(set(tokens)) if not i in stop_words+["br"]]
    stopped_tokens2 = [i for i in stopped_tokens if len(i)>1]
    text.append(stopped_tokens2)
```

text에는 각 리뷰들의 불용어가 제거된 2글자 이상의 토큰들이 저장된다. 하나의 리뷰마다 하나의 리스트를 구성한다. 따라서 text는 100개의 리스트들이 저장된 리스트가 된다.


```python
model = Word2Vec(text, sg=1, window=2, min_count=3)
model.init_sims(replace=True)
model.wv.similarity('film', 'movie')
```




    0.8244013



* `sg=1`은 `Skip-gram`을 적용하기 위한 파라미터이다.  
* `window=2`는 중심 단어로부터 좌우 2개의 단어까지 학습에 적용한다는 의미이다. 
* `min_count=3`은 전체 문서에서 최소 3번 이상 출현한 단어들을 대상으로 학습을 진행한다는 의미이다.  
  
output은 'film'과 'movie'간의 유사도를 나타낸다.


```python
model.wv.most_similar("good", topn=5)
```




    [('movie', 0.80223149061203),
     ('really', 0.7817724943161011),
     ('film', 0.7807669639587402),
     ('time', 0.7668334245681763),
     ('like', 0.7640729546546936)]



good과 가장 유사한 단어 5개를 나타낸 결과이다. `gensim` 패키지는 기본적으로 consine similarity를 적용한다.

***

### 4.5.4 중심성(centrality) 계수

4.5.1~4.5.3은 단어 페어 간의 연관도를 계산하는 방법을 다뤘다. 그러나 전체 단어 군에서 개별 단어의 상대적 중요성은 알 수 없다. 단어별 중심성 계수를 구하면 단어별 상대적 중요성을 파악할 수 있다.  
  
중심성이란 그래프이론에서 쓰이는 용어이다. 단어 간의 연관도를 링크로 표현하면 하나의 그래프가 형성되므로 그래프 이론과 연관도는 쉽게 접목될 수 있다. 중심성 계수는 다음과 같은 종류가 있다. 그래프의 노드는 단어를 의미한다.  
  
* **연결 중심성(degree centrality)** : 하나의 노드가 직접적으로 몇 개의 노드와 연결되어 있는지 측정한다. 즉 개별 노드의 edge 개수를 파악한다. 거리가 1인 링크만을 고려하는 것과 같다. 따라서 국지적인 범위에서 노드의 영향력을 파악한다.  
<br>
* **근접 중심성(closeness centrality)** : 직접연결(거리=1), 간접연결(거리>1)을 모두 포함하여 중심성을 측정한다. 특정 단어와 연속적인 링크로 연결되는 모든 단어와의 거리에 따른 평균적인 연관도를 측정한다. 글로벌적인 중요도 판단이 가능하다.  
<br>
* **매개 중심성(betweenness centrality)** : 노드간 링크를 타고 건너갈 때 핵심적으로 통과해야만 하는 노드를 찾을 때 용이하다. 매개 중심성이 크면 네트워크 내 의사소통 흐름에 영향을 줄 소지가 많다. 텍스트 마이닝에서 자주 활용되진 않는다.  
<br>
* **고유벡터 중심성(eigenvector centrality)** : 각 노드마다 중요성을 부과할 때 해당 노드와 연결된 노드들의 중심성을 고려한다. 높은 고유벡터 중심성을 가진 노드는 높은 점수를 가진 많은 노드와 연결되어 있음을 의미한다.
  
  
파이썬에서는 `networkx` 라이브러리로 중심성 계수를 쉽게 계산할 수 있다.  

***

**< 영화 리뷰 데이터 분석 >**

단어 페어간 연관도는 동시출현 빈도로 구하였다. 긍정 리뷰에 대해서 먼저 수행한다.


```python
# 필요한 패키지 임포트
import pandas as pd
import glob
from afinn import Afinn
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
import numpy as np
import matplotlib.pyplot as plt
```


```python
# 리뷰 데이터 로드
review = pd.read_csv('E:\\text-mining\\IMDB\IMDB-Dataset.csv', engine="python")

# 긍정 리뷰 필터링
is_pos_review = review['sentiment'] == 'positive'
pos_review = review[is_pos_review]['review'][0:100] # 100개만 추출
pos_review.reset_index(inplace=True, drop=True) # 인덱스 초기화
```


```python
tokenizer = RegexpTokenizer('[\w]+')
stop_words = stopwords.words('english')

count = {} # 동시출현 빈도가 저장될 dict
for line in pos_review:
    words = line.lower() # 각 리뷰를 소문자로 변환
    tokens = tokenizer.tokenize(words) # 각 리뷰를 토큰화한 뒤 리스트에 저장
    stopped_tokens = [i for i in list(set(tokens)) if not i in stop_words+["br"]]
    stopped_tokens2 = [i for i in stopped_tokens if len(i)>1]
    for i,a in enumerate(stopped_tokens2):
        for b in stopped_tokens2[i+1:]:
            if a>b:
                count[b,a] = count.get((b,a),0) + 1
            else:
                count[a,b] = count.get((a,b),0) + 1
                
df = pd.DataFrame.from_dict(count, orient='index')

list1=[]
for i in range(len(df)):
    list1.append([df.index[i][0], df.index[i][1], df[0][i]])

df2 = pd.DataFrame(list1, columns=['term1','term2','freq'])
df3 = df2.sort_values(by=['freq'], ascending=False) # freq 기준으로 내림차순 정렬
df3_pos = df3.reset_index(drop=True)
```

부정 리뷰에 대해서도 같은 작업을 수행한다.


```python
# 부정 리뷰 필터링
is_neg_review = review['sentiment'] == 'negative'
neg_review = review[is_neg_review]['review'][0:100] # 100개만 추출
neg_review.reset_index(inplace=True, drop=True) # 인덱스 초기화

tokenizer = RegexpTokenizer('[\w]+')
stop_words = stopwords.words('english')

count = {} # 동시출현 빈도가 저장될 dict
for line in neg_review:
    words = line.lower() # 각 리뷰를 소문자로 변환
    tokens = tokenizer.tokenize(words) # 각 리뷰를 토큰화한 뒤 리스트에 저장
    stopped_tokens = [i for i in list(set(tokens)) if not i in stop_words+["br"]]
    stopped_tokens2 = [i for i in stopped_tokens if len(i)>1]
    for i,a in enumerate(stopped_tokens2):
        for b in stopped_tokens2[i+1:]:
            if a>b:
                count[b,a] = count.get((b,a),0) + 1
            else:
                count[a,b] = count.get((a,b),0) + 1
                
df = pd.DataFrame.from_dict(count, orient='index')

list1=[]
for i in range(len(df)):
    list1.append([df.index[i][0], df.index[i][1], df[0][i]])

df2 = pd.DataFrame(list1, columns=['term1','term2','freq'])
df3 = df2.sort_values(by=['freq'], ascending=False) # freq 기준으로 내림차순 정렬
df3_neg = df3.reset_index(drop=True)
```

이제부터 중심성 계수를 구한다. 긍정 리뷰에 대해 먼저 수행한다.


```python
import networkx as nx
import operator
```


```python
G_pos = nx.Graph()

# 동시출현 빈도가 10 이상인 단어들에 대해서만 중심성 계수 계산
for i in range((len(np.where(df3_pos['freq']>10)[0]))):
    G_pos.add_edge(df3_pos['term1'][i], df3_pos['term2'][i],
                  weight=int(df3_pos['freq'][i]))

dgr = nx.degree_centrality(G_pos) # 연결 중심성
btw = nx.betweenness_centrality(G_pos) # 매개 중심성
cls = nx.closeness_centrality(G_pos) # 근접 중심성
egv = nx.eigenvector_centrality(G_pos) # 고유벡터 중심성

sorted_dgr = sorted(dgr.items(), key=operator.itemgetter(1), reverse=True)
sorted_btw = sorted(btw.items(), key=operator.itemgetter(1), reverse=True)
sorted_cls = sorted(cls.items(), key=operator.itemgetter(1), reverse=True)
sorted_egv = sorted(egv.items(), key=operator.itemgetter(1), reverse=True)

print("** degree **")
for x in range(10):
    print(sorted_dgr[x])

print("** betweenness **")
for x in range(10):
    print(sorted_btw[x])
    
print("** closeness **")
for x in range(10):
    print(sorted_cls[x])

print("** eigenvector **")
for x in range(10):
    print(sorted_egv[x])
```

    ** degree **
    ('film', 0.7678571428571428)
    ('one', 0.6785714285714285)
    ('movie', 0.6428571428571428)
    ('like', 0.5714285714285714)
    ('story', 0.4107142857142857)
    ('see', 0.3571428571428571)
    ('really', 0.33928571428571425)
    ('time', 0.3214285714285714)
    ('good', 0.26785714285714285)
    ('way', 0.23214285714285712)
    ** betweenness **
    ('film', 0.33903040141676494)
    ('one', 0.17709317197953559)
    ('movie', 0.17506936245572613)
    ('like', 0.1421225473660539)
    ('story', 0.07367243867243868)
    ('time', 0.05227973145180937)
    ('really', 0.045019372762879255)
    ('see', 0.019190419969640757)
    ('good', 0.005902438110230317)
    ('never', 0.0030105086111579613)
    ** closeness **
    ('film', 0.8115942028985508)
    ('one', 0.7567567567567568)
    ('movie', 0.7368421052631579)
    ('like', 0.7)
    ('story', 0.6292134831460674)
    ('see', 0.6086956521739131)
    ('really', 0.6021505376344086)
    ('time', 0.5957446808510638)
    ('good', 0.5773195876288659)
    ('way', 0.56)
    ** eigenvector **
    ('film', 0.32577738126513894)
    ('one', 0.3190968223069761)
    ('movie', 0.3111839367992137)
    ('like', 0.28987877428374764)
    ('story', 0.23949399571545077)
    ('see', 0.23487326612885503)
    ('really', 0.2290961198067612)
    ('time', 0.2041526039782131)
    ('good', 0.1999930636714824)
    ('way', 0.1846403088456769)
    

계산량의 문제로 동시출현 빈도가 10 이상인 단어들에 대해서만 중심성 계수를 계산했다. 여기서 10은 동시출현 빈도의 임계값에 해당한다. 임계값은 본인의 연구 특성에 맞게 적절히 조절하면 된다.  
  
부정리뷰에 대해서도 중심성 계수를 구한다.


```python
G_neg = nx.Graph()

for i in range((len(np.where(df3_neg['freq']>10)[0]))):
    G_neg.add_edge(df3_neg['term1'][i], df3_neg['term2'][i],
                  weight=int(df3_neg['freq'][i]))

dgr = nx.degree_centrality(G_neg) # 연결 중심성
btw = nx.betweenness_centrality(G_neg) # 매개 중심성
cls = nx.closeness_centrality(G_neg) # 근접 중심성
egv = nx.eigenvector_centrality(G_neg) # 고유벡터 중심성

sorted_dgr = sorted(dgr.items(), key=operator.itemgetter(1), reverse=True)
sorted_btw = sorted(btw.items(), key=operator.itemgetter(1), reverse=True)
sorted_cls = sorted(cls.items(), key=operator.itemgetter(1), reverse=True)
sorted_egv = sorted(egv.items(), key=operator.itemgetter(1), reverse=True)

print("** degree **")
for x in range(10):
    print(sorted_dgr[x])

print("** betweenness **")
for x in range(10):
    print(sorted_btw[x])
    
print("** closeness **")
for x in range(10):
    print(sorted_cls[x])

print("** eigenvector **")
for x in range(10):
    print(sorted_egv[x])
```

    ** degree **
    ('movie', 0.8714285714285714)
    ('like', 0.7142857142857143)
    ('film', 0.7)
    ('one', 0.5571428571428572)
    ('even', 0.37142857142857144)
    ('good', 0.3142857142857143)
    ('would', 0.3142857142857143)
    ('see', 0.3)
    ('bad', 0.2714285714285714)
    ('get', 0.2714285714285714)
    ** betweenness **
    ('movie', 0.4221446841633175)
    ('film', 0.21467746539175103)
    ('like', 0.18547111618540182)
    ('one', 0.08406324869057788)
    ('even', 0.016866166959334662)
    ('would', 0.011109154245800217)
    ('good', 0.009598906959155409)
    ('time', 0.0066402265781147765)
    ('see', 0.005889765858709958)
    ('bad', 0.0047839381690313355)
    ** closeness **
    ('movie', 0.8860759493670886)
    ('like', 0.7777777777777778)
    ('film', 0.7692307692307693)
    ('one', 0.693069306930693)
    ('even', 0.6140350877192983)
    ('good', 0.5932203389830508)
    ('would', 0.5932203389830508)
    ('see', 0.5882352941176471)
    ('bad', 0.5785123966942148)
    ('get', 0.5785123966942148)
    ** eigenvector **
    ('movie', 0.3236602186574508)
    ('like', 0.30506211998070926)
    ('film', 0.3022658563974347)
    ('one', 0.2816938284701203)
    ('even', 0.2300741218508994)
    ('good', 0.20777906863755186)
    ('see', 0.20582658508163784)
    ('get', 0.20321373041522744)
    ('would', 0.19758153311708543)
    ('bad', 0.19224043093925175)
    


출처 : 잡아라! 텍스트 마이닝 with 파이썬 (서대호)
{: .notice--success}