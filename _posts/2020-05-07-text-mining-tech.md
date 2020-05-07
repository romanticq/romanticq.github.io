---
title: "[텍스트 마이닝] 텍스트 마이닝 기법"
excerpt: "텍스트 마이닝 기법에 대해 정리한다."
toc: true
toc_sticky: true
header:
  teaser: /assets/images/3280_1552550473.jpg

categories:
  - 머신러닝
tags:
  - 텍스트 마이닝
last_modified_at: 2020-05-07
---

# 4. 텍스트 마이닝 기법

## 4.1 단어 빈도분석

단어 빈도분석은 전체 문서 또는 문서별 단어 출현빈도를 보여준다.  
본격적인 분석 전 전체 텍스트 데이터에 대한 흐름을 살펴볼 수 있다.  
  
출현빈도가 높을수록 핵심 단어에 해당한다. 일반적으로 문장에서 불용어는 높은 출현빈도를 보이므로 빈도분석을 위해서는 반드시 불용어를 제거해야한다.

- `단어구름(wordcloud)` : 단어들의 출현 빈도에 따라 크기를 달리 하여 나타내는 방법. 핵심 단어들을 한 눈에 파악할 수 있다.  

***

**< 트럼프 취임 연설문 빈도분석 >**

필요한 패키지들을 import한다.

```python
import pandas as pd
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
```

분석할 텍스트 파일을 불러온다.  
- `open("경로","r")` : 파일을 읽기모드로 연다.  
- `readline()` : 파일의 첫 번째 줄을 읽어 출력한다.

```python
f = open("E:\\text-mining\\tmwithpython-master\\트럼프취임연설문.txt", 'r')
lines = f.readlines()[0]
f.close()

lines[0:100]
```
>' Chief Justice Roberts, President Carter, President Clinton, President Bush, President Obama, fellow'

트럼프 취임 연설문 텍스트파일은 enter가 삽입되지 않은 한 줄짜리 데이터이다. 만일 `lines`변수에 `f.readline()`으로 저장하게 되면

- [ 내용 ]

위처럼 불러온 텍스트 파일 양 옆에 대괄호가 쳐져있다. 만일 슬라이싱을 이용하고 싶다면 대괄호를 벗겨내기 위해 인덱싱(`f.readline()[0]`)을 이용한다.

```python
tokenizer = RegexpTokenizer('[\w]+')

stop_words = stopwords.words('english')

words = lines.lower()

tokens = tokenizer.tokenize(words)
stopped_tokens = [i for i in list((tokens)) if not i in stop_words]
stopped_tokens2 = [i for i in stopped_tokens if len(i)>1]

pd.Series(stopped_tokens2).value_counts().head(10)
```
> america     20  
american    11  
people      10  
country      9  
one          8  
every        7  
nation       7  
new          6  
great        6  
world        6  
dtype: int64

- `RegexpTokenizer('[\w]+')` : 정규표현식 [\w]+에 해당하는 내용을 제거한 뒤 tokenize 실행.

***

**< 문재인 대통령 취임연설문 빈도분석 >**

필요한 패키지를 import하고 분석할 텍스트를 변수에 저장한다.

```python
from konlpy.tag import Hannanum
hannanum = Hannanum()

f = open("E:\\text-mining\\tmwithpython-master\\문재인대통령취임연설문.txt", 'r')
lines = f.readlines()
f.close()
```

```python
def flatten(l):
    flatList = []
    for elem in l:
        if type(elem) == list:
            for e in elem:
                flatList.append(e)
        else:
            flatList.append(elem)
    return flatList
```

```python
word_list=flatten(temp)
word_list=pd.Series([x for x in word_list if len(x)>1])
word_list.value_counts().head(10)
```
>대통령     29  
국민      19  
대한민국     9  
우리       8  
여러분      7  
국민들      6  
나라       6  
역사       6  
세상       5  
대통령의     5  
dtype: int64

