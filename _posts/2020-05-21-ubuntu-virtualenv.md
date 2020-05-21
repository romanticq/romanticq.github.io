---
title: "Ubuntu20에서 가상환경으로 python3.6 이용하기"
excerpt: "우분투(리눅스)에서의 가상환경 구축 방법을 정리한다."
toc: true
toc_sticky: true
header:
  teaser: /assets/images/3280_1552550473.jpg

categories:
  - 우분투
tags:
  - 가상환경

use_math: true
last_modified_at: 2020-05-21
---

파이썬의 라이브러리들을 사용하다보면 여러 버전의 파이썬을 동시에 다뤄야 할 일이 생긴다. 보통은 라이브러리, 또는 패키지의 호환성이 문제가 된다. 버전별로 파이썬을 설치한 다음 로컬 내에서 명령어를 통해 직접 버전을 선택하는 방법도 있지만 대부분의 개발자들은 가상환경 구축을 추천한다.
  
우분투(리눅스)에서 가상환경을 구축하여 동시에 여러 버전의 파이썬을 이용해보자.
  
# 1. 가상환경의 종류

파이썬에서 동작하는 가상환경 라이브러리에는 다음과 같은 것들이 있다.
* `virtualenv`
* `pyenv`
* `pipenv`
* `venv`

무엇을 사용해도 상관 없지만 초심자가 이용하기엔 `virtualenv`가 가장 좋다고 한다. 이유는 관련 reference가 풍부해 오류에 대한 문서를 쉽게 찾을 수 있기 때문이다. `venv`같은 경우 python3에 기본적으로 내장되어 있으며 `virtualenv`와 사용법이 대부분 유사하다. 그러나 이미 구축해놓은 가상 환경의 python에 대해 버전 변경이 안되는 문제가 있다고 한다. 따라서 이 포스팅에서는 `virtualenv`를 통해 가상환경을 구축하는 방법을 소개할 것이다.

***

# 2. 현재 환경

현재 로컬에 설치된 파이썬은 3.8버전이다. tensorflow가 아직 3.8버전에서 완벽히 호환되지 않아 오류가 발생하는 경우가 있으므로 python3.6 버전을 이용할 수 있는 가상 환경을 구축하는 것이 목표이다.

***

# 3. python 3.6 설치

## 3.1 설치 방법

우선 로컬에 python3.6을 먼저 설치한다. 본인이 이용하고자 하는 버전의 python이 이미 설치되어 있다면 건너뛰어도 상관 없다.  
  
방법은 두 가지이다.

1) python 홈페이지에서 소스 파일을 직업 받아서 설치  
<br>
2) PPA(Personal Package Archive) 이용
{: .notice--info}

1번 방법은 찾아보니 너무 복잡하다. 또 시간이 오래걸린다 하여 2번 방법을 이용할 것이다.
  
PPA는 개인용 소프트웨어 패키지 저장소이다. 제 3자가 올려놓은 python 파일을 불러와 설치하는 개념정도로 이해된다. 설치 방법은 다음 코드를 입력하면 된다.

```
$ sudo apt-get update
$ sudo add-apt-repository ppa:deadsnakes/ppa
$ sudo apt-get update
$ sudo apt install python3.6
```

위 방법은 PPA를 이용한 것이며, 사용 방법은 다음과 같다.

```
$ sudo add-apt-repository 저장소 이름
$ sudo apt-get update
$ sudo apt install 프로그램명

# 삭제를 원할 시
$ sudo add-apt-repository --remove 저장소 이름
```

1) `$ sudo add-apt-repository 저장소 이름`을 통해 PPA에서 다운로드 하고자 하는 것을 가져온다.  
<br>
2) `$ sudo apt-get update` 를 통해 수동적으로 ppa를 추가/삭제 하면 추가/삭제 했다는 신호를 줘야 한다.  
<br>
3) `$ sudo apt install 프로그램명` 을 통해 원하는 프로그램을 설치한다.  
{: .notice--info}

deadsnakes는 구글링해보니 파이썬의 구버전과 신버전의 소스파일을 모아놓는 팀이라고 검색된다.

## 3.2 설치 확인

다음 명령어를 통해 python설치 여부를 확인할 수 있다.

```
# 파이썬이 설치된 디렉토리 확인
$ which python
/usr/bin/python

# 디렉토리 내에서 python 검색
$ ls /usr/bin/ | grep python
python
python3
python3-config
python3.6
python3.6m
python3.8
python3.8-config
```

***

# 4. virtualenv 라이브러리 설치

virtualenv는 pip을 통해 설치하므로 pip 설치가 먼저 선행되어야 한다.

## 4.1 virtualenv 설치

pip을 통해 virtualenv를 설치한다.

```
$ sudo pip3 install virtualenv
```

## 4.2 virtualenv 가상환경 구성하기

하나의 가상환경은 하나의 디렉토리에 대응된다. 프로젝트 디렉토리 이름은 `python3.6`으로 설정했다. 해당 프로젝트 디렉토리로 이동한 다음 `virtualenv` 명령어를 통해 가상환경을 구성할 수 있다.

```
$ cd python3.6
$ virtualenv tsflow
```

`$ virtualenv 가상환경이름` 을 통해 현재 디렉토리에 `가상환경이름`으로 디렉토리가 하나 생성된다. 이 디렉토리가 하나의 가상환경에 대응하는 것이다. 여기서 가상환경이름은 tsflow로 설정했다.

***

다음은 virtualenv에 파이썬 버전을 지정한다.

```
$ virtualenv tsflow --python=python3.6
```

virtalenv가 global에 설치된 패키지를 상속받기 원한다면 다음 명령어를 입력하면 된다. 필요에따라 이용하면 되나 여기서는 건너뛸 것이다.
```
$ virtualenv tsflow --system-site-packages
```

## 4.3 생성된 가상환경 활성화

`virtualenv` 가상환경을 활성화하려면 다음 명령어를 입력한다. 해당 가상환경이 생성된 디렉토리인 python3.6에서 실행해야 한다.

```
$ source tsflow/bin/activate
```

다음과 같이 좌측에 `(가상환경이름)`이 보이면 제대로 활성화 된 것이다.

![](https://github.com/romanticq/romanticq.github.io/blob/master/assets/images/virtualenv/virtualenv%20activate.png?raw=true)

***

파이썬 버전 확인 명령어를 입력해서 원하는 버전으로 나오는지 확인해보자.

![](https://github.com/romanticq/romanticq.github.io/blob/master/assets/images/virtualenv/virtualenv%20python%20ver.png?raw=true)

제대로 구현되었다.

***

가상환경 나오려면 다음 명령어를 입력한다.
```
$ deactivate
```

***

# 5. jupyter notebook에서 실행하기

## 5.1 새로운 커널 추가

주피터 노트북을 통해 가상환경 버전의 python으로 작업을 해보자. 새로운 ipynb 파일을 만들기 위해 New를 클릭하면 다음과 같이 나온다.

![](https://github.com/romanticq/romanticq.github.io/blob/master/assets/images/virtualenv/virtualenv%20kernel.png?raw=true)

별다른 작업을 하지 않았다면 python3.6이란 커널은 보이지 않을 것이다. 해당 커널을 추가하는 방법을 알아보자.  

1) 우선 해당 가상 환경을 활성화 한다.

```
$ source tsflow/bin/activate
```

2) `ipykernel` 라이브러리를 다운로드 한다.

```
$ pip3 install ipykernel
```

3) 다음 명령어를 입력한다.

```
$ python -m ipykernel install --user --name={kernelname}
```

{kernelname}에 원하는 커널 이름을 입력하면 된다. 여기서는 python3.6을 다음과 같이 입력했다.

```
$ python -m ipykernel install --user --name=python3.6
```

주피터 노트북을 실행하여 보면 위 그림에서와 같이 새로운 커널 python3.6이 추가된 것을 확인할 수 있다.

## 5.2 작동여부 확인

두개의 서로 다른 커널 python3, python3.6이 제대로 작동하는지 확인해보도록 하자.  
우선 python3과 python3.6 두개의 커널로 각각 주피터 노트북 파일을 생성하여 다음과 같이 입력해보자.

![](https://github.com/romanticq/romanticq.github.io/blob/master/assets/images/virtualenv/python3%20kernel.png?raw=true)

python3 커널로 생성한 파일에서는 3.8버전으로 나온다.
  

![](https://github.com/romanticq/romanticq.github.io/blob/master/assets/images/virtualenv/python3.6%20kernel.png?raw=true)

python3.6 커널로 생성한 파일에서는 3.6버전임을 확인할 수 있다.

## 5.3 필요 라이브러리 설치

기존 로컬에서는 넘파이, 판다스와 같은 여러가지 패키지들이 설치되어있다. 그러나 새로운 가상환경은 깨끗한 상태이다.  
  
가상환경에서 이전과 같은 작업들을 하고싶다면 이용하고자 하는 라이브러리들을 가상환경을 활성화시킨 상태에서 다시 설치해주어야 한다.  

가상환경에 numpy 최신 버전을 설치했더니 import 오류가 발생했다. 최신 버전의 넘파이가 python3.6과 호환에 문제가 있는것으로 판단되어 삭제 후 가상환경에 1.18.1 버전으로 다시 설치했더니 해결되었다.
{: .notice--danger}

***

# 참고
1) <https://medium.com/%EB%8F%84%EC%84%9C-%ED%8C%8C%EC%9D%B4%EC%8D%AC-%EC%9B%B9%ED%94%84%EB%A1%9C%EA%B7%B8%EB%9E%98%EB%B0%8D-%EC%8B%A4%EC%A0%84%ED%8E%B8-%EC%9A%94%EC%95%BD/chapter-6-%EA%B0%80%EC%83%81-%ED%99%98%EA%B2%BD-%EC%82%AC%EC%9A%A9%ED%95%98%EA%B8%B0-%EC%83%88%EB%A1%AD%EA%B2%8C-%EC%A0%95%EB%A6%AC-30d5940de012>  
<br>
2) <https://worthpreading.tistory.com/84>  
<br>
3) <https://anpigon.github.io/blog/kr/@anpigon/-virtualenv-python--1546840427366/>  
<br>
4) <https://medium.com/@equus3144/%ED%8C%8C%EC%9D%B4%EC%8D%AC-%EA%B0%80%EC%83%81%ED%99%98%EA%B2%BD%EC%9D%80-%EC%99%9C-%EC%9D%B4%EB%A0%87%EA%B2%8C-%EB%8B%A4%EC%96%91%ED%95%98%EA%B3%A0-%EA%B0%9C%EB%B0%9C%EC%9E%90%EB%93%A4%EC%9D%80-%EC%99%9C-%EC%9D%B4%EB%A0%87%EA%B2%8C-%EB%8B%A4%EC%96%91%ED%95%9C-%EA%B0%80%EC%83%81%ED%99%98%EA%B2%BD%EC%9D%84-%EB%A7%8C%EB%93%A4%EC%97%88%EC%9D%84%EA%B9%8C-8173992f28e2>  
<br>
5) <https://www.crocus.co.kr/1592>
{: .notice--info}
