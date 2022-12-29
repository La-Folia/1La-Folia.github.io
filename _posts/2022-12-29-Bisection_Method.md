---
layout: post
title:  "Bisection_Method"
date:   2022-12-29 17:44:54 +0900
categories: Bisection_Method
permalink: /Python_Bisection_Method/
---
# Root Finding

### Bisection_Method

루트 찾기는 1차원 방정식의 솔루션을 찾는 문제를 말한다.

루트 찾기는 반복적으로 진행된다. 대략적인 솔루션에서 시작하여 유용한 알고리즘이 미리 결정된 수렴기준을 만족할 때까지 솔루션을 개선한다. 원할하게 변화하는 일부 함수의 경우 초기 추측이 충분하다면 좋은 알고리즘은 항상 수렴된다.

Bisection Method : 이분법


```python
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 15})
%config InlineBackend.figure_format='retina'
```

가장 간단한 루트 찾기 알고리즘은 이분법(이진법)이다. 알고리즘은 모든 연속 함수에 적용된다.

![%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202022-12-28%20%E1%84%8B%E1%85%A9%E1%84%8C%E1%85%A5%E1%86%AB%2011.09.32.png](attachment:%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202022-12-28%20%E1%84%8B%E1%85%A9%E1%84%8C%E1%85%A5%E1%86%AB%2011.09.32.png)

## 이진법 개념
##### -구간을 둘로 나누면 솔루션은 하나의 하위 구간 내에 존재함
##### -솔루션의 구간을 선택하고, 다시 반으로 쪼갬
##### -솔루션을 찾을 때까지 계속해서 반복한다.(원하는 정확도 내에서)

## 알고리즘
### 이분법의 절차는 다음과 같다.

#### 1. 시작 간격 선택 [a0, b0] 우리가 알고 있는 근은 (즉, f(a0)f(b0) < 0).
#### 2. 컴퓨팅 f(m0) 어디 m0 = (a0 + b0) / 2중간점이다.

![%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202022-12-28%20%E1%84%8B%E1%85%A9%E1%84%8C%E1%85%A5%E1%86%AB%2011.14.58.png](attachment:%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202022-12-28%20%E1%84%8B%E1%85%A9%E1%84%8C%E1%85%A5%E1%86%AB%2011.14.58.png)

## 절대 오차

$$
\epsilon_{n + 1} = \epsilon_n/2
$$

#### 한번 반복 후 루트를 포함하는 범위가 2배 감소한다는 것이 알고리즘에서 분명하다.
#### $\epsilon $(원하는 종료 횟수,솔루션을 찾았을때 간격)과 $\epsilon_0(초기의 간격크기)$에 대한 식으로 정리하게 되면 
$$ n = log_2 \frac{\epsilon_0}{\epsilon} $$ 
이것이 허용 오차를 달성하는 데 필요한 반복 횟수이다. (이는 실제로 유도하면 나옴)
$$
|x_{\rm true} - x_n| \le \epsilon = \frac{\epsilon_0}{2^{n + 1}}
$$
그리고 이것을 x_true 즉, 정확한 해, 그리고 x_n은 n번 후 반복 후의 해의 사이 값은 $\epsilon$ 보다 같거나 작다. 


##### 추가 요인 $1/2$은 $n$ 반복 후 하위 구간의 중간점을 반환한다는 사실에서 비롯된다.
# (초기에 한번 1/2로 나눠주고 시작해서??이를 포함시키려고 n+1 하는듯??)

## 구현

###  for _ in range(n) 에서 _ 언더바의 의미
- 인터프리터(Interpreter)에서 마지막 값을 저장할 때
- 값을 무시하고 싶을 때 (흔히 “I don’t care"라고 부른다.)
- 변수나 함수명에 특별한 의미 또는 기능을 부여하고자 할 때
- 국제화(Internationalization, i18n)/지역화(Localization, l10n) 함수로써 사용할 때
- 숫자 리터럴값의 자릿수 구분을 위한 구분자로써 사용할 때



```python
def bisection_by(f, a, b, n):
    a_n = a
    f_of_a_n = f(a_n) #f 함수에 a를 넣은 함수 값
    b_n = b
    f_of_b_n = f(b_n) #f 함수에 b를 넣은 함수 값
    
    #f: 해를 추정하고자 하는 방정식
    #a,b :interval, 간격, 즉 함수에서 [a, b]를 뜻함
    #x_n : n번째 구간의 중간점
    #- 만약 f(m-n) == 0인 경우 중간지점이 m-n = (a_n + b_n) / 2이라면, 이 솔루션을 반환
    #- 만약 f(a-n), f(b-n)의 모든 부호가 동일 --> 이등분 방법 실패를 반환.
    
    #validity check 타당한지 체크
    if f_of_a_n * f_of_b_n >= 0:
        print("Bisection method fails.")
        return None
    
    
    m_n = 0.5 * (a_n + b_n) #첫번째의 중간값.
    f_of_m_n = f(m_n) #중간 값을 넣은 함수 값.
    
    # iterations 반복
    for _ in range(n) : 
        #여기서 _의 의미는 i라고 생각해도 될듯. 
        #그냥 몇 번 반복했는지를 후에 마지막 값 저장하려고 쓰는 듯
        if f_of_m_n == 0: #즉 중간값의 함수 값이 0이 나오면 해를 찾은 거겠지.
            print("Found exact solution.")
            return m_n
        
        elif f_of_a_n * f_of_m_n < 0: 
            #만약 a를 넣은 값이 중간값과 곱했을때 음수이면, 그 사이이에 해가 있다.
            b_n = m_n #b값은 그러면 반으로 잘라서 m_n을 b_n으로 두어도 된다.
            f_of_b_n = f_of_m_n #마찬가지
            
        elif f_of_b_n * f_of_m_n < 0:
            a_n = m_n
            f_of_a_n = f_of_m_n
            
        else:
            print("Bisection method fails.")
            return None
        
        #다시 반으로 쪼개고, 함수값 넣어서 다시 부호를 체크한다.
        m_n = 0.5 * (a_n + b_n)
        f_of_m_n = f(m_n)
        
    return m_n
        
```


```python
#f의 의미.
f = lambda x: x**3
a = 3
f(a)
```




    27




```python
# _ under bar의 의미.
n = 100
for _ in range(n):
    print(_)
```

    0
    1
    2
    3
    4
    5
    6
    7
    8
    9
    10
    11
    12
    13
    14
    15
    16
    17
    18
    19
    20
    21
    22
    23
    24
    25
    26
    27
    28
    29
    30
    31
    32
    33
    34
    35
    36
    37
    38
    39
    40
    41
    42
    43
    44
    45
    46
    47
    48
    49
    50
    51
    52
    53
    54
    55
    56
    57
    58
    59
    60
    61
    62
    63
    64
    65
    66
    67
    68
    69
    70
    71
    72
    73
    74
    75
    76
    77
    78
    79
    80
    81
    82
    83
    84
    85
    86
    87
    88
    89
    90
    91
    92
    93
    94
    95
    96
    97
    98
    99


## 생각해볼 것??
- 모니터 기능을 제공하여 외부에서 반복을 모니터링 하는 것이 유용하다.
- 또한 중간 값을 관찰하여 반복을 종료할 시기를 결정하는 것이 유용하다.

# 이해가 안됨...

## bisection_while 을 정의해보자


```python
def bisection_while(f, xinit, predicate):
    a_n, b_n = xinit 
    f_1st = f(a_n)
    
    #f: 해를 추정하고자 하는 방정식
    #xinit :interval, 간격, 즉 함수에서 [a, b]를 뜻함, 여기선 튜플로 들어감.
    #predicate : 호출 가능, 세 가지 인수를 사용하는 함수를 의미.
    # -i : 반복 횟수
    # -xy : 현재 반복에서 중간점과 함수 값의 쌍
    # -dx : x 값의 변경
    #그리고 boolean을 반환한다, True이면 검색이 계속되고 False이면 검색이 종료된다.
    
    #X_n : 숫자
    #이등분 방법으로 계산된 n번째 구간의 중간점, 초기간격은 a, b로 [a_0, b_0]가 결정됨.
    #위와 마찬가지로 (m_n) == 0인 경우 중간점 m_n = (a_n + b_n)/2이면 함수는 이 솔루션을 반환합니다.
    #만약 값 f(a_n), f(b_n), f(m_n)의 모든 부호가 동일하다면 반복하면 이등분 방법이 실패하고 없음을 반환합니다.
    
    #유효한지 체크
    if f(a_n) * f(b_n) >= 0:
        print("Bisection method fails.")
        return None
    

    i = 1
    x_mid = 0.5 * (a_n + b_n)
    f_mid = f(x_mid)
    
    #반복
    #----------------------------???????????????????----------------------
    while predicate(i, (x_mid, f_mid), 0.5 * abs(a_n - b_n)):
        if f_1st * f_mid > 0:
            a_n = x_mid
            f_1st = f_mid
            
    ##???? 도저히 이해가 안됨???
    #----------------------------???????????????????----------------------
            
        else:
            b_n = x_mid
            
        i = i + 1
        x_mid = 0.5 * (a_n + b_n)
        f_mid = f(x_mid)
 
    return x_mid
    
    
```

# 운동
## 큐브루트
$\sqrt[3]{2}$ 근사치를 구하여
$$
x^3 - 2 = 0
$$
를 해결



```python
cuberoot2_approx = bisection_while(lambda x: x*x*x - 2, (1, 2), 
                                   lambda i, xy, dx: abs(dx) > 1e-10)
(cuberoot2_approx, abs(2**(1/3) - cuberoot2_approx))
```




    (1.2599210498738103, 2.1062929178583545e-11)



# 황금 비율

황금비율은 다음과 같이 주어진다
$$
\phi = \frac{1 + \sqrt 5}{2} \approx 1.6180339887498948482
$$
그리고 그 솔루션은 이러하다.
$f(x) \equiv x^2 - x - 1 = 0$.

여기서 이를 이분법으로 구현해보자, $N = 25$ iterations on $[1, 2]$ to approximate $\phi$.


```python
approx_phi = bisection_by(lambda x: x*(x - 1) - 1, 1, 2, 25)
approx_phi
```




    1.618033990263939



$(2 - 1)/2^{26}$: 절대 오차는 이것 보다 작도록 보장 된다


```python
error_bound = 2 ** (-26)
abs((1 + 5 ** 0.5) / 2 - approx_phi) < error_bound #앞의 공식에 의거하면
```




    True



# 이를 시각화해보자


```python
ix_pairs = []
iy_pairs = []
idx_pairs = []

def intercept(i, xy, dx):
    ix_pairs.append([i, xy[0]])
    iy_pairs.append([i, abs(xy[1])])
    idx_pairs.append([i, abs(dx)])
    return i <= 11

bisection_while(lambda x: x*(x - 1) - 1, (1, 2), intercept )
```




    1.617919921875




```python
plt.figure(figsize=(11, 4))

plt.subplot(1, 3, 1)
plt.plot(*zip(*ix_pairs), 'o:k', label="bisection")
plt.plot([1, len(ix_pairs)], (1 + 5 ** 0.5) / 2 * np.ones((2,)), '--r',
         label="solution")
plt.title("$x$ approximation")
plt.xlabel("iteration count")
plt.ylim([1, 2])
plt.legend()

plt.subplot(1, 3, 2)
plt.semilogy(*zip(*iy_pairs), 'o:k')
plt.title("$|f(x_n)|$")
plt.xlabel("iteration count")
plt.grid()

plt.subplot(1, 3, 3)
plt.semilogy(*zip(*idx_pairs), 'o:k', label="$\Delta x_n$")
plt.semilogy(*zip(*[(idx[0], 2/2**idx[0]) for idx in idx_pairs]),
             'r--', label="$\Delta x \propto 2^{-n}$")
plt.xlabel("iteration count")
plt.title("|$\Delta x$|")
plt.grid()
plt.legend()

plt.tight_layout()
plt.show()
```


    
![png](B![png](/assets/img/blog/Bisection_Method_files/Bisection_Method_33_0.png)    



```python

```
