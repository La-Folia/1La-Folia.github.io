---
layout: post
title:  "Basic Plotting"
date:   2022-12-27 13:13:54 +0900
categories: example
permalink: /Python/
---
# Basic Plotting

## Procedure


```python
import numpy as np
import matplotlib.pyplot as plt
x = [-5, -2, 0, 1, 3]
y = [2, -1, 1, -4 ,3]
plt.plot(x, y)
plt.show()
```


    
![png](/assets/img/blog/Matplotlib_files/Matplotlib_2_0.png)
    



```python
x = np.linspace(-2, 2, 100)
y = x ** 2
plt.plot(x, y)
plt.show()
```


    
![png](/assets/img/blog/Matplotlib_files/Matplotlib_3_0.png)
    


![%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202022-12-27%20%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE%205.00.41.png](attachment:%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202022-12-27%20%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE%205.00.41.png)


```python
x = np.linspace(-2, 2, 41)
y = np.exp(-x ** 2) * np.cos(2 * np.pi * x)
plt.plot(x, y,
        alpha = 0.4, label = 'Decaying Cosine',
        color = 'red', linestyle = 'dashed',
        linewidth = 1, marker = 'o',
        markersize = 3, markerfacecolor = 'blue',
        markeredgecolor = 'blue'
        )

plt.ylim([-2, 2])
plt.legend()
#plt.savefig("14.pdf") --> pdf로 저장하는 법 
```


    
![png](/assets/img/blog/Matplotlib_files/Matplotlib_5_0.png)
    


## 형식 문자열

fmt = '[marker][line][color]'


```python
x = np.linspace(-5, 5, 41)
y = 1/ (1 + x ** 2)
plt.plot(x, y,
        color = 'black',
        linestyle = 'dashed',
        marker = 's'
        )
plt.show()
```


    
![png](/assets/img/blog/Matplotlib_files/Matplotlib_8_0.png)
    



```python
plt.plot(x, y, 'ks--')
plt.show()
```


    
![png](/assets/img/blog/Matplotlib_files/Matplotlib_9_0.png)
    


![%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202022-12-27%20%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE%205.15.54.png](attachment:%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202022-12-27%20%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE%205.15.54.png)


```python
def factorial(n):
    assert(n > 0)
    return np.prod(np.arange(1, n + 1))

x = np.linspace(-1, 1, 200)
y = np.exp(x)

fN_minus_1 = 1
for N in range(1, 5):
    fN = fN_minus_1 + x ** N / factorial(N)
    
    plt.subplot(2, 2, N)
    plt.plot(x, y, 'k-', label = "$f_{N}$")
    plt.plot(x, fN, 'r--', label = f"$f_{N}$")
    plt.title(f"N = {N}")
    plt.legend() 

    plt.xlim([f(x) * 1.1 for f in (np.min, np.max)])
    plt.ylim([f(y) * 1.1 for f in (np.min, np.max)])
    
    plt.xlabel('x')
    plt.ylabel('y')
    
    fN_minus_1 = fN #1이 끝나고 다시 2가 되어야 하는데, 1에서 곱해준 값으로 취해버리면 안되니, 값을 초기화 시켜주기 위해, 다시 1로 만듬.
    
plt.tight_layout()
plt.show()
```


    
![png](/assets/img/blog/Matplotlib_files/Matplotlib_11_0.png)
    



```python
N = 2000
points = np.random.rand(2, N)
sizes = np.random.randint(20, 120, (N,))

colors = np.random.rand(N, 4)

plt.figure(figsize = (12, 5))
plt.scatter(*points, c = colors, s = sizes)
plt.axis('off')
plt.show()
```


    
![png](/assets/img/blog/Matplotlib_files/Matplotlib_12_0.png)
    



```python
samples = np.random.randn(10000)
plt.hist(samples,
        bins = 20, density = True,
        alpha = 0.5, color = (0.3,0.8,0.1))
plt.title('Random Samples = Normal Distribution')
plt.ylabel('PDF')

x = np.linspace(-4, 4, 100)
y = 1 / (2 * np.pi) ** 0.5 * np.exp(-0.5 * x ** 2)
plt.plot(x, y, 'b-', alpha = 0.8)

plt.show()
```


    
![png](/assets/img/blog/Matplotlib_files/Matplotlib_13_0.png)
    



```python
x = np.linspace(-1, 1, 50) * np.pi * 2
y  = np.cos(x)
plt.plot(x, y, 'b', label = 'cos(x)')

y2 = 1 - x ** 2 / 2
plt.plot(x, y2, 'r-.', label = 'Degree 2')

y4 = 1- x**2 / 2 + x ** 4 / 24
plt.plot(x, y4, 'g:', label = 'Degree 4')

plt.legend(loc = 'upper center')
plt.grid(True, linestyle = ':')

plt.xlim([f(x) * 1 for f in (np.min, np.max)])
plt.ylim([f(y) * 3 for f in (np.min, np.max)])

plt.title('Taylor Polynomials of cos(x) at x = 0')
plt.xlabel('x')
plt.ylabel('y')

plt.show()


```


    
![png](/assets/img/blog/Matplotlib_files/Matplotlib_14_0.png)
    



```python
np.linspace(-1, 1, 50) * np.pi * 2
```




    array([-6.28318531, -6.02672876, -5.77027222, -5.51381568, -5.25735913,
           -5.00090259, -4.74444605, -4.48798951, -4.23153296, -3.97507642,
           -3.71861988, -3.46216333, -3.20570679, -2.94925025, -2.6927937 ,
           -2.43633716, -2.17988062, -1.92342407, -1.66696753, -1.41051099,
           -1.15405444, -0.8975979 , -0.64114136, -0.38468481, -0.12822827,
            0.12822827,  0.38468481,  0.64114136,  0.8975979 ,  1.15405444,
            1.41051099,  1.66696753,  1.92342407,  2.17988062,  2.43633716,
            2.6927937 ,  2.94925025,  3.20570679,  3.46216333,  3.71861988,
            3.97507642,  4.23153296,  4.48798951,  4.74444605,  5.00090259,
            5.25735913,  5.51381568,  5.77027222,  6.02672876,  6.28318531])




```python
t = np.linspace(0, 2 * np.pi, 100)

x = 16 * np.sin(t) ** 3
y = 13 * np.cos(t) - 5 * np.cos (2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t)

plt.plot(x, y, c = (1, 0.2, 0.5), lw = 5)

plt.title('Heart!')
plt.axis('equal')
plt.axis('off')

plt.show()
```


    
![png](/assets/img/blog/Matplotlib_files/Matplotlib_16_0.png)
    



```python
plt.subplot(2, 2, 1)
x = np.linspace(-9, 9, 200)
y = np.sqrt(np.abs(x))
plt.plot(x, y , 'b')
plt.title("function 1")

plt.subplot(2, 2, 2)
x = np.linspace(0, 4 * np.pi, 200)
y = np.sin(x) + np.sin(2 * x)
plt. plot(x, y, 'b')
plt.title("function 2")

plt.subplot(2, 2, 3)
x = np.linspace(-5, 5, 200)
y = np.arctan(x)
plt. plot(x, y, 'b')
plt.title("function 3")

plt.subplot(2, 2, 4)
x = np.linspace(-2, 3, 200)
y = np.array([x + a for a in [2, 1, -1, -2, -3]]).prod(0) 
#2, 1, -1...일때를 각각 행렬로 만들고 그걸 요소들을 곱하면, 
# 위의 식과 같은 결과를 내보내겠지.
plt. plot(x, y, 'b')
plt.title("function 4")

```




    Text(0.5, 1.0, 'function 4')




    
![png](/assets/img/blog/Matplotlib_files/Matplotlib_17_1.png)
    



```python
np.array([x + a for a in [2, 1, -1, -2, -3]]).prod(0)

```




    array([  0.        ,   1.44091956,   2.75228435,   3.9395244 ,
             5.00794782,   5.96274197,   6.80897468,   7.55159544,
             8.19543662,   8.74521463,   9.20553117,   9.58087442,
             9.8756202 ,  10.09403325,  10.24026833,  10.31837154,
            10.3322814 ,  10.28583016,  10.18274492,  10.02664889,
             9.82106253,   9.56940483,   9.27499443,   8.94105089,
             8.57069585,   8.16695424,   7.73275549,   7.27093474,
             6.78423401,   6.27530342,   5.7467024 ,   5.20090089,
             4.64028052,   4.06713584,   3.4836755 ,   2.89202346,
             2.2942202 ,   1.69222391,   1.08791169,   0.48308075,
            -0.12055036,  -0.72134059,  -1.31772515,  -1.90821435,
            -2.49139237,  -3.06591604,  -3.63051369,  -4.1839839 ,
            -4.72519435,  -5.25308054,  -5.76664469,  -6.26495443,
            -6.74714169,  -7.21240146,  -7.65999056,  -8.08922649,
            -8.49948621,  -8.89020493,  -9.2608749 ,  -9.61104423,
            -9.94031568, -10.24834547, -10.53484204, -10.79956489,
           -11.04232337, -11.26297546, -11.46142658, -11.6376284 ,
           -11.79157761, -11.92331474, -12.03292296, -12.12052687,
           -12.1862913 , -12.2304201 , -12.25315497, -12.25477419,
           -12.23559152, -12.19595489, -12.1362453 , -12.05687553,
           -11.95828899, -11.84095851, -11.70538513, -11.5520969 ,
           -11.38164768, -11.19461596, -10.9916036 , -10.7732347 ,
           -10.54015434, -10.29302743, -10.03253744,  -9.75938527,
            -9.47428803,  -9.17797778,  -8.87120042,  -8.55471442,
            -8.22928965,  -7.89570616,  -7.55475299,  -7.20722698,
            -6.85393154,  -6.49567547,  -6.13327175,  -5.76753634,
            -5.39928697,  -5.02934197,  -4.65851903,  -4.287634  ,
            -3.91749972,  -3.54892481,  -3.18271243,  -2.81965912,
            -2.4605536 ,  -2.10617554,  -1.75729437,  -1.41466807,
            -1.07904202,  -0.75114771,  -0.43170161,  -0.12140395,
             0.17906251,   0.46903363,   0.74786516,   1.01493386,
             1.26963877,   1.51140237,   1.73967183,   1.95392012,
             2.15364734,   2.33838179,   2.50768127,   2.66113425,
             2.79836103,   2.91901501,   3.02278386,   3.1093907 ,
             3.17859534,   3.23019547,   3.26402784,   3.27996948,
             3.27793893,   3.25789736,   3.21984986,   3.1638466 ,
             3.08998403,   2.99840609,   2.8893054 ,   2.76292449,
             2.61955697,   2.45954874,   2.28329921,   2.09126248,
             1.88394855,   1.66192452,   1.42581579,   1.17630727,
             0.91414457,   0.64013522,   0.35514983,   0.06012335,
            -0.24394376,  -0.55598435,  -0.87486333,  -1.19937651,
            -1.52824936,  -1.86013586,  -2.19361725,  -2.52720083,
            -2.85931881,  -3.18832706,  -3.5125039 ,  -3.83004896,
            -4.1390819 ,  -4.43764127,  -4.72368328,  -4.9950806 ,
            -5.24962116,  -5.48500697,  -5.69885287,  -5.88868538,
            -6.05194146,  -6.18596733,  -6.28801727,  -6.35525239,
            -6.38473948,  -6.37344974,  -6.31825765,  -6.21593972,
            -6.06317329,  -5.85653537,  -5.59250138,  -5.267444  ,
            -4.87763193,  -4.41922872,  -3.88829153,  -3.28076997,
            -2.59250489,  -1.81922713,  -0.9565564 ,   0.        ])




```python
plt.figure(figsize = (5, 10))

t = np.linspace(0, 2 * np.pi, 200)

plt.subplot(3, 1, 1)
plt.plot(np.sin(t), np.sin(t) * np.cos(t), 'b')
plt.title("Figure 8")

plt.subplot(3, 1, 2)
plt.plot(np.sin(t) + 2 * np.sin(2 * t), np.cos(t) -2 * np.cos(2 * t), 'b')
plt.title("Trefoil knot")

t = np.linspace(0, 12 * np.pi, 2000)
plt.subplot(3, 1, 3)
plt.plot(np.sin(t) * (np.exp(np.cos(t)) - 2 * np.cos(4 * t) - np.sin(t / 12) ** 5),
         np.cos(t) * (np.exp(np.cos(t)) - 2 * np.cos(4 * t) - np.sin(t / 12) ** 5),
                    'b')
plt.title("Trefoil knot")

plt.tight_layout()
plt.show()
```


    
![png](/assets/img/blog/Matplotlib_files/Matplotlib_19_0.png)
    



```python

```
