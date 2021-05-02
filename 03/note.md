<!--
Filename:	note.md
Project:	~/Developer/PyML3/03
Authors:	shumez <https://github.com/shumez>
Created:	2020-01-02 13:50:23
Modified:	2020-01-13 14:32:43
-----
Copyright (c) 2020 shumez
-->

# 03. A Tour of Machine Learning Classifiers Using Schikit-Learn

## Contents

- [03.00. Intro][0300]
- [03.01. Choosing a classification algorithm][0301]
- [03.02. First steps with scikit-learn: training a perceptron][0302]
- [03.03. Modeling class probabilities via logistic regression][0303]
    - [03.03.01. Logistic regression intuition & conditional probabilities][030301]
    - [03.03.02. Learning the weights of the logistic cost function][030302]
    - [03.03.03. Converting an Adaline implementation into an ][030303]
    - [03.03.04. Training a logistic regression model with scikit-learn][030304]
    - [03.03.05. Tackling overfitting via regularization][030305]
- [03.04. Maximum margin classifcation with support vector machines][0304]
    - [03.04.01. Maximum margin intuition][030401]
    - [03.04.02. Dealing with the nonlinearly separable case using slack variables][030402]
    - [03.04.03. Alternative implementations in scikit-learn][030403]
- [03.05. Solving non-linear problems using a kernel SVM][0305]
    - [03.05.01. Kernel methods for linearly inseparable data][030501]
    - [03.05.02. Using the kernel trick to find separating hyperplanes in higher dimensional space][030502]
- [03.06. Decision tree learning][0306]
    - [03.06.01. Maximizing information gain  getting the most bang for the buck][030601]
    - [03.06.02. Building a decision tree][030602]
    - [03.06.03. Combining weak to strong learners via random forests][030603]
- [03.07. K-nearest neighbors a lazy learning algorithm][0307]
- [03.08. Summary][0308]


## 03.00 Intro

1. algorithms for classification:
    - logistic regression
    - support vector machines
    - decision tree
2. sklearn ML library
3. strength / weakness of classifiers w/ linear & nonlinear decision boudaries


## 03.01. Choosing a classification algorithm

---

**no free lunch theorem**:  

* あらゆるシナリオに最適な1つのclassifierは存在しない
* by David H. Wolpert, William G. Macrredy

---
  

1. feat を選択, training sample を収集
2. 性能指標を選択
3. classifier, 最適化 algorithm を選択
4. model の性能を評価
5. algorithm を調整


## 03.02. First steps with scikit-learn: training a perceptron

```py
from sklearn import datasets
import numpy as np

# load Iris dataset
iris = datasets.load_iris()
# 3,4 col
X = iris.data[:, [2, 3]]
# class label 
y = iris.target
# output class label
print('Class labels:', np.unique(y))
```


```py
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.3, random_state=1, stratify=y)
```

`train_test_split()`

- `random_state` 
- `stratify=y`: 層化サンプリング, training subset, test subsetのlabelの比率がdatasetを同じ


scaling: `preprocessing` moduleの `StandardScaler` classを使って

```py
from sklearn.preprocessing import StandardScaler 
sc = StandardScaler()

# mean, sd 
sc.fit(X_train)

X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
```

* instantialize `StandardScaler()` class 
* `.fit()` method: calc mean & sd
* `.transform()` method: standardize


```py
from sklearn.linear_model import Perceptron
# Epochs 40; Learning rate 0.1
ppn = Perceptron(n_iter=40, eta0=0.1, random_state=1)
# fit to training data
ppn.fit(X_train_std, y_train)
```

```py
# predict test data
y_pred = ppn.predict(X_test_std)
# error samples 
print('Misclassified samples: %d' % (y_test != y_pred).sum())
```

```py
from sklearn.metrics import accuracy_score
# accuracy
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
```

```py
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
    
    # marker, colormap
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'grey', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    # plot decision region
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    # grid point 
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                          np.arange(x2_min, x2_max, resolution))
    # features -> 1d-array
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    # -> data size
    Z = Z.reshape(xx1.shape)
    
    # plot: grid point 
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    # axis limit 
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    # plot: samples per class 
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                   alpha=0.8,
                   c=colors[idx],
                   marker=markers[idx],
                   label=cl,
                   edgecolor='black')
        
    # test sample
    if test_idx:
        # plot all samples
        X_test, y_test = X[test_idx, :], y[test_idx]
        plt.scatter(X_test[:, 0], X_test[:, 1],
                   c='',
                   edgecolor='black',
                   alpha=1.0,
                   linewidth=1,
                   marker='o',
                   s=100,
                   label='test set')
```

[![fig.3.1][fig_03_01]][fig_03_01]


## 03.03. Modeling class probabilisties via logistic regression

**logistic regression**:   
linear & binary classification problems; NOT regression


### 03.03.01. Logistic regression intuition and conditional probabilities


```
Logistic regression for multiple classes

https://sebastianraschka.com/pdf/lecture-notes/stat479ss19/L05_gradient-descent_slides.pdf or http://rasbt.github.io/mlxtend/user_guide/classifier/SoftmaxRegression/
```

**odds ratio** \( \frac{p}{(1-p)} \)

**logit fn**

\[ \text{logit}(p) = \log{\frac{p}{1-p}} \]

\[ \text{logit}(p(y = 1|x)) = w_0 x_0 + w_1 x_1 + \cdots + w_m x_m = \sum_{i=0}^m w_i x_i = w^T x \]

**Logistic sigmoid fn** (sigmoid fn): logistic fn の inverse fn

\[ \phi(z) = \frac{1}{1 + e^{-z}} \tag{3.3.3} \]

\[ 
    \begin{align*} 
        z 
        &= w^T x \\
        &= w_0 x_0 + w_1 x_1 + \cdots + w_m x_m 
    \end{align*}
    \tag{3.3.4} 
\]

[![fig.3.2][fig_03_02]][fig_03_02]

[![fig.3.3][fig_03_03]][fig_03_03]


\[ 
    \hat{y} = 
    \begin{cases} 
        1 & (\phi(z) ≥ 0.5) \\ 
        0 & (\phi(z) < 0.5) 
    \end{cases} 
    \tag{3.3.5} 
\]

\[ 
    \hat{y} = 
    \begin{cases} 
        1 & (z ≥ 0.0) \\ 
        0 & (z < 0.0) 
    \end{cases} 
    \tag{3.3.6} 
\]


### 03.03.02. Learning the weights of the logistic cost function

ADALINE では, Cost fn は SSE

SSE
\[ J(w) = \sum_i \frac{1}{2} \Big( \phi(z^{(i)}) - y^{(i)} \Big)^2 \tag{3.3.7}\]

\(L\) Likelihood 

\[ 
    \begin{align*}
        L(w) 
        &= P(y | x;w) \\
        &= \prod_{i=1}^n P(y^{(i)} | x^{(i)};w) \\
        &= \prod_{i=1}^n \Big( \phi(z^{(i)})^{y^{(i)}} \Big) \Big( 1 - \phi(z^{(i)}) \Big)^{1-y^{(i)}} 
    \end{align*}
    \tag{3.3.8} 
\]

**log-likelihood fn**

\[ 
    \begin{align*}
        l(w) 
        &= \log{L(w)} \\
        &= \sum_{i=1}^n \Big[ y^{(i)} \log{\big( \phi(z^{(i)}) \big)} + \big( 1 - y^{(i)} \big) \log{ \big( 1 - \phi(z^{(i)}) \big) } \Big] 
    \end{align*}
    \tag{3.3.9} 
\]


cost fn \(J\) として, log-likelihood fn を書き直す

\[ J(w) = \sum_{i=1}^n \Big[ -y^{(i)} \log{\big( \phi(z^{(i)}) \big)} - \big( 1 - y^{(i)} \big) \log{\big( 1 - \phi(z^{(i)}) \big)} \Big] \tag{3.3.10} \]

\[ J(\phi(z), y; w) = -y \log{ \big( \phi(z) \big)} - \big( 1-y \big) \log{ \big( 1-\phi(z) \big)} \tag{3.3.11} \]

\[ J(\phi(z), y; w) = 
    \begin{cases}
        - \log{(\phi(z))} &(y=1) \\
        - \log{(1 - \phi(z))} &(y=0) 
    \end{cases} \tag{3.3.12} \]

[![fig.3.4][fig_03_04]][fig_03_04]


### 03.03.03. 

ADALINE の cost fn を新しいcost fnに置換

\[ J(w) = - \sum_i \Big[ y^{(i)} \log{\big( \phi(z^{(i)}) \big)} + \Big( 1 - y^{(i)} \Big) \log{\Big( 1 - \phi\big( z^{(i)} \big) \Big)} \Big] \tag{3.3.13} \]


[LogisticRegressionGD]

[![fig.3.5][fig_03_05]][fig_03_05]


---
Logistic regression での GDに基づくlearning algorithm
 
\[ 
    \frac{\partial}{\partial w_j} l(w) 
    = \Big( y\frac{1}{\phi(z)} - (1 - y) \frac{1}{1 - \phi(z)} \Big) \frac{\partial}{\partial w_j} \phi(z) 
    \tag{3.3.14} 
\]

\[ 
    \begin{align*}
        \frac{\partial}{\partial z} \phi(z) 
        &= \frac{\partial}{\partial z} \frac{1}{1 + e^{-z}} \\
        &= \frac{1}{(1 + e^{-z})^2} e^{-z} \\
        &= \frac{1}{1+e^{-z}} \big( 1 - \frac{1}{1+e^{-z}} \big) \\
        &= \phi(z) (1 - \phi(z))
    \end{align*} 
    \tag{3.3.15} 
\]

\[ 
    \begin{align*}
        \Big( y\frac{1}{\phi(z)} - (1-y)\frac{1}{1 - \phi(z)} \Big) \frac{\partial}{\partial w_j} \phi(z) 
        &=  \Big( y\frac{1}{\phi(z)} - (1-y)\frac{1}{1 - \phi(z)} \Big) \phi(z) (1 - \phi(z)) \frac{\partial}{\partial w_j} z \\
        &= \Big( y(1 - \phi(z)) - (1 - y) \phi(z) \Big) x_j \\
        &= (y - \phi(z)) x_j
    \end{align*}
    \tag{3.3.16}
\]

---


### 03.03.04. Training a logistic regression model with scikit-learn

[![fig.3.6][fig_03_06]][fig_03_06]

`C=100.0`

```py
lr.predict_proba(X_test_std[:3, :])
```

```py
lr.predict_proba(X_test_std[:3, :]).argmax(axis=1)
# >>> array([2, 0, 0])
```

```py
lr.predict(X_test_std[:3, :])
# >>> array([2, 0, 0])
```

```py
lr.predict(X_test_std[0, :].reshape(1, -1))
# >>> array([2])
```

`reshap()` method 次元を追加


### 03.03.05. Tacking overfitting via regularization

- Underfitting: high bias
- Overfitting: high variance

[![fig.3.7][fig_03_07]][fig_03_07]

**Colinearity** (共線性): featsの間の相関の高さ

**Regularization** (正則化)

L2 regularization (L2 shrinkage / weight decay (荷重減衰))

\[ \frac{\lambda}{2} ||w||^2 := \frac{\lambda}{2} \sum_{j=1}^m w_j^2 \tag{3.3.21} \]

- \(\lambda\): regularization param

\[
    J(w) = \sum_{i=1}^{m} [-y^{(i)} \log{(\phi(z^{(i)}))} - (1 - y^{(i)}) \log{(1 - \phi(z^{(i)}) )} ] + \frac{\lambda}{2} ||w||^2 \tag{3.3.22}
\]

`LogisticRegression` class の param `C`  は
\[ C := \frac{1}{\lambda} \tag{3.3.23} \]

\[
    J(w) = C \sum_{i=1}^{n} \Big[ -y^{(i)} \log{(\phi(z^{(i)}))} - (1 - y^{(i)}) \log{(1 - \phi(z^{(i)}))} \Big] + \frac{1}{2} ||w||^2 \tag{3.3.24}
\]


[![fig.3.8][fig_03_08]][fig_03_08]

[Menard, S., 2009]


## 03.04. Maximum margin classifcation with support vector machines

**Support Vector Machine** SVM

**margin** (Hyperplane(超平面)に最も近いtraining sampleとの距離) を最大化
**support vec**: hyperplane に最も近い training sample

[![fig.3.9][fig_03_09]][fig_03_09]

### 03.04.01. Maximum margin intuition

**positive**, **negative** 

\[ w_0 + w^T x_{pos} = 1 \tag{3.4.1} \]

\[ w_0 + w^T x_{neg} = -1 \tag{3.4.2} \]

\[ w^T (x_{pos} - x_{neg}) = 2 \tag{3.4.3} \]

vecの長さを定義
\[ ||w|| := \sqrt{ \sum_{j=1}^m w_j^2 } \tag{3.4.4} \]

\[ \frac{w^T (x_{pos} - x_{neg})}{||w||} = \frac{2}{||w||} \tag{3.4.5} \]

\[ \begin{cases} 
    w_0 + w^T x^{(i)} ≥ 1 & (y^{(i)} = 1) \\ 
    w_0 + w^T x^{(i)} ≤ -1 & (y^{(i)} = -1) 
\end{cases} \\ i = 1 \cdots N \tag{3.4.6} \]

- \(N\): dataset のsample数

\[ y^{(i)} (w_0 + w^Tx^{(i)}) ≥ 1 \forall_i \tag{3.4.7} \]

\(\frac{2}{||w||}\) 

[Vapnik, V., 2013]
[Burges, C. J., 1998]


### 03.04.02. Dealing with the nonlinearly separable case using slack variables


**soft-margin classification**

\(\xi\): slack var, 線形分離不能なデータのためんい, 線形制約を緩和するため ([Vapnik, V., 1995])

\[ 
    \begin{cases}
        w_0 + w^Tx^{(i)} ≥ 1 - \xi^{(i)} & (y^{(i)} = 1) \\
        w_0 + w^Tx^{(i)} ≤ -1 + \xi^{(i)} & (y^{(i)} = -1)
    \end{cases} \\
    i = 1 \cdots N \tag{3.4.8}
\]

\[ \frac{1}{2} ||w||^2 + C \Big( \sum_i \xi^{(i)} \Big) \tag{3.4.9} \]

- \(N\): # of sample 

[![fig.3.10][fig_03_10]][fig_03_10]

```py
from sklearn.svm import SVC
# linear SVM 
svm = SVC(kernel='linear', C=1.0, random_state=1)
# fit to training data
svm.fit(X_train_std, y_train)
plot_decision_regions(
    X_combined_std, 
    y_combined, 
    classifier=svm,
    test_idx=range(105, 150))

ax.set_xlabel('petal length [standardized]')
ax.set_ylabel('petal width [standardized]')
ax.legend(loc='upper left')
plt.tight_layout()
plt.show()
```

[![fig.3.11][fig_03_11]][fig_03_11]


> Logistic regression & SVM
> 
> lr: 
> - likelihood (尤度) を最大化
> - 外れ値に影響されやすい
> 
> SVM: 
> - 決定領域に関心
> 

### 03.04.03. Alternative implementations in scikit-learn

SVM: 

[LIBLINEAR]
[LIBSVM]

```py
from sklearn.linear_model import SGDClassifier
# stochastic GD perceptron
ppn = SGDClassifier(loss='perceptron')
# stochastic GD Logistic reg
lr = SGDClassifier(loss='log')
# stochastic GD SVM
svm = SGDClassifier(loss='hinge')
```


## 03.05. Solving non-linear problems using a kernel SVM

**kernel SVM**


### 03.05.01 

[![fig.03.12][fig_03_12]][fig_03_12]

projection fn (射影) \(\phi\)

\[ \phi(x_1, x_2) = (z_1, z_2, z_3) = (x_1, x_2, x_1^2 + x_2^2) \]

[![fig.3.13][fig_03_13]][fig_03_13]


### 03.05.02. Using the kernel trick to find separating hyperplanes in higher dimensional space

\[ \mathcal{K}(x^{(i)}, x^{(j)}) = \phi(x^{(i)})^T \phi(x^{(j)}) \tag{3.5.2} \]

**Radial Basis Function kernel**, RBF a.k.a., Gaussian (動径基底関数)

\[ \mathcal{K}(x^{(i)}, x^{(j)}) = \exp{\bigg( - \frac{||x^{(i)} - x^{(j)} ||^2}{2\sigma^2} \bigg)} \tag{3.5.3} \]

\[ \mathcal{K}(x^{(i)}, x^{(j)}) = \exp{\big( - \gamma ||x^{(i)} - x^{(j)} ||^2 \big)} \tag{3.5.4} \]

\( \gamma = \frac{1}{2\sigma^2} \): hyperplameter

**kernel**: 類似性を表すfn

- 1: 全く同じsample
- 0: 全く異なるsample

- `kernel='rbf`
- `gamma=0.1`

[![fig.03.14][fig_03_14]][fig_03_14]

```py
# RBF kernel のよる SVM の instance を生成 (2つのparameter を変更)
svm = SVC(kernel='rbf', random_state=1, gamma=0.2, C=1.0)
svm.fit(X_train_std, y_train)
plot_decision_regions(
    X_combined_std, 
    y_combined, 
    classifier=svm,
    test_idx=range(105, 150))
plt.xlabel('Petal Length [Standardized]')
plt.ylabel('Petal Width [Standardized ]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()
```

[![fig.3.15][fig_03_15]][fig_03_15]

\(\gamma\) を大きく
`gamma=100.0`

[![fig.3.16][fig_03_16]][fig_03_16]


## 03.06. Decision tree learning

**decision tree** classifier: interpretability (意味解釈可能性)に配慮する場合にいいモデル

[![fig.3.17][fig_03_17]][fig_03_17]

**information gain** (分割された集合の要素についてのばらつきの減少)が最大となるfeatures でデータを分割する

### 03.06.01. Maximizing information gain  getting the most bang for the buck

\[ \text{IG}(D_p, f) = \text{I}(D_p) - \sum_{j=1}^m  \frac{N_j}{N_p} \text{I}(D_j) \tag{3.6.1} \]

- \(f\): 分割を行うfeat
- \(D_p\): 親のdataset
- \(D_j\): \(j\)-th dataset
- \(\text{I}\): Impurity 不純度
- \(N_p\): 親nodeのsample 総数
- \(N_j\): \(j\)-th node sample数

IG は親nodeのimpurity と 子nodeのimpurityの合計の差

単純化のため \(m=2\), \(D_{left}\), \(D_{right}\)

\[ \text{IG}(D_p, f) = \text{I}(D_p) - \frac{N_{left}}{N_p} \text{I}(D_{left}) - \frac{N_{right}}{N_p} \text{I}(D_{right}) \tag{3.6.2} \]

binary DT でよく使われる imurity index  

- **Gini impurity** \(I_G\)
- **entropy** \(I_H\)
- **classification error** 分類誤差 \(I_E\)


\[ I_H(t) := -\sum_{i=1}^c p(i|t) \log_2{p(i|t)} \tag{3.6.3} \]

- \(p(i|t)\): node\(t\) において class\(i\) に属するsampleの割合

[![entropy][fig_entropy]][fig_entropy]

\[ 
    \begin{align*}
        I_G(t) 
        &= \sum_{i=1}^c p(i|t)(1 - p(i|t)) \\
        &= 1 - \sum_{i=1}^c p(i|t)^2
    \end{align*} 
    \tag{3.6.4} 
\]

[![Gini][fig_Gini]][fig_Gini]

classification error

\[ I_E(t) = 1 - max[p(i|t)] \tag{3.6.6} \]

[![fig.3.19][fig_03_19]][fig_03_19]


### 03.06.02. Building a decision tree

```py
tree = DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=1)
```

[![fig.3.20][fig_03_20]][fig_03_20]


[![tree][tree]][tree]


### 03.06.03. Combining weak to strong learners via random forests

**Random forest** 

random forest algolithm: variance が高い, 複数のDTを平均化することで, よる汎化性能が高い頑健なmodelを構築

1. size n の random な bootstrap sample を復元抽出 (training data  から nコの sample を random に選択)
2. bootstrap sample から DT を成長させる: 各 node で
    1. dコの features を random に非復元抽出
    2. IG を最大化することにより, 目的fn に従って, 最適な分割となる features を使って node を分割
3. 1-2 をk回繰り返す
4. DT ごとの予測をまとめて, "多数決" にもとづいて, class label を割当て

\( d = \sqrt{m} \)

- \(d\): feats
- \(m\): training dataset feats

```py
forest = RandomForestClassifier(
    criterion='gini', 
    n_estimators=25, 
    random_state=1, 
    n_jobs=2)
```

`n_estimators`: 
`n_jobs`: 

[![fig.3.22][fig_03_22]][fig_03_22]


## 03.07. K-nearest neighbors: a lazy learning algorithm

**k-nearest neighbor classifier** (**KNN**)

---

ML algorithm:

- **Parametric model**
    - training datasets から params を推定
    - e.g., perceptron, logistic reg, linear SVM
- **Non-parametric model**
    - params の個数はtraining datasetsとともに増加
    - e.g., decision tree / random forest, kernel SVM, KNN

---

KNN

1. k の値と距離指標を選択
2. 分類したい sample から kコの最近傍の data point を見つける
3. 多数決により class label を割当てる

[![fig.3.23][fig_03_23]][fig_03_23]


```py
knn = KNeighborsClassifier(
    n_neighbors=5, 
    p=2, 
    metric='minkowski')
```

[![fig.3.24][fig_03_24]][fig_03_24]


**minlowski**: 

\[ d(x^{(i)}, x^{(j)}) := \sqrt[p]{ \sum_k|x_k^{(i)} - x_k^{(j)}|^p } \tag{3.7.1} \]

\(p=2\) のときは, Euclidean distance, \(p=1\) のときは, Manhattan distance


##
<!-- toc -->
[0301]: #0301-choosing-a-classification-algorithm
[0302]: #0302_first_steps_with_scikit-learn_training_a_perceptron
[0303]: #0303_modeling_class_probabilisties_via_logistic_regression
[030301]: #030301_logistic_regression_intuition_conditional_probabilities
[030302]: #030302_learning_the_weights_of_the_logistic_cost_function
[030303]: #030303
[030304]: #030304_training_a_logistic_regression_model_with_scikit-learn
[030305]: #030305_tacking_overfitting_via_regularization
[0304]: #0304_maximum_margin_classifcation_with_support_vector_machines
[030401]: #030401_maximum_margin_intuition
[030402]: #030402_dealing_with_the_nonlinearly_separable_case_using_slack_variables
[030403]: #030403_alternative_implementations_in_scikit-learn
[0305]: #0305_solving_non-linear_problems_using_a_kernel_svm
[030501]: #030501
[030502]: #030502_using_the_kernel_trick_to_find_separating_hyperplanes_in_higher_dimensional_space
[0306]: #0306_decision_tree_learning
[030601]: #030601_maximizing_information_gain_getting_the_most_bang_for_the_buck
[030602]: #030602_building_a_decision_tree
[030603]: #030603_combining_weak_to_strong_learners_via_random_forests
[0307]: #0307_k-nearest_neighbors_a_lazy_learning_algorithm
[0308]: #0308-summary

<!-- link -->
[Menard, S., 2009]: https://books.google.co.jp/books?id=JSJzAwAAQBAJ "Menard, S., 2009. Logistic Regression: From Introductory to Advanced Concepts and Applications."
[Vapnik, V., 2013]: https://books.google.co.jp/books?hl=en&lr=&id=EqgACAAAQBAJ&oi=fnd&pg=PR7&dq=Vladimir+Vapnik,+The+Nature+of+Statistical+Learning+Theory&ots=g3F1ixbX34&sig=SA2N56D24_pBHMLBQSPcoc42mNw#v=onepage&q=Vladimir%20Vapnik%2C%20The%20Nature%20of%20Statistical%20Learning%20Theory&f=false "Vapnik, V. (2013). The nature of statistical learning theory. Springer science & business media."
[Burges, C. J., 1998]: http://www.di.ens.fr/~mallat/papiers/svmtutorial.pdf "Burges, C. J. (1998). A tutorial on support vector machines for pattern recognition. Data mining and knowledge discovery, 2(2), 121-167."
[Vapnik, V., 1995]: http://image.diku.dk/imagecanon/material/cortes_vapnik95.pdf
[LIBLINEAR]: https://www.csie.ntu.edu.tw/~cjlin/liblinear/ "LIBLINEAR - A Library for Large Linear Classification"
[LIBSVM]: https://www.csie.ntu.edu.tw/~cjlin/libsvm/ "LIBSVM - A Library for Support Vector Machines"

<!-- fig -->
[fig_03_01]: https://raw.githubusercontent.com/shumez/PyML/master/03/img/fig_03_01.png "fig.3.1"
[fig_03_02]: https://raw.githubusercontent.com/shumez/PyML/master/03/img/fig_03_02.png "fig.3.2"
[fig_03_03]: https://raw.githubusercontent.com/shumez/PyML/master/03/img/03_03.png "fig.3.3"
[fig_03_04]: https://raw.githubusercontent.com/shumez/PyML/master/03/img/fig_03_04.png "fig.3.4"
[fig_03_05]: https://raw.githubusercontent.com/shumez/PyML/master/03/img/03_05.png "fig.3.5"
[fig_03_06]: https://raw.githubusercontent.com/shumez/PyML/master/03/img/fig_03_06.png "fig.3.6"
[fig_03_07]: https://raw.githubusercontent.com/shumez/PyML/master/03/img/03_07.png "fig.3.7"
[fig_03_08]: https://raw.githubusercontent.com/shumez/PyML/master/03/img/fig_03_08.png "fig.3.8"
[fig_03_09]: https://raw.githubusercontent.com/shumez/PyML/master/03/img/03_09.png "fig.3.9"
[fig_03_10]: https://raw.githubusercontent.com/shumez/PyML/master/03/img/03_10.png "fig.3.10"
[fig_03_11]: https://raw.githubusercontent.com/shumez/PyML/master/03/img/fig_03_11.png "fig.3.11"
[fig_03_12]: https://raw.githubusercontent.com/shumez/PyML/master/03/img/fig_03_12.png
[fig_03_13]: https://raw.githubusercontent.com/shumez/PyML/master/03/img/03_13.png "fig.3.13"
[fig_03_14]: https://raw.githubusercontent.com/shumez/PyML/master/03/img/fig_03_14.png "fig.3.14"
[fig_03_15]: https://raw.githubusercontent.com/shumez/PyML/master/03/img/fig_03_15.png "fig.3.15"
[fig_03_16]: https://raw.githubusercontent.com/shumez/PyML/master/03/img/fig_03_16.png "fig.3.16"
[fig_03_17]: https://raw.githubusercontent.com/shumez/PyML/master/03/img/03_17.png "fig.3.17"
[fig_entropy]: https://raw.githubusercontent.com/shumez/PyML/master/03/img/fig_entropy.png "fig.entropy"
[fig_Gini]: https://raw.githubusercontent.com/shumez/PyML/master/03/img/fig_Gini.png "fig.Gini"
[fig_03_19]: https://raw.githubusercontent.com/shumez/PyML/master/03/img/03_19.png "fig.3.19"
[fig_03_20]: https://raw.githubusercontent.com/shumez/PyML/master/03/img/03_20.png "fig.3.20"
[tree]: https://raw.githubusercontent.com/shumez/PyML/master/03/img/tree.png "fig.tree"
[fig_03_22]: https://raw.githubusercontent.com/shumez/PyML/master/03/img/03_22.png "fig.3.22"
[fig_03_23]: https://raw.githubusercontent.com/shumez/PyML/master/03/img/03_23.png "fig.3.23"
[fig_03_24]: https://raw.githubusercontent.com/shumez/PyML/master/03/img/03_24.png "fig.3.24"

<!-- code -->
[LogisticRegressionGD]: https://raw.githubusercontent.com/shumez/PyML/master/03/code/LogisticRegressionGD.py

<style type="text/css">
	img{width: 55%; float: right;}
</style>