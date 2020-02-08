<!--
Filename:	note.md
Project:	/Users/shume/Developer/PyML3/15
Authors:	shumez <https://github.com/shumez>
Created:	2019-12-17 14:24:36
Modified:	2020-02-08 14:55:34
-----
Copyright (c) 2019 shumez
-->

# 15. Classifying Images with Deep Convolutional Neural Networks

## Contents

- [15.01. The building blocks of CNNs][1501]
    - [15.01.01. Understanding CNNs and features hierachies][150101]
    - [15.01.02. Performing discrete convolutions][150102]
        - [15.01.02.01. Discrete convolutions in one dimension][15010201]
        - [15.01.02.02. Padding inputs to control the size of the output feature maps][15010202]
        - [15.01.02.03. Determining the size of the convolution output][15010203]
        - [15.01.02.04. Performing a discrete convoution in 2D][15010204]
    - [15.01.03. Subsampling layers][150103]
- [15.02. Putting everything together - implementing a CNN][1502]
    - [15.02.01. Working with multiple input or color channels][150201]
    - [15.02.02. Regularizing an NN with dropout][150202]
    - [15.02.03. Loss functions for classification][150203]
- [15.03. Implementing a deep CNN using TensorFlow][1503]
    - [15.03.01. The multilayer CNN architecture][150301]
    - [15.03.02. Loading and preprocessing the data][150302]
    - [15.03.03. Implemeting a CNN using the TensorFlow Keras API][150303]
        - [15.03.03.01. Configuration CNN layers in Keras][15030301]
        - [15.03.03.02. Constructing a CNN in Keras][15030302]
- [15.04. Gender classigcation from face images using][1504]
    - [15.04.01. Loading the CelebA dataset][150401]
    - [15.04.02. Image transformation and data augmentation][150401]
    - [15.04.03. Training a CNN gender classifier][150403]
- [15.05. Summary][1505]


## 15.00. Intro

topics:

1. convolution operation
2. building blocks
3. deep CNNs in TF
4. data augmentation techniques
5. face image-based CNN classifier


## 15.01. The building blocks of CNNs

Yann LeCun, 1990s

[LeCun, Y., Boser, B., Denker, J. S., Henderson, D., Howard, R. E., Hubbard, W., & Jackel, L. D. (1989). Backpropagation applied to handwritten zip code recognition. Neural computation, 1(4), 541-551.][1989_LeCunYann]

---
The human visual cortex

[Hubel, David H., Wissel, Torsten, 1959][1959_HubelDavidH_WisselTorsten]

---


### 15.01.01. Understanding CNNs and features hierachies

**slient (relevant) features**

**low-level features**

**feature hierachy**

feature map

local receptive field

- **sparce connectivity**
- **parameter-sharing**


### 15.01.02. Performing discrete convolutions

**discrete convolution** (**convolution**)

---
Mathematical notation

\( \mathbf{A_{n_1 \times n_2}} \) is \( n_1 \times n_2 \) dim

\( \mathbf{A[i, j]} \) index \(i\), \(j\) of mat \(\mathbf{A}\)

---

#### 15.01.02.01. Discrete convolutions in one dimension

\( y = x \ast w \) 

\( x \): input, signal
\( w \): filter, kernel

\[ \mathbf{y} = \mathbf{x} \ast \mathbf{w} \rightarrow y[i] = \sum_{k = -\infty}^{+\infty}{x[i - k] w[k]} \]

**zero-pading**

\[ \mathbf{y} = \mathbf{x} \ast \mathbf{w} \rightarrow y[i] = \sum_{k=0}^{k=m-1}{x^p[i+m-k] w[k]} \]

![][fig1503]

#### 15.01.02.02. Padding inputs to control the size of the output feature maps

- Full padding
- Valid padding 
- Same padding

#### 15.01.02.03. Determining the size of the convolution output 

\[ o = \bigg\lfloor \frac{n + 2p - m}{s} \bigg\rfloor +1 \]

\[ n = 10, m = 5, p =2, s = 1 \newline \rightarrow o = \bigg\lfloor \frac{10 + 2 \times 2 - 5}{1} \bigg\rfloor + 1 = 10 \]

\[ n=10, m=3, p=2, s=2 \newline \rightarrow o = \bigg\lfloor \frac{10+2\times2-3}{2} \bigg\rfloor +1 = 6 \]

#### 15.01.02.04. Performing a discrete convoution in 2D

\(X_{n_1 \times n_2}\), filter mat \(W_{m_1 \times m_2}\) (\(m_1≤n_1\), \(m_2≤n_2\))
\(\mathbf{Y} = \mathbf{X} * \mathbf{W}\)

\[ \mathbf{Y} = \mathbf{X} * \mathbf{W} \newline \rightarrow Y[i,j] = \sum_{k_1=-\infty}^{+\infty} \sum_{k_2=-\infty}^{+\infty}{ X[i-k_1, j-k_2] W[k_1, k_2] } \]

\[ \mathbf{W^r} = \begin{bmatrix} 0.5 & 1.0 & 0.5\\ 0.1 & 0.4 & 0.3\\ 0.4 & 0.7 & 0.5 \end{bmatrix} \]

``` W_rot = W[::-1,::-1] ```

### 15.01.03. Subsampling layers

\[ \mathbf{X_1} = 
\begin{bmatrix}
10 & 255 & 125 & 0 & 170 & 100 \\
70 & 255 & 105 & 25 & 25 & 70 \\
255 & 0 & 150 & 0 & 10 & 10 \\
0 & 255 & 10 & 10 & 150 & 20 \\
70 & 15 & 200 & 100 & 95 & 0 \\
35 & 25 & 100 & 20 & 0 & 60
\end{bmatrix}, \mathbf{X_2} = 
\begin{bmatrix}
100 & 100 & 100 & 50 & 100 & 50 \\
95 & 255 & 100 & 125 & 125 & 170 \\
80 & 40 & 10 & 10 & 125 & 150 \\
255 & 30 & 150 & 20 & 120 & 125 \\
30 & 30 & 150 & 100 & 70 & 70 \\
70 & 30 & 100 & 200 & 70 & 95
\end{bmatrix} \newline \xrightarrow{\text{max pooling }P_{2\times2}} \begin{bmatrix} 255 & 125 & 170\\ 255 & 150 & 150\\ 70 & 200 & 95 \end{bmatrix} \]

[Springenberg, J. T., Dosovitskiy, A., Brox, T., & Riedmiller, M. (2014). Striving for simplicity: The all convolutional net. arXiv preprint arXiv:1412.6806.](https://arxiv.org/abs/1412.6806)

## 15.02. Putting everything together - implementing a CNN

\[ \mathbf{Z} = \mathbf{W} * \mathbf{X} + \mathbf{b} \]

\( \mathbf{A} = \phi(Z) \)

### 15.02.01. Working with multiple input or color channels

**channels**  2D array / mat w \( N_1 \times N_2 \)  

\( \mathbf{X}_{N_1 \times N_2 \times C_{in}} \)

---

Reading an image fie

---

\[ \text{Given an example } \mathbf{X}_{n_1 \times n_2 \times c_{in'}} \newline \text{a kernel matrix } \mathbf{W}_{m_1 \times m_2 \times c_{in'}} \newline \text{an bias } b \newline \Downarrow \newline \mathbf{Z}^{Conv} = \sum_{c=1}^{c_{in}}{\mathbf{W}[:,:,c] * \mathbf{X}[:,:,c]} \newline \text{Pre-activation: } \mathbf{Z} = \mathbf{Z}^{Conv} + b_c \newline \text{Feature map: } \mathbf{A} = \phi(\mathbf{Z}) \]


![](https://raw.githubusercontent.com/rasbt/python-machine-learning-book-3rd-edition/master/ch15/images/15_09.png)

### 15.02.02. Regularizing an NN with dropout
### 15.02.03. Loss functions for classification

## 15.03. Implementing a deep CNN using TensorFlow

### 15.03.01. The multilayer CNN architecture
### 15.03.02. Loading and preprocessing the data
### 15.03.03. Implemeting a CNN using the TensorFlow Keras API

#### 15.03.03.01. Configuration CNN layers in Keras
#### 15.03.03.02. Constructing a CNN in Keras

## 15.04. Gender classigcation from face images using

### 15.04.01. Loading the CelebA dataset
### 15.04.02. Image transformation and data augmentation
### 15.04.03. Training a CNN gender classifier

## 15.05. Summary


## 




##
<!-- toc -->

<!-- ref -->

<!-- fig -->
[fig1503]: https://raw.githubusercontent.com/rasbt/python-machine-learning-book-3rd-edition/master/ch15/images/15_03.png

<!-- term -->

<style type="text/css">
	img{width: 51%; float: right;}
</style>