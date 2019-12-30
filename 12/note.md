<!--
Filename:	note.md
Project:	/Users/shume/Developer/PyML3/12
Authors:	shumez <https://github.com/shumez>
Created:	2019-12-21 11:07:53
Modified:	2019-12-30 14:18:35
-----
Copyright (c) 2019 shumez
-->

# 12. Implementing a Multilayer Artificial Neural Network from Scratch

## Contents

- 12.01. Modeling complex functions with artificial neural entworks
    - 12.01.01. Single-layer neural network recap
    - 12.01.02. Introducing the multilayer neural network architecture
    - 12.01.03. Activating a neural network via forward propagation
- 12.02. Classifying handwritten digits
    - 12.02.01 Obtaining and preparing the MNIST dataset
    - 12.02.02. Implementing a multilayer perceptron
- 12.03. Training an artificial neural network
    - 12.03.01. Computing the  logistic cost function
    - 12.03.02. Developing your understanding of backpropagation
    - 12.03.03. Training neural networks via backpropagation
- 12.04. About the convergence in neural networks
- 12.05. A few last words about the neural network implementation
- 12.06. Summary


## 12.01. Modeling complex functions with artificial neural entworks

[A logical calculus of the ideas immanent in nervous activity, W. S. McCulloch and W. Pitts. The Bulletin of Mathematical Biophysics, 5(4):115–133, 1943.](http://aiplaybook.a16z.com/reference-material/mcculloch-pitts-1943-neural-networks.pdf)


**McCulloch-Pitts neuron model**

[Learning representations by back-propagating errors, D. E. Rumelhart, G. E. Hinton, R. J. Williams, Nature, 323 (6088): 533–536, 1986).](http://www.cs.toronto.edu/~hinton/absps/naturebp.pdf)

[AI winter](https://en.wikipedia.org/wiki/AI_winter)


- [Evans, R., Jumper, J., Kirkpatrick, J., Sifre, L., Green, T., Qin, C., ... & Petersen, S. (2018). De novo structure prediction with deeplearning based scoring. Annu Rev Biochem, 77(363-382), 6.][]
- [Esteva, A., Kuprel, B., Novoa, R. A., Ko, J., Swetter, S. M., Blau, H. M., & Thrun, S. (2017). Dermatologist-level classification of skin cancer with deep neural networks. Nature, 542(7639), 115.][]
- [Henaff, M., Canziani, A., & LeCun, Y. (2019). Model-predictive policy learning with uncertainty regularization for driving in dense traffic. arXiv preprint arXiv:1901.02705.][2019_HenaffMikael]


### 12.01.01. Single-layer neural network recap

ADAaptive LInear NEuron (ADALINE)

![fig.12.1][fig1201]

\( w := w + \Delta w \), 
where \( \Delta w = - \eta \nabla J(w) \)

**sum of squared errors (SSE)**

\( \frac{\partial}{\partial w_j} J(w) = - \sum_i (y^{(i)} - a^{(i)}) x^{(i)}_j \)

\(y^{(i)}\) target class label of a particular sample \(x^{(i)}\)  

\(a^{(i)}\) activation 

activation fn \(\phi\)  
\( \phi(z) = z = a \)

net input \(z\)  
\( z = \sum_j{w_j x_j} = \bf{w}^T x \)

threshold  
\( \hat{y} = \begin{cases} 1 & \text{if } g(z) ≥ 0 \\ -1 & \text{otherwise} \end{cases} \)

**stochastic gradient discent SGD**


### 12.01.02. Introducing the multilayer neural network architecture

![][fig1202]

---
Adding additional hidden layers

---

\(i\)th activation unit in the \(l\)th layer as \(a^{(i)}_l\)

\(a^{(\text{in})} = \begin{bmatrix} a^{(\text{in})}_0 \\ a^{(\text{in})}_1 \\ \vdots \\ a^{(\text{in})}_m \end{bmatrix} = \begin{bmatrix} 1 \\ x^{(\text{in})}_1 \\ \vdots \\ x^{(\text{in})}_m  \end{bmatrix} \)

---
Notational convention for the bias units


---

\(w^{(l)}_{k,j}\)

one-versus-all OvA

\( 0 = \begin{bmatrix} 1 \\ 0 \\ 0 \end{bmatrix}, 1 = \begin{bmatrix} 0 \\ 1 \\ 0 \end{bmatrix}, 2 = \begin{bmatrix} 0 \\ 0 \\ 1 \end{bmatrix} \)

\( W^{(h)} \in \mathbb{R}^{m \times d} \)

the number of hidden units \(d\), the number of input units \(m\)

![fig.12.3][fig1203]


### 12.01.03. Activating a neural network via forward propagation

**forward propagation**

1. forward propagate
2. calculate error, mimimize using cost fn
3. backpropagate the error

unit of hidden layer \( a^{(h)}_1 \)

\( z^{(h)}_1 = a^{(in)}_0 w^{(h)}_{0,1} + a^{(in)}_1 w^{(h)}_{1,1} + \cdots +a^{(in)}_m w^{(h)}_{m,1} \)
\( a^{(h)}_1 = \phi(z^{(h)}_1) \)

net input \( z^{(h)}_1 \)  
activation fn \( \phi(\cdot) \)

sigmoid fn

\( \phi(z) = \frac{1}{1 + e^{-z}} \)

![fig.12.04][fig1204]

\[ z^{(h)}  = \mathbf{a}^{(in)} \mathbf{W^{(h)}} \]
\[ a^{(h)} = \phi(z^{(h)}) \]

\(a^{(in)}\): \( 1 \times m \) dim; \(x^{(in)}\) + bias unit  
- \(w^{(h)}\): \(m \times d\) dim; weight mat
- \(d\): # of units in the hidden layer
- \(z^{(h)}\): \(1 \times d \) dim; net input vec

generalize
\[ Z^{(h)} = A^{(in)} W^{(h)} \]

- \( A^{(in)} \): \( n \times m \) mat  
- \( Z^{(h)} \): \( n \times d \) dim, net input mat

\[ A^{(h)} = \phi(Z^{(h)}) \]

\[ Z^{(out)} = A^{(h)} W^{(out)} \]

- \( W^{(out)} \): \( d \times t \) mat
- \( A^{(h)} \): \( n \times d \) dim mat
- \( Z^{(out)} \): \( n \times t \) dim mat

\[ A^{(out)} = \phi(Z^{(out)}), \, A^{(out)} \in \mathbb{R}^{n \times t} \]


## 12.02. Classifying handwritten digits
### 12.02.01 Obtaining and preparing the MNIST dataset
### 12.02.02. Implementing a multilayer perceptron
## 12.03. Training an artificial neural network
### 12.03.01. Computing the  logistic cost function
### 12.03.02. Developing your understanding of backpropagation
### 12.03.03. Training neural networks via backpropagation
## 12.04. About the convergence in neural networks
## 12.05. A few last words about the neural network implementation
## 12.06. Summary

##
<!-- toc -->
[01]: #
[0101]: #

<!-- ref -->

<!-- fig -->
[fig1201]: https://raw.githubusercontent.com/rasbt/python-machine-learning-book-3rd-edition/master/ch12/images/12_01.png
[fig1202]: https://raw.githubusercontent.com/rasbt/python-machine-learning-book-3rd-edition/master/ch12/images/12_02.png
[fig1203]: https://raw.githubusercontent.com/rasbt/python-machine-learning-book-3rd-edition/master/ch12/images/12_03.png
[fig1204]: fig/1204.png

<!-- term -->

<style type="text/css">
	img{width: 51%; float: right;}
</style>