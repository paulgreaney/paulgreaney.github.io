---
title: 'Computing Gradient Descent in Neural Networks By Hand'
date: 2023-09-29
permalink: /posts/2023/09/computing-gradient-descent-nn/
tags:
  - gradient descent
  - neural networks
  - computational mathematics
---

# Gradient Descent in Neural Networks

## Introduction

I haven't been able to find an explanation of the computation of updated network weights in a neural network with more than one node in more than one layer, which doesn't involve computing the error and jumping to define a formula using the element-wise product of vectors or matrices. 
Here we do this computation by computing the partial derivatives directly, either in vector or component form.

## Network, Activation Function, and Loss Function

<p align="center">
<img width="350" src="https://raw.githubusercontent.com/paulgreaney/paulgreaney.github.io/master/_posts/nn.png" />
</p>

We take a neural network with an input layer consisting of three nodes, a hidden layer of two nodes, and an output node with one node. For the activation function, we use the sigmoid function $$\sigma(z)=\frac{1}{1+e^{-z}}.$$ The values of the hidden layer $a_j^{(1)}$ are calculated by applying the sigmoid activation function to the product of the weights $w_{i,j}$ with the input values $x_i$.

The mean-squared error loss function for $N$ examples is $$\mathcal{L}=\frac{1}{N}\sum_{i=1}^N (y_i-\hat{y_i})^2,$$ where $\hat{y}$ is the network output of that example at the output node, and $y$ is the  target output (label) for that example.
Suppose we want to train our network to give output $y=1$ for input values $x_1=0$, $x_2=1$, $x_3=2$. For gradient descent, we need to calculate the gradient of the output with respect to the parameters $w_{i,j}^{(k)}$.

We can write the weights between layer $i$ and layer $(i-1)$ as a matrix, so that $$\boldsymbol{a}^{(1)}=\sigma(W^{(1)}\boldsymbol{x}), \quad \hat{y}=\sigma(W^{(2)}\boldsymbol{a}^{(1)}).$$


For convenience later, we define $$\boldsymbol{z}^{(1)}=W^{(1)}\boldsymbol{x}, \quad z^{(2)}=W^{(2)}\boldsymbol{a}^{(1)}.$$

In component form, the first equation here is

$$\left(\begin{matrix}a_1^{(1)}\\ 
a_2^{(1)}\end{matrix}\right) = \sigma\left(\begin{matrix}z_1^{(1)}\\ 
z_2^{(1)}\end{matrix}\right) = \sigma\left(\left(\begin{matrix}w_{1,1}&w_{2,1}&w_{3,1}\\
w_{2,1}&w_{2,2}&w_{2,3}\end{matrix}\right)\left(\begin{matrix}
x_1\\ 
x_2\\ 
x_3\end{matrix}\right)\right)$$

The computation of the second equation follows a similar process.

## Training

Our overall goal is to _train_ our network to make good predictions $\hat{y}$, by finding model parameters (weights) that minimise a loss function for inputs $x$ and labels $y$.
We do this by initialising the weights to some random values, and then using gradient descent to improve them:

$$W^{(i)}\to W^{(i)}-\frac{\partial \mathcal{L}}{\partial W^{(i)}}.$$

## Initialisation and Forward Pass

To initialise the network, take 

$$W^{(1)}=\begin{pmatrix}0.4&0.5&0.3\\ 
0.2&0.7&0.1\end{pmatrix},\quad W^{(2)}=\begin{pmatrix}0.1&0.2\end{pmatrix},$$

$x_1=0$, $x_2=1$, $x_3=2$, and set the target output to $y=1$ for this example.

The first step is to calculate output values at each node in the hidden layer and at the output $\hat{y}$ for the given input values.

We first need to calculate the intermediate values, 

$$ \boldsymbol{z}^{(1)}=W^{(1)}\boldsymbol{x}, \boldsymbol{a}^{(1)}=\sigma(\boldsymbol{z}^{(1)})=\sigma\left(W^{(1)}\boldsymbol{x}\right), \text{ and } \boldsymbol{z}^{(2)}=W^{(2)}\boldsymbol{a}^{(1)},$$

and the output value $\hat{y}=\sigma(\boldsymbol{z}^{(2)}).%=W^{(2)}\boldsymbol{a}^{(1)})=%\sigma\left(W^{(2)}\sigma\left(W^{(1)}\boldsymbol{x}\right)\right).$

## Gradient Computatation

We want to compute the gradient of

$$\mathcal{L} = \frac{1}{N}\sum_{i=1}^N(y_i-\hat{y_i})^2 = \frac{1}{N}\left(\boldsymbol{y}-\hat{\boldsymbol{y}}\right)^T\left(\boldsymbol{y}-\hat{\boldsymbol{y}}\right)
$$

which in the case of one sample reduces to 
$$\mathcal{L}=(y-\hat{y})^2.$$

In order to update the weights using gradient descent, we need to calculate

$$
\dfrac{\partial\mathcal{L}}{\partial W^{(2)}}\text{ and  } \dfrac{\partial\mathcal{L}}{\partial W^{(1)}}.
$$

Applying the chain rule gives

$$\frac{\partial\mathcal{L}}{\partial W^{(2)}}=\frac{\partial\mathcal{L}}{\partial\hat{y}}\frac{\partial\hat{y}}{\partial \boldsymbol{z}^{(2)}}\frac{\partial \boldsymbol{z}^{(2)}}{\partial W^{(2)}}$$

Now we can use the definitions of $\mathcal{L}$, $\hat{y}$, and $\boldsymbol{z}^{(2)}$ to calculate

$$\frac{\partial\mathcal{L}}{\partial\hat{y}}=-2(y-\hat{y})=2(\hat{y}-y),$$

and since $\hat{y}=\sigma\left(W^{(2)}\boldsymbol{a}^{(1)}\right)$, $\boldsymbol{z}^{(2)}=W^{(2)}\boldsymbol{a}^{(1)}$, we have

$$\frac{\partial\hat{y}}{\boldsymbol{z}^{(2)}}= \sigma'\left(\boldsymbol{z}^{(2)}\right), \quad \frac{\partial z^{(2)}}{\partial W^{(2)}}={\boldsymbol{a}^{(1)}}^T,$$

$$\frac{\partial\mathcal{L}}{\partial W^{(2)}} = 2(\hat{y}-y)\sigma'\left(z^{(2)}\right)\boldsymbol{a}^{(1)}}^T.$$

Note the dimensions of these quantities ($1\times 1$ and $1\times 2$ respectively).

## Calculating the Backward Pass

We have enough information to calculate numerical values now:

$$\frac{\partial\mathcal{L}}{\partial W^{(2)}}
= 2(0.554-1)\times\sigma'\left(0.217\right)\times\begin{pmatrix}0.75&0.71\end{pmatrix} = -0.22\begin{pmatrix}0.75&0.71\end{pmatrix} =\begin{pmatrix}-0.165&-0.156\end{pmatrix}$$

Then taking a step-size of $\eta=0.1$,

$$
W^{(2)} \to W^{(2)}-\eta \frac{\partial\mathcal{L}}{\partial W^{(2)}} = \begin{pmatrix}0.1& 0.2\end{pmatrix}-0.1\begin{pmatrix}-0.165& -0.156\end{pmatrix}
 = \begin{pmatrix}0.1165& 0.2156\end{pmatrix}.
 $$

These are the new weights applied to the outputs of the hidden layer to calculate the network output $\hat{y}$.
We can similarly compute

$$
\frac{\partial\mathcal{L}}{\partial W^{(1)}}= 
\frac{\partial\mathcal{L}}{\partial\hat{y}}
\frac{\partial\hat{y}}{\partial \boldsymbol{z}^{(2)}}
\frac{\partial{\boldsymbol{z}^{(2)}}}{\partial \boldsymbol{a}^{(1)}}
\frac{\partial{\boldsymbol{a}^{(1)}}}{\partial \boldsymbol{z}^{(1)}}
\frac{\partial \boldsymbol{z}^{(1)}}{\partial W^{(1)}}.
$$

Again note the quantities involved here: the first two terms on the right-hand side are $1\times 1$ (scalars), the third is a $1\times 2$ vector, the fourth a $2\times 2$ matrix, and the last term is the gradient of a $2\times 1$ vector with respect to a $2\times 3$ matrix, which gives a $2\times (2\times 3)$ tensor.

This will become rather unwieldy if we use the $2\times(2\times 3)$ representation of the last term, so let's compute it in component form instead. The tensor computation is given at the end for completeness.

Now, for the component form we have

$$
\frac{\partial \mathcal{L}}{\partial W^{(1)}}=
\begin{pmatrix}
\dfrac{\partial \mathcal{L}}{\partial w_{1,1}}
&\dfrac{\partial \mathcal{L}}{\partial w_{2,1}}
&\dfrac{\partial \mathcal{L}}{\partial w_{3,1}}\\
\dfrac{\partial \mathcal{L}}{\partial w_{2,1}}
&\dfrac{\partial \mathcal{L}}{\partial w_{2,2}}
&\dfrac{\partial \mathcal{L}}{\partial w_{2,3}}
\end{pmatrix},
$$

so let's calculate each of these six components.
We have 

$$
\frac{\partial \mathcal{L}}{\partial w_{1,1}} = \frac{\partial\mathcal{L}}{\partial\hat{y}}\frac{\partial\hat{y}}{\partial z^{(2)}} \left(\frac{\partial{z^{(2)}}}{\partial a_1^{(1)}}\frac{\partial{a_1^{(1)}}}{\partial z_1^{(1)}}\frac{\partial z_1^{(1)}}{\partial w_{1,1}^{(1)}} + \frac{\partial{z^{(2)}}}{\partial a_2^{(1)}}\frac{\partial{a_2^{(1)}}}{\partial z_2^{(1)}}\frac{\partial z_2^{(1)}}{\partial w_{1,1}^{(1)}}
\right) =2(\hat{y}-y)\sigma'\left(z^{(2)}\right)w_{1,1}^{(2)}\sigma'\left(z_1^{(1)}\right)x_1.$$

The full matrix is then

$$
\frac{\partial \mathcal{L}}{\partial W^{(1)}}=
2(\hat{y}-y)\sigma'\left(z^{(2)}\right)
\begin{pmatrix}
w_{1,1}^{(2)}\sigma'(z_1^{(1)})x_1
&w_{1,1}^{(2)}\sigma'(z_1^{(1)})x_2
&w_{1,1}^{(2)}\sigma'(z_1^{(1)})x_1\\
w_{2,1}^{(2)}\sigma'(z_2^{(1)})x_1
&w_{2,1}^{(2)}\sigma'(z_2^{(1)})x_2
&w_{2,1}^{(2)}\sigma'(z_2^{(1)})x_3
\end{pmatrix}$$

We found earlier that 

$$\sigma'(\boldsymbol{z}^{(2)})=\sigma'(2.17)=0.092,$$ 

and

$$\boldsymbol{z}^{(1)}=\begin{pmatrix}1.1\\ 
0.9\end{pmatrix}$$

so we calculate

$$
\sigma'(\boldsymbol{z}^{(1)})=\sigma\begin{pmatrix}1.1\\ 
0.9\end{pmatrix}\left(1-\sigma\begin{pmatrix}1.1\\
0.9\end{pmatrix}\right)=\begin{pmatrix}0.187\\ 
0.206\end{pmatrix}.
$$

Substituting gives

$$
\frac{\partial\mathcal{L}}{\partial W^{(1)}}
=-2(0.554-1)(0.092)
\begin{pmatrix}
0.187(0.1)(0)
&0.187(0.1)(1)
&0.187(0.1)(2)\\
0.206(0.2)(0)
&0.206(0.2)(1)
&0.206(0.2)(2)
\end{pmatrix}$$

$$ = 0.082
\begin{pmatrix}
0
&0.0187
&0.0374\\
0
&0.0412
&0.0824
\end{pmatrix}=\begin{pmatrix}
0&0.0015&0.0031\\
0&0.0034&0.0068
\end{pmatrix}.$$

Our updated weights after one pass are then

$$
W^{(1)} \to W^{(1)}-\eta\frac{\partial\mathcal{L}}{\partial W^{(1)}}\\
=\begin{pmatrix}0.4&0.5&0.3\\
0.2&0.7&0.1\end{pmatrix}-0.1\begin{pmatrix}
0&0.0015&0.0031\\
0&0.0034&0.0068
\end{pmatrix} =\begin{pmatrix}
0.4&0.4999&0.2997\\
0.2&0.6997&0.0993
\end{pmatrix}.
$$

## Updated Prediction - Another Forward Pass

Remember that we're trying to train our network to predict $\hat{y}=1$ when $\boldsymbol{x}=(0,1,2)$.
If we're making progress, our new weights should give a better prediction than $\hat{y}=0.5541$ found from the initial set of weights.
We have

$$
\boldsymbol{z}^{(1)}=W^{(1)}\boldsymbol{x}=\begin{pmatrix}
0.4& 0.4998& 0.2996\\
0.2& 0.6997& 0.0993
\end{pmatrix}\begin{pmatrix}
0\\
1\\
2
\end{pmatrix}=\begin{pmatrix}
1.099\\
0.8983
\end{pmatrix},
$$

$$
\boldsymbol{a}^{(1)}=\sigma(\boldsymbol{z}^{(1)})=\begin{pmatrix}
0.7501\\
0.7106
\end{pmatrix},
$$

and

$$
\boldsymbol{z}^{(2)}=W^{(2)}\boldsymbol{a}^{(1)}=\begin{pmatrix}
0.1165 &0.2156
\end{pmatrix}
\begin{pmatrix}
0.7501\\
0.7106
\end{pmatrix}=0.2406.
$$

Finally, our new prediction is

$$
\hat{y}=\sigma\left(\boldsymbol{z}^{(2)}\right)=0.5599,
$$

which means the error is indeed reduced after one step of gradient descent.

## Tensor Computation of Weight Updates

To avoid working with the $2\times (2\times 3)$ tensor directly, we computed the gradient of the loss with respect to $W^{(1)}$ in component form. We can do this directly by writing

$$\frac{\partial \boldsymbol{z}^{(1)}}{\partial W^{(1)}} = \begin{pmatrix}
\dfrac{\partial z_1^{(1)}}{\partial w_{1,1}^{(1)}}
& \dfrac{\partial z_1^{(1)}}{\partial w_{2,1}^{(1)}}
& \dfrac{\partial z_1^{(1)}}{\partial w_{3,1}^{(1)}}\\
0& 0& 0\\
0& 0& 0\\
\dfrac{\partial z_2^{(1)}}{\partial w_{1,2}^{(1)}}
& \dfrac{\partial z_2^{(1)}}{\partial w_{2,2}^{(1)}}
& \dfrac{\partial z_2^{(1)}}{\partial w_{3,2}^{(1)}}
\end{pmatrix} = \begin{pmatrix}
x_1& x_2& x_3\\
0& 0& 0\\
0& 0& 0\\
x_1& x_2& x_3
\end{pmatrix},
$$
where we have written the tensor as two $2\times 3$ matrices stacked on top of each other. 
To do the full calculation, think of this as two rows, which we'll multiply with the two columns of the vector preceding it.
Then with
$$
\frac{\partial{z^{(2)}}}{\partial \boldsymbol{a}^{(1)}}=W^{(2)},
$$

$$
\frac{\partial{\boldsymbol{a}^{(1)}}}{\partial \boldsymbol{z}^{(1)}} = \begin{pmatrix}
\dfrac{\partial a_1^{(1)}}{\partial z_1^{(1)}}
&\dfrac{\partial a_1^{(1)}}{\partial z_2^{(1)}}\\
\dfrac{\partial a_2^{(1)}}{\partial z_1^{(1)}}
&\dfrac{\partial a_2^{(1)}}{\partial z_2^{(1)}}
\end{pmatrix} = \begin{pmatrix}
\sigma'\left(z_1^{(1)}\right)
&0\\
0
&\sigma'\left(z_2^{(1)}\right)
\end{pmatrix}.
$$

we have

$$\frac{\partial \mathcal{L}}{\partial W^{(1)}} = 2(\hat{y}-y)\sigma'(z^{(2)})
\begin{pmatrix}
w_{1,1}^{(2)}\sigma'\left(z_1^{(1)}\right)
& w_{2,1}^{(2)}\sigma'\left(z_2^{(1)}\right)
\end{pmatrix}
\begin{pmatrix}
x_1& x_2& x_3\\
0& 0& 0\\
0& 0& 0\\
x_1&x_2&x_3
\end{pmatrix}$$

$$ = 2(\hat{y}-y)\sigma'(z^{(2)})\left[\sigma'(z_1^{(1)})\begin{pmatrix}
w_{1,1}^{(2)}x_1
& w_{1,1}^{(2)}x_2
& w_{1,1}^{(2)}x_3\\
0& 0& 0
\end{pmatrix} + \sigma'(z_2^{(1)})\begin{pmatrix}
0& 0& 0\\
w_{2,1}^{(2)}x_1
& w_{2,1}^{(2)}x_2
& w_{2,1}^{(2)}x_3
\end{pmatrix}
\right] $$

$$ = 2(\hat{y}-y)\sigma'(z^{(2)})
\begin{pmatrix}
\sigma'(z_1^{(1)})w_{1,1}^{(2)}x_1
& \sigma'(z_1^{(1)})w_{1,1}^{(2)}x_2
& \sigma'(z_1^{(1)})w_{1,1}^{(2)}x_3\\
\sigma'(z_2^{(1)})w_{2,1}^{(2)}x_1
& \sigma'(z_2^{(1)})w_{2,1}^{(2)}x_2
& \sigma'(z_2^{(1)})w_{2,1}^{(2)}x_3
\end{pmatrix}.
$$

This is the same expression as obtained using the component form of the derivative.
