---
title: 'Computing Gradient Descent in Neural Networks By Hand'
date: 2023-09-29
permalink: /posts/2023/09/computing-gradient-descent-nn/
tags:
  - gradient descent
  - neural networks
  - computational mathematics
---

**Gradient Descent in Neural Networks**

I haven't been able to find an explanation of the computation of updated network weights in a neural network with more than one node in more than one layer, which doesn't involve computing the error and jumping to define a formula using the element-wise product of vectors or matrices. 
Here we do this computation by computing the partial derivatives directly, either in vector or component form, with activation function $$\sigma(z)=\frac{1}{1+e^{-z}}.$$
