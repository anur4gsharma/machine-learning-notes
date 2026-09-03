# 01-introduction_neural_networks

# Deep Learning Notes

> Transcribed and cleaned up from handwritten notes. Equations are written in LaTeX and notation has been standardized where possible.
> 

## 1. Linear and Nonlinear Models

### Linear model

A linear model can be written as

$$
h_\theta(x)
=
\theta_1 x_1 + \theta_2 x_2 + \theta_3 x_3 + \cdots + \theta_d x_d,
\qquad x \in \mathbb{R}^d
$$

where $x$ and $\theta$ are $d$-dimensional vectors.

A linear model is **linear in its parameters**: the parameters appear only to the first power and are not passed through nonlinear functions.

### Nonlinear model

A nonlinear model is generally a model that is nonlinear in its parameters (as opposed to merely being nonlinear in the input data).

For example,

$$
h_\theta(x)
=
\theta_1^2 x_1 + \theta_2 x_2 + \theta_3^3 x_3 + \cdots
$$

is nonlinear in the parameters because parameters appear with powers such as $\theta_1^2$ and $\theta_3^3$.

---

## 2. Multiclass Classification

Suppose we have $K$ classes:

$$
y \in \{1,2,\ldots,K\},
\qquad
x \in \mathbb{R}^d.
$$

A model maps the input to $K$ scores:

$$
h_\theta : \mathbb{R}^d \rightarrow \mathbb{R}^K.
$$

Let

$$
h_\theta(x)
=
\begin{bmatrix}
h_1(x)\\
h_2(x)\\
\vdots\\
h_K(x)
\end{bmatrix}.
$$

### Predicted probability

The probability assigned to class $j$ is given by the softmax function:

$$
P(y=j\mid x)
=
\frac{\exp(h_j(x))}
{\displaystyle\sum_{s=1}^{K}\exp(h_s(x))}.
$$

The softmax converts the model’s $K$ scores into probabilities that sum to $1$.

### Negative log-likelihood / cross-entropy

For an example whose true class is $y$, the negative log-likelihood is

$$
-\log P(y\mid x).
$$

Therefore, the cross-entropy loss for one example is

$$
\ell_{\mathrm{CE}}(h_\theta(x),y)
=
-\log
\left(
\frac{\exp(h_y(x))}
{\displaystyle\sum_{s=1}^{K}\exp(h_s(x))}
\right).
$$

For a dataset of $n$ examples,

$$
\mathcal{L}
=
\frac{1}{n}
\sum_{i=1}^{n}
\ell_{\mathrm{CE}}
\left(h_\theta(x^{(i)}),y^{(i)}\right).
$$

---

# 3. Neural Networks

## 3.1 Why introduce nonlinearities?

Consider the ReLU activation function:

$$
\operatorname{ReLU}(z)
=
\max\{z,0\}.
$$

A neuron computes

$$
h_0(x)
=
\operatorname{ReLU}(w^\top x+b),
$$

where

$$
x\in\mathbb{R}^d,
\qquad
w\in\mathbb{R}^d,
\qquad
b\in\mathbb{R}.
$$

Here:

- $w$ contains the **weights**.
- $b$ is the **bias**.
- ReLU is the **activation function**.

The resulting neuron produces a nonlinear transformation of the input.

The ReLU function can be visualized as a function that is zero for negative inputs and grows linearly for positive inputs.

Other nonlinear activation functions include:

- Sigmoid
- Tanh
- ReLU

---

## 3.2 A layer of neurons

Suppose a layer contains $m$ neurons.

For each neuron,

$$
a_i
=
\operatorname{ReLU}(w_i^\top x+b_i),
\qquad i=1,\ldots,m.
$$

Stacking all neuron outputs into one vector gives

$$
a
=
\operatorname{ReLU}(Wx+b),
$$

where

$$
W
=
\begin{bmatrix}
w_1^\top\\
w_2^\top\\
\vdots\\
w_m^\top
\end{bmatrix}
\in\mathbb{R}^{m\times d},
$$

and

$$
b
=
\begin{bmatrix}
b_1\\
b_2\\
\vdots\\
b_m
\end{bmatrix}
\in\mathbb{R}^{m}.
$$

Therefore,

$$
Wx+b\in\mathbb{R}^{m},
$$

and the layer maps

$$
\mathbb{R}^{d}\rightarrow\mathbb{R}^{m}.
$$

---

## 3.3 Multiple layers

A neural network is formed by composing multiple layers.

For example, a two-layer network can be written as

$$
a
=
\operatorname{ReLU}(W^{[1]}x+b^{[1]}),
$$

followed by

$$
h_\theta(x)
=
W^{[2]}a+b^{[2]}.
$$

Combining them,

$$
h_\theta(x)
=
W^{[2]}
\operatorname{ReLU}
\left(
W^{[1]}x+b^{[1]}
\right)
+b^{[2]}.
$$

The superscript $[l]$ denotes the layer number.

More generally, a network can be represented as a composition such as

$$
f(x)
=
f_L\circ f_{L-1}\circ\cdots\circ f_1(x).
$$

The nonlinear activation between linear transformations is what allows a deep network to represent nonlinear functions.

### Important point

Without nonlinear activation functions, composing linear/affine layers would still result in a single affine transformation. For example,

$$
W_2(W_1x+b_1)+b_2
=
(W_2W_1)x+(W_2b_1+b_2).
$$

So depth by itself does not create the desired nonlinear expressive power; the nonlinear activations are crucial.

---

# 4. Residual Connections / Residual Networks

A residual connection (or **skip connection**) adds the input of a block directly to its output.

Suppose a normal network block computes some transformation $F(z)$:

$$
F(z)
=
\operatorname{ReLU}
\left(
W_2\operatorname{ReLU}(W_1z+b_1)+b_2
\right).
$$

A residual block instead computes something of the form

$$
\operatorname{ReLU}(F(z)+z).
$$

The direct $z$ term is the **skip connection**.

---

## 4.1 Residual learning

Suppose the desired mapping is

$$
y\approx H(z).
$$

Instead of learning $H$ directly, a residual network learns the residual

$$
F(z)=H(z)-z.
$$

Then

$$
H(z)=F(z)+z.
$$

Thus the network can produce

$$
y
=
F(z)+z
$$

or, with an activation after the addition,

$$
y
=
\operatorname{ReLU}(F(z)+z).
$$

The residual branch learns the difference between the desired output and the input.

---

## 4.2 Why use residual connections?

A residual connection gives the network a direct path from the input to later layers.

Conceptually:

```
          ┌──────────────────────────────┐
          │                              │
          │          skip connection     │
          │                              ▼
input ────┼──► [ Layer → ReLU → Layer ] ──► (+) ──► ReLU ──► output
          │                              ▲
          └──────────────────────────────┘
```

The idea is that the block can learn a correction to the input rather than having to learn the entire mapping from scratch.

Residual connections became a key component of **ResNets (Residual Neural Networks)** and are especially useful when building very deep neural networks.

---