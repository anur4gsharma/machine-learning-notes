# Logistic Regression

## Hypothesis Function

We want the model output to satisfy:

$$
h_\theta(x) \in [0, 1]
$$

This allows the output of the model to be interpreted as a probability.

The hypothesis function is:

$$
h_\theta(x) = g(\theta^\top x) = \frac{1}{1 + e^{-\theta^\top x}}
$$

---

## Sigmoid Function

The sigmoid function is:

$$
g(z) = \frac{1}{1 + e^{-z}}
$$

It maps any real-valued input to the interval $(0, 1)$.

---

## Probability of Classification

For binary classification:

$$
y \in \{0, 1\}
$$

The probability that $y = 1$ is:

$$
P(y = 1 \mid x;\, \theta) = h_\theta(x)
$$

The probability that $y = 0$ is:

$$
P(y = 0 \mid x;\, \theta) = 1 - h_\theta(x)
$$

These can be combined into one expression:

$$
P(y \mid x;\, \theta) = h_\theta(x)^{y} \left(1 - h_\theta(x)\right)^{1-y}
$$

---

# Maximum Likelihood Estimation

Given $m$ training examples, the likelihood function is:

$$
L(\theta) = \prod_{i=1}^{m} P\!\left(y^{(i)} \mid x^{(i)};\, \theta\right)
$$

Using the Bernoulli model:

$$
L(\theta) = \prod_{i=1}^{m} h_\theta\!\left(x^{(i)}\right)^{y^{(i)}} \left(1 - h_\theta\!\left(x^{(i)}\right)\right)^{1 - y^{(i)}}
$$

We choose the parameters $\theta$ that maximize the likelihood.

---

# Log-Likelihood

Taking the logarithm:

$$
\ell(\theta) = \log L(\theta)
$$

Therefore:

$$
\ell(\theta) = \sum_{i=1}^{m} \left[ y^{(i)} \log h_\theta\!\left(x^{(i)}\right) + \left(1 - y^{(i)}\right) \log\!\left(1 - h_\theta\!\left(x^{(i)}\right)\right) \right]
$$

We choose $\theta$ to maximize $\ell(\theta)$.

---

# Algorithm Used to Maximize the Log-Likelihood

## Batch Gradient Ascent

The general gradient-ascent update is:

$$
\theta_j \leftarrow \theta_j + \alpha \frac{\partial \ell(\theta)}{\partial \theta_j}
$$

For logistic regression:

$$
\theta_j \leftarrow \theta_j + \alpha \sum_{i=1}^{m} \left(y^{(i)} - h_\theta\!\left(x^{(i)}\right)\right) x_j^{(i)}
$$

In vector form:

$$
\theta \leftarrow \theta + \alpha \sum_{i=1}^{m} \left(y^{(i)} - h_\theta\!\left(x^{(i)}\right)\right) x^{(i)}
$$

---

# Newton's Method

Newton's method can be used to find the optimum more quickly than gradient ascent.

For a one-dimensional function:

$$
\theta^{(t+1)} = \theta^{(t)} - \frac{f'(\theta^{(t)})}{f''(\theta^{(t)})}
$$

For multiple parameters, the Hessian matrix is used:

$$
\theta^{(t+1)} = \theta^{(t)} - H^{-1} \nabla J(\theta^{(t)})
$$

A common compact form is:

$$
\theta \leftarrow \theta - H^{-1} \nabla J(\theta)
$$

The Hessian matrix contains second-order partial derivatives:

$$
H_{ij} = \frac{\partial^2 J(\theta)}{\partial \theta_i \, \partial \theta_j}
$$

---

# Summary

| Topic                   | Key Idea                                                           |
| ----------------------- | ------------------------------------------------------------------ |
| **Sigmoid**             | Converts a real number into a value between 0 and 1                |
| **Logistic Regression** | Uses the sigmoid output as a probability                           |
| **Maximum Likelihood**  | Chooses $\theta$ to maximize the probability of the training data  |
| **Log-Likelihood**      | Logarithm of the likelihood, making products easier to work with   |
| **Gradient Ascent**     | Iteratively updates $\theta$ in the direction that increases likelihood |
| **Newton's Method**     | Uses first- and second-order derivatives for faster optimization   |
