# Perceptron and Generalized Linear Models

## Perceptron

### Logistic Regression vs. Perceptron

| Logistic / Sigmoid Function | Perceptron / Step Function |
| --- | --- |
| **Sigmoid:** $g(z) = \frac{1}{1 + e^{-z}}$ | **Step:** $g(z) = 1$ if $z \ge 0$, $g(z) = 0$ if $z < 0$ |
| **Hypothesis:** $h_\theta(x) = \frac{1}{1 + e^{-\theta^\top x}}$ | **Hypothesis:** $h_\theta(x) = g(\theta^\top x)$ |
| Output is a probability between 0 and 1. | Output is either 0 or 1. |

---

### Batch Gradient Ascent for the Perceptron

The parameter update is:

$$
\theta_j \leftarrow \theta_j + \alpha \sum_{i=0}^{m} \left(y^{(i)} - h_\theta\!\left(x^{(i)}\right)\right) x_j^{(i)}
$$

Where:

* $\alpha$ = learning rate
* $y^{(i)}$ = true label
* $h_\theta(x^{(i)})$ = predicted label
* $x_j^{(i)}$ = feature $j$ of training example $i$

For the perceptron:

* If the prediction is **correct**, then the error is **0**, so there is no update.
* If the prediction is **incorrect**, then the error is either **+1** or **−1**.

---

# Exponential Families

A probability distribution belongs to the **exponential family** if it can be written in the form:

$$
P(y;\, \eta) = b(y) \exp\!\left(\eta^\top T(y) - a(\eta)\right)
$$

### Components

| Symbol   | Meaning                |
| -------- | ---------------------- |
| $y$      | Observed data          |
| $\eta$   | Natural parameter      |
| $T(y)$   | Sufficient statistic   |
| $b(y)$   | Base measure           |
| $a(\eta)$ | Log-partition function |

---

# Examples

## Bernoulli Distribution

For binary data:

$$
y \in \{0, 1\}
$$

Let:

$$
\phi = P(y = 1)
$$

The Bernoulli probability mass function is:

$$
P(y;\, \phi) = \phi^{y}(1 - \phi)^{1-y}
$$

Rewrite it using an exponential:

$$
P(y;\, \phi) = \exp\!\left(\log\!\left(\phi^{y}(1 - \phi)^{1-y}\right)\right)
$$

Using logarithm rules:

$$
P(y;\, \phi) = \exp\!\left(y \log \phi + (1 - y) \log(1 - \phi)\right)
$$

Expand the expression:

$$
P(y;\, \phi) = \exp\!\left(y \log \phi - y \log(1 - \phi) + \log(1 - \phi)\right)
$$

Therefore:

$$
P(y;\, \phi) = \exp\!\left(y \log\!\frac{\phi}{1 - \phi} + \log(1 - \phi)\right)
$$

Compare this with the exponential-family form:

$$
P(y;\, \eta) = b(y) \exp\!\left(\eta\, T(y) - a(\eta)\right)
$$

We can identify:

$$
b(y) = 1
$$

$$
T(y) = y
$$

$$
\eta = \log\!\frac{\phi}{1 - \phi}
$$

### Solving for φ

Starting from:

$$
\eta = \log\!\frac{\phi}{1 - \phi}
$$

Exponentiating both sides:

$$
e^{\eta} = \frac{\phi}{1 - \phi}
$$

Rearranging:

$$
\phi = \frac{e^{\eta}}{1 + e^{\eta}}
$$

Therefore:

$$
\phi = \frac{1}{1 + e^{-\eta}}
$$

### Finding a(η)

From the exponential-family form:

$$
-a(\eta) = \log(1 - \phi)
$$

Therefore:

$$
a(\eta) = -\log(1 - \phi)
$$

Substituting:

$$
\phi = \frac{1}{1 + e^{-\eta}}
$$

gives:

$$
a(\eta) = -\log\!\left(1 - \frac{1}{1 + e^{-\eta}}\right)
$$

which simplifies to:

$$
a(\eta) = \log(1 + e^{\eta})
$$

### Final Bernoulli Exponential-Family Form

| Component  | Value |
| ---------- | ----- |
| $b(y)$     | $1$ |
| $T(y)$     | $y$ |
| $\eta$     | $\log\frac{\phi}{1 - \phi}$ |
| $\phi$     | $\frac{1}{1 + e^{-\eta}}$ |
| $a(\eta)$  | $\log(1 + e^{\eta})$ |

---

## Gaussian Distribution (Fixed Variance)

Consider:

$$
y \sim \mathcal{N}(\mu, \sigma^2)
$$

The Gaussian probability density function is:

$$
P(y;\, \mu) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\!\left(-\frac{(y - \mu)^2}{2\sigma^2}\right)
$$

Expand the squared term:

$$
(y - \mu)^2 = y^2 - 2\mu y + \mu^2
$$

Substituting:

$$
P(y;\, \mu) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\!\left(\frac{\mu}{\sigma^2} y - \frac{\mu^2}{2\sigma^2} - \frac{y^2}{2\sigma^2}\right)
$$

Compare this with:

$$
P(y;\, \eta) = b(y) \exp\!\left(\eta\, T(y) - a(\eta)\right)
$$

We can identify:

$$
T(y) = y
$$

$$
\eta = \frac{\mu}{\sigma^2}
$$

Since:

$$
\mu = \eta \sigma^2
$$

we get:

$$
a(\eta) = \frac{\sigma^2 \eta^2}{2}
$$

The terms that depend only on $y$ are included in the base measure:

$$
b(y) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\!\left(-\frac{y^2}{2\sigma^2}\right)
$$

### Final Gaussian Exponential-Family Form

| Component  | Value |
| ---------- | ----- |
| $b(y)$     | $\frac{1}{\sqrt{2\pi\sigma^2}} \exp\!\left(-\frac{y^2}{2\sigma^2}\right)$ |
| $T(y)$     | $y$ |
| $\eta$     | $\frac{\mu}{\sigma^2}$ |
| $a(\eta)$  | $\frac{\sigma^2 \eta^2}{2}$ |
