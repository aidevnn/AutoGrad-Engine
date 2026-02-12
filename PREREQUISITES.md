# Prerequisites — What You Need to Know

Everything you need to understand MicroGPT, starting from scratch. No prior machine learning experience required. We assume basic secondary school math and build up from there.

---

## Table of Contents

1. [Functions](#1-functions)
2. [Exponents](#2-exponents)
3. [Logarithms](#3-logarithms)
4. [Vectors](#4-vectors)
5. [Matrices and Matrix Multiplication](#5-matrices-and-matrix-multiplication)
6. [Probability Basics](#6-probability-basics)
7. [Derivatives — The Concept of "How Fast Is It Changing?"](#7-derivatives--the-concept-of-how-fast-is-it-changing)
8. [The Chain Rule](#8-the-chain-rule)
9. [Computation Graphs and Automatic Differentiation](#9-computation-graphs-and-automatic-differentiation)
10. [What Is a Neural Network?](#10-what-is-a-neural-network)
11. [Parameters and Weights](#11-parameters-and-weights)
12. [The Forward Pass — Making a Prediction](#12-the-forward-pass--making-a-prediction)
13. [Loss Functions — "How Wrong Are We?"](#13-loss-functions--how-wrong-are-we)
14. [Gradient Descent — Learning by Nudging](#14-gradient-descent--learning-by-nudging)
15. [Backpropagation — Computing All Gradients at Once](#15-backpropagation--computing-all-gradients-at-once)
16. [Learning Rate](#16-learning-rate)
17. [The Adam Optimizer](#17-the-adam-optimizer)
18. [Activation Functions](#18-activation-functions)
19. [Softmax — Turning Numbers into Probabilities](#19-softmax--turning-numbers-into-probabilities)
20. [Tokenization](#20-tokenization)
21. [Embeddings — Numbers That Represent Meaning](#21-embeddings--numbers-that-represent-meaning)
22. [Sequence Models and Language Modeling](#22-sequence-models-and-language-modeling)
23. [Attention — "What Should I Focus On?"](#23-attention--what-should-i-focus-on)
24. [Multi-Head Attention](#24-multi-head-attention)
25. [Residual Connections](#25-residual-connections)
26. [Normalization](#26-normalization)
27. [The Transformer Architecture](#27-the-transformer-architecture)
28. [Autoregressive Generation](#28-autoregressive-generation)
29. [Putting It All Together](#29-putting-it-all-together)
30. [Further Reading](#30-further-reading)

---

## 1. Functions

A **function** takes an input and produces an output. You've seen these since school:

$$f(x) = 2x + 3$$

Feed in $x = 4$, get out $f(4) = 11$.

In this project, every operation the model does is a function. The entire neural network is one big function: it takes in a letter, and outputs a prediction for the next letter.

**Why it matters:** The whole model is a chain of simple functions (add, multiply, exponentiate) applied one after another. Understanding that each step is "input → do something → output" is the foundation for everything that follows.

---

## 2. Exponents

An **exponent** means repeated multiplication:

$$2^3 = 2 \times 2 \times 2 = 8$$

The special number $e \approx 2.718$ shows up everywhere in math. The function $e^x$ (often written `exp(x)`) has a magical property: **it's always positive**, and it grows very fast for large $x$.

```
e^0   = 1
e^1   ≈ 2.72
e^2   ≈ 7.39
e^10  ≈ 22026
e^(-2) ≈ 0.14
```

**Why it matters:** The model uses $e^x$ in **softmax** to convert raw scores into probabilities. Because $e^x$ is always positive, it guarantees we never get negative probabilities. And because it amplifies differences — $e^{10}$ is vastly larger than $e^2$ — it makes the model "decisive." If one option scores much higher, it dominates.

---

## 3. Logarithms

The **logarithm** is the reverse of an exponent. If $e^x = y$, then $\ln(y) = x$.

```
ln(1)    = 0       because e^0 = 1
ln(2.72) ≈ 1       because e^1 ≈ 2.72
ln(0.5)  ≈ -0.69   (negative! log of numbers < 1 is negative)
ln(0.01) ≈ -4.6    (very negative for numbers close to 0)
```

Key properties:
- $\ln(a \times b) = \ln(a) + \ln(b)$ — turns multiplication into addition
- $\ln(1) = 0$
- $\ln$ of numbers between 0 and 1 is **negative**
- As the input gets closer to 0, $\ln$ goes toward $-\infty$

**Why it matters:** The **loss function** uses $-\ln(\text{probability})$. If the model assigns 90% probability to the correct answer, the loss is $-\ln(0.9) \approx 0.1$ (small, good). If it assigns 1%, the loss is $-\ln(0.01) \approx 4.6$ (big, bad). The log harshly punishes confident wrong answers — exactly what we want.

---

## 4. Vectors

A **vector** is just a list of numbers:

$$\mathbf{v} = [0.5, -1.2, 3.0, 0.8]$$

That's it. Think of it as a row in a spreadsheet. In this project, every token (letter) is represented by a vector of 16 numbers.

**Common operations:**

**Addition** — add corresponding elements:
$$[1, 2, 3] + [4, 5, 6] = [5, 7, 9]$$

**Scalar multiplication** — multiply every element by a number:
$$3 \times [1, 2, 3] = [3, 6, 9]$$

**Dot product** — multiply corresponding elements, then sum:
$$[1, 2, 3] \cdot [4, 5, 6] = 1{\times}4 + 2{\times}5 + 3{\times}6 = 32$$

The dot product measures **similarity**. If two vectors point in the same direction, their dot product is large and positive. If they're unrelated, it's near zero. If they point in opposite directions, it's negative.

**Why it matters:** The model represents every letter as a vector. Attention uses the dot product to measure "how relevant is this past letter to predicting the next one?" High dot product = very relevant.

---

## 5. Matrices and Matrix Multiplication

A **matrix** is a grid (table) of numbers. Think of it as a stack of vectors:

$$W = \begin{bmatrix} 2 & 1 \\ 0 & 3 \end{bmatrix}$$

**Matrix-vector multiplication** takes a vector and produces a new vector. Each element of the output is a dot product of one row of the matrix with the input vector:

$$\begin{bmatrix} 2 & 1 \\ 0 & 3 \end{bmatrix} \times \begin{bmatrix} 0.5 \\ 0.3 \end{bmatrix} = \begin{bmatrix} 2 \times 0.5 + 1 \times 0.3 \\ 0 \times 0.5 + 3 \times 0.3 \end{bmatrix} = \begin{bmatrix} 1.3 \\ 0.9 \end{bmatrix}$$

Each row of the matrix decides "how much of each input do I want?" The numbers in the matrix are the **weights** — they control which combinations of inputs are useful.

**Why it matters:** This is the `Linear` operation in the code, and it's the single most common operation in neural networks. Every layer of the model does matrix-vector multiplication. The entire GPT model is mostly a series of these operations. When we say the model "learns," we mean it adjusts the numbers inside these matrices.

---

## 6. Probability Basics

A **probability** is a number between 0 and 1 that represents how likely something is. Probabilities of all possible outcomes must sum to 1.

Example — predicting the next letter after "Em":
```
P(m) = 0.45    ← most likely (Emma, Emily, ...)
P(i) = 0.20    ← possible (Emil, Emilio, ...)
P(a) = 0.10
...all others... small
Total = 1.0
```

**Probability distribution** — the full list of probabilities for all possible outcomes. The model outputs one of these at every step: a probability for each character in the vocabulary.

**Sampling** — picking a random outcome according to the probabilities. If $P(m) = 0.45$, then about 45% of the time we'd pick "m." This is how the model generates diverse outputs — it doesn't always pick the most likely letter.

**Why it matters:** The model's entire job is to produce a probability distribution over the next character. Training adjusts the model so that the correct next character gets a higher probability.

---

## 7. Derivatives — The Concept of "How Fast Is It Changing?"

The **derivative** answers: "If I nudge the input a tiny bit, how much does the output change?"

For $f(x) = x^2$:
- At $x = 3$: $f(3) = 9$. If we nudge to $x = 3.001$: $f(3.001) = 9.006001$. The output changed by about $0.006$ for a $0.001$ nudge — that's a rate of $6$. The derivative is $f'(3) = 6$.
- Formula: $f'(x) = 2x$. At $x = 3$: $f'(3) = 2 \times 3 = 6$. ✓

**Common derivative rules you'll see in the code:**

| Function | Derivative | In plain English |
|---|---|---|
| $x^n$ | $n \cdot x^{n-1}$ | Power rule |
| $e^x$ | $e^x$ | Exponential is its own derivative! |
| $\ln(x)$ | $1/x$ | Log's derivative is the reciprocal |
| $a + b$ | $1$ for both $a$ and $b$ | Adding doesn't change the rate |
| $a \times b$ | $b$ for $a$, $a$ for $b$ | Each input's rate is scaled by the other |

You don't need to memorize these. The point is: **every math operation has a derivative rule**, and the code implements each one.

**Why it matters:** Derivatives are how the model learns. The derivative of the loss with respect to a weight tells us: "Should I increase or decrease this weight to reduce the error, and by how much?" This is called the **gradient**.

---

## 8. The Chain Rule

What if functions are **chained** together? If $y = f(g(x))$ — that is, first apply $g$, then apply $f$ to the result — the chain rule says:

$$\frac{dy}{dx} = \frac{dy}{dg} \times \frac{dg}{dx}$$

**Concrete example:**

Let $g(x) = 3x$ and $f(g) = g^2$. So $y = (3x)^2$.

- $\frac{dg}{dx} = 3$ (nudging $x$ by 1 changes $g$ by 3)
- $\frac{dy}{dg} = 2g$ (power rule on $g^2$)
- $\frac{dy}{dx} = 2g \times 3 = 6g = 6(3x) = 18x$

At $x = 2$: the derivative is $18 \times 2 = 36$. Check: $(3 \times 2)^2 = 36$, $(3 \times 2.001)^2 = 36.036...$, change ≈ $0.036$ for a $0.001$ nudge → rate = $36$. ✓

**The key insight:** No matter how many functions you chain, you just **multiply the derivatives** of each step. A neural network is hundreds of operations chained together. The chain rule lets us compute the derivative of the final output with respect to any input — even one buried deep in the chain.

**Why it matters:** This is the mathematical foundation of **backpropagation**. The model chains together hundreds of additions, multiplications, and exponentials. The chain rule lets us trace back through all of them to figure out how each weight affects the final error.

---

## 9. Computation Graphs and Automatic Differentiation

A **computation graph** is a diagram showing how values flow through operations. Every operation is a node, and the arrows show which values feed into which operations.

Example: $y = (a + b) \times c$

```
a ──┐
    ├── [+] ── d ──┐
b ──┘               ├── [×] ── y
               c ──┘
```

Each node knows:
1. What operation it performs (its forward rule)
2. How to pass gradients backward (its backward rule)

**Automatic differentiation (autograd)** means the computer builds this graph as you compute, then walks it backward to compute all gradients automatically. You never write derivative formulas by hand.

**In the code:** The `Value` class is exactly this. Every time you write `a + b` or `a * b`, it creates a new `Value` node that remembers its inputs and the operation. When you call `Backward()`, it walks the graph in reverse, applying the chain rule at each node.

**Why it matters:** This is the engine that makes learning possible. Without autograd, you'd have to manually derive gradient formulas for every possible network architecture — which would be impractical for anything complex.

---

## 10. What Is a Neural Network?

A **neural network** is a function that:
1. Takes an input (e.g., a letter)
2. Passes it through a series of **layers** (each layer is matrix multiplication + some nonlinear function)
3. Produces an output (e.g., probabilities for the next letter)

The key idea: the numbers inside the matrices (**weights**) are not designed by a human. They are **learned** from data. The network starts with random weights, makes terrible predictions, and gradually adjusts its weights to make better ones.

A "layer" is just: **multiply by a matrix** (mix the inputs together in learnable proportions), then **apply a simple nonlinear function** (like ReLU — see section 18). Stack several layers and you get a deep neural network.

**Why it matters:** MicroGPT is a neural network. Understanding that it's ultimately "multiply, add a nonlinearity, repeat" demystifies the whole thing.

---

## 11. Parameters and Weights

**Parameters** (also called **weights**) are all the numbers inside the model that get adjusted during training. They live in matrices.

In MicroGPT (with default settings), there are **3,648 parameters**. They include:
- **Token embeddings** — a vector for each character in the vocabulary
- **Position embeddings** — a vector for each position in the sequence
- **Attention weights** — matrices that compute Query, Key, and Value projections
- **MLP weights** — matrices in the feed-forward layers

At the start of training, all parameters are **random** (small random numbers from a bell curve). After training, they encode the statistical patterns of the training data — in our case, what English names look and sound like.

**Why it matters:** When someone says "GPT-4 has 1.8 trillion parameters," they mean it has 1.8 trillion learnable numbers in its matrices. MicroGPT has 3,648. Same concept, vastly different scale.

---

## 12. The Forward Pass — Making a Prediction

The **forward pass** is running an input through the model to get an output. Data flows forward through the layers:

```
Input letter → Embedding → Transformer layers → Output scores → Probabilities
```

Each step applies a mathematical operation. The forward pass is deterministic — same input and same weights always produce the same output.

In the code, this is the `GPT()` function. It takes a token ID and position, runs it through attention and MLP layers, and returns a score for every possible next token.

**Why it matters:** This is the prediction step. During both training and generation, the model does a forward pass to produce its guess for the next character.

---

## 13. Loss Functions — "How Wrong Are We?"

The **loss** is a single number measuring how bad the model's prediction was. Lower is better.

MicroGPT uses **cross-entropy loss**:

$$\text{loss} = -\ln(p_{\text{correct}})$$

Where $p_{\text{correct}}$ is the probability the model assigned to the **correct** next character.

| Model's confidence in the right answer | Loss |
|---|---|
| 90% (0.9) | $-\ln(0.9) \approx 0.11$ |
| 50% (0.5) | $-\ln(0.5) \approx 0.69$ |
| 10% (0.1) | $-\ln(0.1) \approx 2.30$ |
| 1% (0.01) | $-\ln(0.01) \approx 4.60$ |

The loss drops toward 0 as the model gets more confident in the right answer, and shoots up when the model is confidently wrong.

**Baseline:** Random guessing with 28 characters means each gets $\frac{1}{28} \approx 3.6\%$. The loss is $-\ln(1/28) = \ln(28) \approx 3.33$. So any loss below 3.33 means the model has learned something.

**Why it matters:** The loss is the model's "report card." The entire training process exists to make this number go down.

---

## 14. Gradient Descent — Learning by Nudging

**Gradient descent** is the algorithm that adjusts parameters to reduce the loss. The idea is beautifully simple:

1. Compute the gradient (derivative) of the loss with respect to each parameter
2. The gradient tells you the direction that **increases** the loss
3. Move each parameter a tiny step in the **opposite** direction (to decrease the loss)
4. Repeat

$$w_{\text{new}} = w_{\text{old}} - \eta \times \text{gradient}$$

Where $\eta$ (eta) is the **learning rate** — how big a step to take.

**Analogy:** You're standing on a hilly landscape in thick fog, trying to find the lowest point. You can't see anything, but you *can feel* which direction is downhill under your feet (that's the gradient). So you take a small step downhill. Repeat. Eventually you reach a valley.

**Why it matters:** This is **the** learning algorithm. Every neural network — from MicroGPT to GPT-4 — learns by gradient descent. The only difference is the details of how step sizes are computed (see Adam optimizer, section 17).

---

## 15. Backpropagation — Computing All Gradients at Once

**Backpropagation** (backprop) is the efficient algorithm for computing gradients of the loss with respect to every parameter in the network.

The naive approach — nudge each parameter one at a time and measure the change — would require one forward pass per parameter. With 5,000 parameters, that's 5,000 forward passes per training step. Unacceptable.

Backprop does it in **two passes total**:
1. **Forward pass:** Compute the output and loss (as normal)
2. **Backward pass:** Walk backward through the computation graph, applying the chain rule at each node

After the backward pass, every parameter knows its gradient. This is the `Backward()` method in `Value.cs`.

**How it works step by step:**

```
Input → [op1] → [op2] → [op3] → Loss
```

1. Start at the loss. Its gradient with respect to itself is 1.
2. Go to op3. Using the chain rule, compute how much each of op3's inputs affects the loss.
3. Go to op2. Same thing. The gradient "flows backward" through each operation.
4. Continue until you've reached every parameter.

**Why it matters:** Backprop is what makes training neural networks practical. Without it, we couldn't efficiently compute how to adjust thousands (or billions) of parameters.

---

## 16. Learning Rate

The **learning rate** ($\eta$) controls how big each parameter update step is.

- **Too high:** Parameters overshoot the optimal values. The loss jumps around wildly or explodes.
- **Too low:** Training is too slow. The model barely moves toward better values.
- **Just right:** The loss decreases smoothly over time.

MicroGPT uses a learning rate of $0.01$ and applies **linear decay** — the learning rate starts at full strength and linearly decreases to 0 by the end of training. This makes sense intuitively: make big moves to explore at the start, then make smaller, more careful moves to fine-tune.

$$\eta_t = \eta_0 \times \left(1 - \frac{t}{T}\right)$$

Where $t$ is the current step and $T$ is the total number of steps.

**Why it matters:** The learning rate is one of the most important settings in training. Too high and nothing works. Too low and you wait forever. Learning rate schedules (like linear decay) help get the best of both worlds.

---

## 17. The Adam Optimizer

**Adam** (Adaptive Moment Estimation) is a smarter version of gradient descent. Plain gradient descent uses the raw gradient directly. Adam improves on this in two ways:

**1. Momentum (first moment, $m$)**

Instead of using just the current gradient, Adam keeps a running average of recent gradients:

$$m_t = 0.9 \times m_{t-1} + 0.1 \times g_t$$

Where $g_t$ is the current gradient. This smooths out noisy gradients and helps the optimizer build momentum in consistent directions — like a ball rolling downhill that builds speed.

**2. Adaptive learning rate (second moment, $v$)**

Adam also tracks how much each parameter's gradient varies:

$$v_t = 0.95 \times v_{t-1} + 0.05 \times g_t^2$$

Parameters with consistently large gradients get smaller updates. Parameters with small, noisy gradients get larger updates. Each parameter effectively gets its own tuned learning rate.

**The update rule:**

**3. Bias correction**

Since $m$ and $v$ are initialized to zero, they're biased toward zero during early steps. Adam corrects this:

$$\hat{m}_t = \frac{m_t}{1 - 0.9^t} \qquad \hat{v}_t = \frac{v_t}{1 - 0.95^t}$$

At step 1, $1 - 0.9^1 = 0.1$, so $\hat{m}$ is 10× larger than $m$ — compensating for the zero initialization. As $t$ grows, the correction factor approaches 1 and vanishes.

**The update rule:**

$$w_{\text{new}} = w_{\text{old}} - \eta \times \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

The $\epsilon$ (a tiny number like $10^{-8}$) prevents division by zero.

**Why it matters:** Adam is the default optimizer for almost all modern neural networks. It handles the learning rate tuning problem automatically — you set one global learning rate, and Adam adapts it per-parameter.

---

## 18. Activation Functions

An **activation function** is a simple nonlinear function applied after matrix multiplication. Without it, stacking layers would be pointless — multiple matrix multiplications in a row can always be collapsed into a single matrix multiplication. The nonlinearity is what gives neural networks their power.

**ReLU (Rectified Linear Unit):** The simplest and most popular.

$$\text{ReLU}(x) = \max(0, x)$$

- If $x > 0$: output is $x$ (pass through unchanged)
- If $x < 0$: output is $0$ (block it)

```
Input:   -3  -1   0   2   5
Output:   0   0   0   2   5
```

It acts like a gate — only positive signals pass through. This simple rule, applied across thousands of numbers, lets the network learn complex patterns.

**Squared ReLU:** What MicroGPT uses. It's ReLU followed by squaring:

$$\text{SquaredReLU}(x) = (\max(0, x))^2$$

The squaring makes it smoother and more selective — it emphasizes larger values more than smaller ones.

**Why it matters:** Activation functions are what make each layer do something that a single matrix multiplication can't. They're the reason deep networks can learn complex patterns.

---

## 19. Softmax — Turning Numbers into Probabilities

**Softmax** converts a list of arbitrary numbers (called **logits**) into probabilities that sum to 1.

$$\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}$$

**Step by step example:**

Logits: $[2.0, 1.0, 0.1]$

1. Exponentiate each: $[e^{2.0}, e^{1.0}, e^{0.1}] = [7.39, 2.72, 1.11]$
2. Sum: $7.39 + 2.72 + 1.11 = 11.22$
3. Divide each by sum: $[7.39/11.22, 2.72/11.22, 1.11/11.22] = [0.66, 0.24, 0.10]$

Result: probabilities $[0.66, 0.24, 0.10]$ that sum to $1.0$.

**Properties:**
- Larger inputs get exponentially larger probabilities
- All outputs are positive (because $e^x > 0$)
- Outputs sum to 1 (because we divide by the total)

**The max-subtraction trick:** In practice, $e^{1000}$ causes overflow. So we subtract the maximum value from all logits before exponentiating. This doesn't change the result (it cancels out in the division) but keeps the numbers manageable.

**Why it matters:** The model's final output is raw scores (one per character). Softmax converts them into a probability distribution — "35% chance the next letter is 'a', 20% chance it's 'm', ..." — which we can then sample from or compare against the correct answer.

---

## 20. Tokenization

**Tokenization** is converting text into numbers that the model can process.

In MicroGPT, each character becomes a number:

```
<BOS> = 0    (Beginning Of Sequence — "a name starts here")
<EOS> = 1    (End Of Sequence — "this name is done")
a = 2, b = 3, c = 4, ..., z = 27
```

The name "emma" becomes the sequence: `[0, 6, 14, 14, 2, 1]` → BOS, e, m, m, a, EOS.

**BOS** and **EOS** are special markers. BOS tells the model "start predicting a name." EOS tells it "this name is finished." Without EOS, the model wouldn't know when to stop generating.

**Real GPTs** use the same concept but with a much larger vocabulary (~100,000 tokens). Instead of single characters, they use word pieces — common words are one token ("the"), rare words are split into pieces ("un" + "believ" + "able").

**Why it matters:** Neural networks can only process numbers. Tokenization is the bridge between human-readable text and the model's numerical world. The vocabulary size determines how many possible outputs the model has at each step.

---

## 21. Embeddings — Numbers That Represent Meaning

An **embedding** is a learned vector (list of numbers) that represents a token. Instead of feeding the raw token ID (just a number like 7) into the model, we look up its embedding — a richer representation.

```
Token 'a' (ID 2)  → [0.12, -0.34, 0.56, 0.78, ...]    (16 numbers)
Token 'b' (ID 3)  → [-0.23, 0.45, 0.01, -0.67, ...]   (16 numbers)
Token 'm' (ID 15) → [0.89, 0.11, -0.44, 0.33, ...]    (16 numbers)
```

These vectors are **not designed by a human.** They start as random numbers and are adjusted during training. After training, tokens with similar roles develop similar vectors. For example, vowels might cluster together in this vector space.

**Position embeddings** work the same way — each position (0, 1, 2, ...) gets its own learned vector. The token embedding and position embedding are added together, so the model knows both *what* the token is and *where* it appears.

**Why it matters:** Embeddings are how the model represents discrete symbols (letters) as continuous numbers that can be processed by matrix multiplication and differentiated by backpropagation.

---

## 22. Sequence Models and Language Modeling

A **language model** predicts the next token given the previous tokens. That's the entire goal.

Given: "E", "m", "m" → Predict: "a" (with high probability)

**Training:** Show the model real names and ask it to predict each next character. Compare its prediction to the actual next character. Compute the loss. Backpropagate. Update weights. Repeat.

For the name "emma" (tokens: BOS, e, m, m, a, EOS), we get 5 prediction tasks:
- Given [BOS] → predict e
- Given [BOS, e] → predict m
- Given [BOS, e, m] → predict m
- Given [BOS, e, m, m] → predict a
- Given [BOS, e, m, m, a] → predict EOS

Each prediction is a forward pass through the model. The loss is averaged across all positions.

**Why it matters:** This is exactly what GPT does — at every scale, from MicroGPT to ChatGPT. The only difference is the size of the model and the amount of training data. ChatGPT was trained on trillions of words, so it learned vastly more complex patterns. But the core algorithm is identical.

---

## 23. Attention — "What Should I Focus On?"

**Attention** is the mechanism that lets the model look at all previous tokens and decide which ones are relevant for predicting the next one.

**The restaurant analogy:**

Imagine you're at a restaurant ordering food.
- **Query (Q):** Your question — "What's good for a vegetarian?"
- **Key (K):** Each menu item's description — "Grilled chicken", "Pasta primavera", "Beef stew"
- **Value (V):** The actual food you'd get if you picked that item

You compare your Query against each Key (the dot product). Items whose Keys match your Query well get high scores. Then you get a weighted blend of the Values — mostly the items that matched well, with a little bit of everything else.

**In the model:**

For each token, three projections are computed:
- **Q (Query):** "What information am I looking for?" — computed by multiplying the token's vector by a weight matrix
- **K (Key):** "What information do I contain?" — computed by multiplying by a different weight matrix
- **V (Value):** "What information can I offer?" — computed by multiplying by a third weight matrix

The attention score between two tokens is the dot product of one token's Q and another token's K:

$$\text{score}(i, j) = \frac{Q_i \cdot K_j}{\sqrt{d}}$$

The division by $\sqrt{d}$ (where $d$ is the vector dimension) keeps the scores from getting too large, which would make softmax output nearly one-hot (too confident about one token).

These scores are passed through softmax to get weights that sum to 1. The output is a weighted sum of all V vectors.

**Causality:** The model should only attend to past tokens, not future ones (that would be cheating). In MicroGPT, this is enforced naturally — tokens are processed one at a time and only past tokens' keys and values are in the cache.

**Why it matters:** Attention is **the** key innovation of the Transformer architecture. Before attention, models processed sequences position-by-position with no direct connection between distant tokens. Attention lets every token directly look at every other (past) token, enabling the model to capture long-range patterns like "names that start with 'Ch' often end in a certain way."

---

## 24. Multi-Head Attention

**Multi-head attention** runs several independent attention mechanisms in parallel, each on a different slice of the vector.

If the embedding size is 16 and we have 4 heads, each head works on a 4-number slice:
- Head 0: dimensions 0–3
- Head 1: dimensions 4–7
- Head 2: dimensions 8–11
- Head 3: dimensions 12–15

Each head has its own Q, K, V weight matrices and can learn to focus on different patterns:
- One head might track vowel-consonant patterns
- Another might track name length
- Another might focus on common letter pairs

After all heads compute their outputs, the results are concatenated back together and projected through one more matrix.

**Why it matters:** A single attention head can only focus on one pattern at a time. Multiple heads let the model attend to several different types of relationships simultaneously, making it much more expressive.

---

## 25. Residual Connections

A **residual connection** (or skip connection) means adding the input of a block back to its output:

$$\text{output} = \text{Block}(x) + x$$

Instead of the layer completely replacing the input, it only needs to learn the *difference* (the residual) from the input.

**Why this helps:**

1. **Information flow:** If a layer has nothing useful to add, it can output near-zero, and the input passes through unchanged. The model doesn't have to "re-learn" information that was already there.

2. **Gradient flow:** During backpropagation, the addition operation lets gradients flow straight through to earlier layers without being modified. Without residual connections, gradients can shrink (vanish) as they pass through many layers, making deep networks impossible to train.

**In MicroGPT:** Every attention block and every MLP block has a residual connection. The pattern is: save input → process → add saved input back.

**Why it matters:** Residual connections are what made deep networks practical. Before them (ResNet, 2015), networks deeper than ~20 layers were nearly impossible to train. With them, networks with hundreds of layers work fine.

---

## 26. Normalization

**Normalization** rescales numbers to keep them in a healthy range.

As data flows through many layers, values can grow very large or shrink to near zero. Both are bad — large values cause numerical overflow, and tiny values cause underflow. Either way, training becomes unstable.

**RMSNorm** (Root Mean Square Normalization) — used in MicroGPT:

1. Compute the average squared value: $\text{ms} = \frac{1}{n}\sum x_i^2$
2. Scale all values so the average squared magnitude is ~1: $\hat{x}_i = \frac{x_i}{\sqrt{\text{ms} + \epsilon}}$

The $\epsilon$ (a tiny number like $10^{-5}$) prevents division by zero.

**Analogy:** Think of it like an automatic volume control on a microphone. Whether someone whispers or shouts, the output level stays consistent.

**Why it matters:** Normalization is applied before every attention and MLP block. Without it, training deeper models (more layers) becomes extremely difficult because values accumulate and drift to extreme ranges.

---

## 27. The Transformer Architecture

The **Transformer** is the architecture used by all GPT models. Now you know every piece — here's how they fit together:

```
Input token
    ↓
[Token Embedding + Position Embedding]
    ↓
┌──────── Transformer Layer (repeated N times) ────────┐
│                                                       │
│   ┌─── Residual Connection ───┐                       │
│   ↓                           │                       │
│   RMSNorm                     │                       │
│   ↓                           │                       │
│   Multi-Head Attention        │                       │
│   ↓                           │                       │
│   + ←─────────────────────────┘                       │
│   ↓                                                   │
│   ┌─── Residual Connection ───┐                       │
│   ↓                           │                       │
│   RMSNorm                     │                       │
│   ↓                           │                       │
│   MLP (expand → activate →    │                       │
│        compress)              │                       │
│   ↓                           │                       │
│   + ←─────────────────────────┘                       │
│                                                       │
└───────────────────────────────────────────────────────┘
    ↓
[Output projection → logits for each token in vocabulary]
    ↓
[Softmax → probabilities]
```

Each transformer layer does two things:
1. **Attention:** Look at the past and decide what's relevant
2. **MLP:** Process the gathered information

Both have residual connections and normalization. Stack more layers for deeper "reasoning."

**Why it matters:** This is the complete GPT architecture. Every piece — embeddings, attention, MLP, normalization, residual connections — is something you've already seen. The transformer just combines them in this specific pattern.

---

## 28. Autoregressive Generation

**Autoregressive generation** means generating one token at a time, where each generated token becomes input for the next step.

```
Step 1: Input [BOS]        → Model predicts probabilities → Sample 'S'
Step 2: Input [BOS, S]     → Model predicts probabilities → Sample 'a'
Step 3: Input [BOS, S, a]  → Model predicts probabilities → Sample 'm'
Step 4: Input [BOS, S, a, m] → Model predicts probabilities → Sample EOS → Stop
Result: "Sam"
```

The **sampling** step is probabilistic — the model doesn't always pick the highest-probability token. It samples randomly according to the distribution, which is why you get different outputs each time.

**KV Cache:** A performance optimization. When generating token 4, the model needs to attend to tokens 1, 2, and 3. But the Key and Value vectors for tokens 1–3 were already computed in earlier steps. The KV cache stores them so they don't need to be recomputed. Only the new token's Q, K, V need to be calculated.

**Why it matters:** This is exactly how ChatGPT generates text. Every word you see streaming in real-time is one token being generated, fed back in, and used to generate the next. MicroGPT does the same thing, just with characters instead of words.

---

## 29. Putting It All Together

Here's the complete picture of what happens when you run MicroGPT:

**Setup:**
1. Download 32,000 human names
2. Build a tokenizer (28 characters + BOS + EOS)
3. Initialize random weight matrices (3,648 numbers with default settings)

**Training (1,000 steps):**
1. Pick a name (e.g., "Emma")
2. Tokenize it: [BOS, E, m, m, a, EOS]
3. For each position, run the model forward to predict the next character
4. Compute the loss (how wrong was the prediction?)
5. Run backpropagation to compute gradients (how should each weight change?)
6. Use Adam optimizer to update all weights
7. Print the loss and repeat

**Generation (5 samples):**
1. Start with BOS
2. Run the model forward → get probabilities for the next character
3. Randomly sample a character
4. Feed it back into the model → repeat until EOS or max length
5. Print the generated name

**That's it.** Every conceptual piece of ChatGPT is here. The only differences are scale (more parameters, more data, more compute) and engineering optimizations (GPU parallelism, efficient attention algorithms, etc.).

---

## 30. Further Reading

**Math foundations:**
- [3Blue1Brown — Essence of Calculus](https://www.youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr) — Beautiful visual explanations of derivatives and the chain rule
- [3Blue1Brown — Essence of Linear Algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab) — Vectors and matrices explained visually

**Neural networks from scratch:**
- [3Blue1Brown — Neural Networks](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) — Excellent visual introduction
- [Karpathy — Neural Networks: Zero to Hero](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ) — Complete video series building up to GPT from scratch

**Transformers and GPT:**
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — Step-by-step visual walkthrough
- [Attention Is All You Need (2017)](https://arxiv.org/abs/1706.03762) — The original transformer paper
- [GPT-2 Paper](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) — The GPT-2 paper

**The original project:**
- [Karpathy's microgpt.py](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95) — The Python source this project is ported from
- [Karpathy's micrograd](https://github.com/karpathy/micrograd) — The autograd engine it builds on
