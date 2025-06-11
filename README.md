# csc413-2516---homework-1-solved
**TO GET THIS SOLUTION VISIT:** [CSC413-2516 – Homework 1 Solved](https://mantutor.com/product/csc413-2516-homework-1-solved/)


---

**For Custom/Order Solutions:** **Email:** mantutorcodes@gmail.com  

*We deliver quick, professional, and affordable assignment help.*

---

<h2>Description</h2>



<div class="kk-star-ratings kksr-auto kksr-align-center kksr-valign-top" data-payload="{&quot;align&quot;:&quot;center&quot;,&quot;id&quot;:&quot;105251&quot;,&quot;slug&quot;:&quot;default&quot;,&quot;valign&quot;:&quot;top&quot;,&quot;ignore&quot;:&quot;&quot;,&quot;reference&quot;:&quot;auto&quot;,&quot;class&quot;:&quot;&quot;,&quot;count&quot;:&quot;2&quot;,&quot;legendonly&quot;:&quot;&quot;,&quot;readonly&quot;:&quot;&quot;,&quot;score&quot;:&quot;5&quot;,&quot;starsonly&quot;:&quot;&quot;,&quot;best&quot;:&quot;5&quot;,&quot;gap&quot;:&quot;4&quot;,&quot;greet&quot;:&quot;Rate this product&quot;,&quot;legend&quot;:&quot;5\/5 - (2 votes)&quot;,&quot;size&quot;:&quot;24&quot;,&quot;title&quot;:&quot;CSC413-2516 - Homework 1 Solved&quot;,&quot;width&quot;:&quot;138&quot;,&quot;_legend&quot;:&quot;{score}\/{best} - ({count} {votes})&quot;,&quot;font_factor&quot;:&quot;1.25&quot;}">

<div class="kksr-stars">

<div class="kksr-stars-inactive">
            <div class="kksr-star" data-star="1" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="2" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="3" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="4" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="5" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>

<div class="kksr-stars-active" style="width: 138px;">
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>
</div>


<div class="kksr-legend" style="font-size: 19.2px;">
            5/5 - (2 votes)    </div>
    </div>
Version: 1.1

Q1.3: Change W(3) ∈ R4 to W(3) ∈ R ×4,

Q2.2.2: Change ,

Q2.2.2: Fix missing transpose,

Q3.2.2: Change ,

Q3.2.2: Change “in terms of n and σ” to “in terms of n, d and σ”

Submission: You must submit your solutions as a PDF file through MarkUs1. You can produce the file however you like (e.g. LaTeX, Microsoft Word, scanner), as long as it is readable.

1 Hard-Coding Networks

Can we use neural networks to tackle coding problems? Yes! In this question, you will build a neural network to find the kth smallest number from a list using two different approaches: sorting and counting (Optional). You will start by constructing a two-layer perceptron “Sort 2” to sort two numbers and then use it as a building block to perform your favorite sorting algorithm (e.g., Bubble Sort, Merge Sort). Finally, you will output the kth element from the sorted list as the final answer.

1.1 Sort two numbers [1pt]

Please specify the weights and activation functions for your network. Your answer should include:

• Two weight matrices: W(1), W(2) ∈ R2×2

• Two bias vector: b(1), b(2) ∈ R2

• Two activation functions: ϕ(1)(z), ϕ(2)(z) You do not need to show your work.

Hints: Sorting two numbers is equivalent to finding the min and max of two numbers.

,

1.2 Perform Sort [1.5pt]

Draw a computation graph to show how to implement a sorting function fˆ : R4 → R4 where fˆ(x1,x2,x3,x4) = (xˆ1,xˆ2,xˆ3,xˆ4) where (ˆx1,xˆ2,xˆ3,xˆ4) is (x1,x2,x3,x4) in sorted order. Let us assume ˆx1 ≤ xˆ2 ≤ xˆ3 ≤ xˆ4 and x1,x2,x3,x4 are positive and distinct. Implement fˆ using your favourite sorting algorithms (e.g. Bubble Sort, Merge Sort). Let us denote the “Sort 2” module as S, please complete the following computation graph. Your answer does not need to give the label for intermediate nodes, but make sure to index the “Sort 2” module.

Hints: Bubble Sort needs 6 “Sort 2” blocks, while Merge Sort needs 5 “Sort 2” blocks.

1.3 Find the kth smallest number [0pt]

Hints: W(3) ∈ R1×4 .

1.4 Counting Network [0pt]

The idea of using a counting network to find the kth smallest number is to build a neural network that can determine the rank of each number and output the number with the correct rank. Specifically, the counting network will count how many elements in a list are less than a value of interest. And you will apply the counting network to all numbers in the given list to determine their rank. Finally, you will use another layer to output the number with the correct rank.

The counting network has the following architecture, where y is the rank of x1 in a list containing x1,x2,x3,x4.

Please specify the weights and activation functions for your counting network. Draw a diagram to show how you will use the counting network and give a set of weights and biases for the final layer to find the kth smallest number. In other words, repeat the process of sections 1.1, 1.2, 1.3 using the counting idea.

2) Indicator activation function:

1 if z ∈ [−1,1]

0 otherwise

2 Backpropagation

This question helps you to understand the underlying mechanism of back-propagation. You need to have a clear understanding of what happens during the forward pass and backward pass and be able to reason about the time complexity and space complexity of your neural network. Moreover, you will learn a commonly used trick to compute the gradient norm efficiently without explicitly writing down the whole Jacobian matrix.

2.1 Automatic Differentiation

Consider a neural network defined with the following procedure:

z1 = W(1)x + b(1)

h1 = ReLU(z1)

z2 = W(2)x + b(2)

h2 = σ(z2)

g = h1 ◦ h2 y = W(3)g + W(4)x,

y′ = softmax(y)

N

S = XI(t = k)log(yk′ )

k=1

J = −S

for input x with class label t where ReLU(z) = max(z,0) denotes the ReLU activation function, z denotes the Sigmoid activation function, both applied elementwise, and softmax(y) = . Here, ◦ denotes element-wise multiplication.

2.1.1 Computational Graph [0pt]

Draw the computation graph relating x, t, z1, z2, h1, h2 , g, y, y′, S and J.

2.1.2 Backward Pass [1pt]

Derive the backprop equations for computing x¯ = ∂∂Jx ⊤, one variable at a time, similar to the vectorized backward pass derived in Lec 2.

Hints: Be careful about the transpose and shape! Assume all vectors (including error vector) are column vector and all Jacobian matrices adopt numerator-layout notation . You can use softmax′(y) for the Jacobian matrix of softmax.

2.2 Gradient Norm Computation

Many deep learning algorithms require you to compute the L2 norm of the gradient of a loss function with respect to the model parameters for every example in a minibatch. Unfortunately, most differentiation functionality provided by most software frameworks (Tensorflow, PyTorch) does not support computing gradients for individual samples in a minibatch. Instead, they only give one gradient per minibatch that aggregates individual gradients for you. A naive way to get the perexample gradient norm is to use a batch size of 1 and repeat the back-propagation N times, where N is the minibatch size. After that, you can compute the L2 norm of each gradient vector. As you can imagine, this approach is very inefficient. It can not exploit the parallelism of minibatch operations provided by the framework.

In this question, we will investigate a more efficient way to compute the per-example gradient norm and reason about its complexity compared to the naive method. For simplicity, let us consider the following two-layer neural network.

z = W(1)x

h = ReLU(z) y = W(2)h,

where W and W .

2.2.1 Naive Computation [1pt]

Let us assume the input and the error vector . In this question, write down the Jacobian matrix (numerical value) and using back-propagation.

Then, compute the square of Frobenius Norm of the two Jacobian matrices, . The square of Frobenius norm of a matrix A is defined as follows:

m n

∥A∥2F = XX|aij|2 = trace

i=1 j=1

2.2.2 Efficient Computation [0.5pt]

Notice that weight Jacobian can be expressed as the outer product of the error vector and activation and . We can compute the Jacobian norm more efficiently using

the following trick:

!

= trace(Definition)

= trace

= trace (Cyclic Property of Trace)

(Scalar Multiplication)

Compute the square of the Frobenius Norm of the two Jacobian matrices by plugging the value into the above trick.

2.2.3 Complexity Analysis [1.5pt]

Now, let us consider a general neural network with K − 1 hidden layers (K weight matrices). All input units, output units, and hidden units have a dimension of D. Assume we have N input vectors. How many scalar multiplications T (integer) do we need to compute the per-example gradient norm using naive and efficient computation, respectively? And, what is the memory cost M (big O notation)?

For simplicity, you can ignore the activation function and loss function computation. You can assume the network does not have a bias term. You can also assume there are no in-place operations. Please fill up the table below.

T (Naive) T (Efficient) M (Naive) M (Efficient)

Forward Pass

Backward Pass

Gradient Norm Computation

2.3 Inner product of Jacobian: JVP and VJP [0pt]

A more general case of computing the gradient norm is to compute the inner product of the Jacobian matrices computed using two different examples. Let f1,f2 and y1,y2 be the final outputs and layer outputs of two different examples respectively. The inner product Θ of Jacobian matrices of layer parameterized by θ is defined as:

,

O×YY×P P×Y Y×O

Where O,Y,P represent the dimension of the final output, layer output, model parameter respectively. How to formulate the above computation using Jacobian Vector Product (JVP) and Vector Jacobian Product (VJP)? What are the computation cost using the following three ways of contracting the above equation?

(a) Outside-in: M1M2M3M4 = ((M1M2)(M3M4))

(b) Left-to-right and right-to-left: M1M2M3M4 = (((M1M2)(M3)M4) = (M1(M2(M3M4)))

(c) Inside-out-left and inside-out-right: M1M2M3M4 = ((M1(M2M3))M4) = (M1((M2M3)M4))

3 Linear Regression

Given n pairs of input data with d features and scalar label (xi,ti) ∈ Rd × R, we wish to find a linear model f(x) = wˆ ⊤x with wˆ ∈ Rd that minimizes the squared error of prediction on the training samples defined below. This is known as an empirical risk minimizer. For concise notation, denote the data matrix X ∈ Rn×d and the corresponding label vector t ∈ Rn. The training objective is to minimize the following loss:

n

1 ⊤ 2 1 2 t X .

n

We assume X is full rank: X⊤X is invertible when n &gt; d, and XX⊤ is invertible otherwise. Note that when d &gt; n, the problem is underdetermined, i.e. there are less training samples than parameters to be learned. This is analogous to learning an overparameterized model, which is common when training of deep neural networks.

3.1 Deriving the Gradient [0pt]

Write down the gradient of the loss w.r.t. the learned parameter vector wˆ.

3.2 Underparameterized Model

3.2.1 [0.5pt]

First consider the underparameterized d &lt; n case. Show that the solution obtained by gradient descent is wˆ = (X⊤X)−1X⊤t, assuming training converges. Show your work.

3.2.2 [0.5pt]

Now consider the case of noisy linear regression. The training labels ti = w∗⊤xi +ϵi are generated by a ground truth linear target function, where the noise term, ϵi, is generated independently with zero mean and variance σ2. The final training error can be derived as a function of n and ϵ, as:

Error ,

Show this is true by substituting your answer from 3.2.1 into . Also, find the expectation of the above training error in terms of n,d and σ.

Hints: you might find the cyclic property of trace useful.

3.3 Overparameterized Model

3.3.1 [0pt]

Now consider the overparameterized d &gt; n case. We first illustrate that there exist multiple empirical risk minimizers. For simplicity we let n = 1 and d = 2. Choose x1 = [1;1] and t1 = 3, i.e. the one data point and all possible wˆ lie on a 2D plane. Show that there exists infinitely many wˆ satisfying wˆ ⊤x1 = y1 on a real line. Write down the equation of the line.

3.3.2 [1pt]

Now, let’s generalize the previous 2D case to the general d &gt; n. Show that gradient descent from zero initialization i.e. wˆ(0) = 0 finds a unique minimizer if it converges. Show that the solution by gradient decent is wˆ = X⊤(XX⊤)−1t. Show your work.

Hints: You can assume that the gradient is spanned by the rows of X and write wˆ = X⊤a for some a ∈ Rn.

3.3.3 [0pt]

Repeat part 3.2.2 for the overparameterized case.

3.3.4 [0.5pt]

Visualize and compare underparameterized with overparameterized polynomial regression: https:

3.3.5 [0pt]

Give n1, n2 with n1 ≤ n2, and fixed dimension d for which L2 ≥ L1, i.e. the loss with n2 data points is greater than loss with n1 data points. Explain the underlying phenomenon. Be sure to also include the error values L1 and L2 or provide visualization in your solution.

Hints: use your code to experiment with relevant parameters, then vary to find region and report one such setting.
