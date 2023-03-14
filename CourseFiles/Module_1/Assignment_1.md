# Assignment 1

<p style="text-align: center;">Written by Eric Johnson and Madhav Mani</p>

<p style="text-align: center;">Most recently compiled on March 14, 2023</p>

---

**Assignment Due Dates** (Fall 2020):

 - The *Assignment Attempt* is due on October 2, 2020 at 11:59 PM CST
 - The *Complete Assignment* is due on October 9, 2020 at 11:59 PM CST

---

Please follow the guidelines for assignment [attempts](../HowTo_AssignmentAttempt) and [completion](../HowTo_AssignmentCompletion). In particular, the attempt must be a Jupyter notebook or PDF that clearly enumerates the problem attempts, and the completed assignment must be code *and* a PDF containing completed solutions. Please use complete sentences in your solutions. All figures should have labeled axes, legends, and appropriate annotations.

Please read the *entire assignment* before getting too deep into any one problem so that you can properly allocate your time and questions.

## Learning Objectives:

The module learning objectives assessed in this assignment are that a student will be able to:

- <mark style="background-color: #e26563">PCS-1</mark>: Calculate summary quantities from data or a distribution.
- <mark style="background-color: #e26563">PCS-2</mark>: Construct probability distributions.
- <mark style="background-color: #e26563">PCS-3</mark>: Implement simple algorithms, especially to generate synthetic data sets.
- <mark style="background-color: #ffda5c">TS</mark>: Understand and discuss core concepts in probability.
- <mark style="background-color: #6b9cee">MVD</mark>: Graph core concepts in probability.
- <mark style="background-color: #92c57a">NQP</mark>: Illustrate the effect of parameter dependence on distributions.

You can learn more about where to study and practice these skills in the [curriculum alignment table](./Module_1).

---

1. This first problem is focused on plotting and generating theoretical and empirical probability distributions. Complete the following tasks:
	1. <mark style="background-color: #e26563">PCS-2,3</mark>, <mark style="background-color: #6b9cee">MVD</mark> Make a two row and four column figure. In the top row, plot a binomial, Gaussian, Poisson, and uniform PDFs and in the bottom row plot the corresponding CDFs.  For each type of distribution, show 3 curves with varying parameters (even the uniform distribution!  What are the parameters of a uniform distribution?).  Include a legend that shows the values of the relevant parameters for each of the three curves.  Label the axes and provide titles above each column.
        
        ```{note}
		This problem is all about getting familiar with some basic tools in plotting and figure-making.  Make sure to look at the [`scipy.stats`](https://docs.scipy.org/doc/scipy/reference/stats.html) module when you're looking to generate these distributions.
        ```
            
    2. <mark style="background-color: #e26563">PCS-2,3</mark>, <mark style="background-color: #6b9cee">MVD</mark> Repeat the previous problem with *empirical* PDFs and CDFs in a new figure.  That is, generate data sets by sampling random variables from each of these distributions.  Use the same three sets of parameters for each type of distribution. You can choose how many samples to generate.  eCDFs *should not be binned*.
        
        ```{note}
        Again this problem might seem repetitive, but it will help you start to work out what parts of plotting make sense, and what you'll need to ask about.

        Hopefully this problem also underscores the difference between empirical and theoretical PDFs and CDFs.  Theoretical distributions are nice and smooth while empirical distribution are always somewhat noisy.  In this figure (and always), you should **never** use a binned histogram to form your CDF, you should use the code in the notes to get an *exact* CDF for your data.  This is important because putting data into bins is an arbitrary process and it destroys information.  When possible, you should always do calculations with eCDFs.
        ```
            
    3. <mark style="background-color: #e26563">PCS-1</mark> Report the mean, standard deviation, median, and IQR of each of the distributions in questions 1 and 2.  For the theoretical distributions, you can use theory to compute these values, but indicate that you are doing so.  The values for question 2 must be generated from the data shown in your figure.

2. In this problem we will consider the behavior of a [random walker](https://en.wikipedia.org/wiki/Random_walk) (although on a weekend night it may be a [drunken walker](https://medium.com/i-math/the-drunkards-walk-explained-48a0205d304)).  More specifically, we consider the case of a very strange person who, in moving down the road, flips a coin, and if the coin is heads, she takes a step to the left, and if it is tails, a step to the right.
    1. <mark style="background-color: #e26563">PCS-3</mark>, <mark style="background-color: #6b9cee">MVD</mark> Write a **function** that returns the path of such a random walker given a number of steps, $M$, and a coin bias $p$.  Plot the path of a single random walker generated by your function for $M=100$ and $p=0.5$.
            
    2. <mark style="background-color: #e26563">PCS-3</mark>, <mark style="background-color: #6b9cee">MVD</mark> Generate trajectories (paths) for $N=100$ walkers, each taking $M=150$ steps  with $P(left)=p=0.5$.  Create a plot showing all of these trajectories as a function of step number.
            
    3. <mark style="background-color: #e26563">PCS-1</mark>, <mark style="background-color: #6b9cee">MVD</mark> Calculate the mean and standard deviation of the position of the walkers at each of their $M=150$ steps.  Plot the mean and standard deviation on top of the trajectories (the `alpha` keyword might be useful to maintain figure clarity).  Describe (in words) how the mean and standard deviation scale with the number of steps.
            
    4. <mark style="background-color: #e26563">PCS-2</mark>, <mark style="background-color: #6b9cee">MVD</mark>, <mark style="background-color: #ffda5c">TS</mark> Plot the distribution of positions that the walkers have gotten to after 144 steps.  Using the Central Limit Theorem, explain why this distribution might be Gaussian.
            
    5. <mark style="background-color: #e26563">PCS-3</mark>, <mark style="background-color: #92c57a">NQP</mark> Repeat parts 1.3 and 1.4 for $N = 100$ walkers for $M = 150$ steps and $p = 0.75$.  Discuss what changes after *biasing* the walk in a specific direction.
    
    In fact, what you have just simulated is the **diffusion** of molecules in one dimension.  Molecules in gas can be reliably modeled as random walkers that move with some characteristic speed, quantified by their *diffusion constant*, $D$.  The probability of observing a molecule at a given position after they have been released at a central location can be written
    ```{math}
    P(x, t) = \frac{1}{\sqrt{4\pi D t}}e^{-\frac{x^2}{4Dt}},
    ```
    which you can verify for yourself is a Gaussian distribution in $x$.

    ```{note}
    This problem is meant to introduce you to the ways that subtle changes to how you simulate processes can allow you to gain intuition on a much wider variety of processes.  Reinterpreting a coin flip as a random walker has just allowed you to simulate molecular diffusion!

    You should also start to see how writing code and generating figures that can show you how results depend on *parameters* can be very useful.  In particular, going forward, you should always feel comfortable ignoring our parameter prescriptions to see what happens in different scenarios.  We'll generally ask you to look at interesting situations, but as we progress, more of this is left to you!
    ```

