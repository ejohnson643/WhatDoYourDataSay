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

3. In this problem we will work with some data from an experiment on bacterial chemotaxis, as discussed in the course notes.  You can download this data [here](https://github.com/ejohnson643/WhatDoYourDataSay/blob/main/CourseFiles/Module_2/Resources/omega.txt).  This link leads to a text file containing a list of numbers.  These numbers are the angular velocity of spinning bacteria recorded at a rate of 60 measurements per second.
        
    1. <mark style="background-color: #6b9cee">MVD</mark> Plot the entire list as a function of time-step (how many seconds in each time-step?).  Describe what you see.  How many values are there?  What are the largest and smallest values?  Is this a useful figure?  What might be a better way of looking at this data?  Show such a figure.
    
    	```{note}
    	Part of the reason we want you to become strong figure-makers is exactly this: when you get a new data set, you may have no idea what it will look like.  You will desperately hope that there are some obvious features, but often, like the data given here, it will look like a mess.  Using plotting techniques and different calculations can help you orient yourself.
    	```

    2. <mark style="background-color: #e26563">PCS-1</mark> Calculate the mean, standard deviation, median, and IQR of the angular velocity.  Report and comment on these values.
    
    3. <mark style="background-color: #e26563">PCS-3</mark> As mentioned in the text, these bacteria are attempting to move, so they are alternating between spinning one direction and another.  Verify that you see this behavior.  Assemble a list of the *lengths of the time intervals* that the bacterium is spending spinning one direction.  Assemble a similar list of interval lengths for when the bacterium is spinning the other direction.

    	```{note}
    	If you are new to programming, this task may seem daunting, but can readily be attacked if you step back and try and solve the problem on paper before returning to your notebook.  In particular, try to think about you, yourself, as a human going through this data one observation at a time.  How can *you* tell that the bacteria has switched direction?  Is there a simple criteria you can write down to make this same assessment?  Can you convert that criteria into something the computer understands?  If you are having trouble here, make sure to ask a question about **how to detect switching**; this isn't inherently a Python or coding problem!  This is part of the struggle of how to think quantitatively.
                
        Once you have done this step, the problem again becomes a coding problem: how are you going to set up your loop, what do you need to keep track of, where are you storing these intervals?  If you are having trouble here, ask about **coding**.  Tips on coding practices will likely help.
        
        Also make sure that you are checking the results that you get.  If you get negative time intervals, something has gone wrong.  If all your time intervals are the same, something has gone wrong.  If you don't get anything but your code runs without error, guess what?  Something has gone wrong.  This is ok!  This is a part of learning to code, and you should never feel like you are bad at this because it doesn't work the first time.  When this happens, insert some print statements, shorten your loop, and see what happens.  After a while, go back to the drawing board, see if a different code structure will help.  Finally, if it's been **30 minutes** and you have no idea what's going on, **ASK FOR HELP!**  (And move on to the next problem.)  This course is not meant to be an exercise in futility.
    	```

    4. <mark style="background-color: #e26563">PCS-2</mark>, <mark style="background-color: #6b9cee">MVD</mark> Plot the PDFs and CDFs of time intervals for each direction of spinning.  Calculate the mean, standard deviation, median, and IQR of both lists and format them as a table.  Comment on the values.

    ```{admonition} **Bonus problems**
    The remaining problems in this assignment are \textbf{bonus} problems.  You do not need to attempt or complete these problems to receive a "B" on the assignment.  To receive a higher grade you must acceptably complete most of the bonus problems.
    ```

    5.  **BONUS**: <mark style="background-color: #e26563">PCS-2</mark>, <mark style="background-color: #ffda5c">TS</mark> It turns out that the bacterium's switching from one direction to another is an example of a **Poisson process**.  One characteristic of a Poisson process is that the time interval between events (the switches from one direction to another) are **exponentially distributed**, that is
        ```{math}
        P(t) = \lambda e^{-\lambda t}.
        ```
        For an exponential distribution, the mean is given by $1/\lambda$.  Using your calculation of the mean, provide an estimate for $\lambda$.  You may use either list of intervals or both combined.
        
        ```{note}
        This is your first time (in this course) where you are confronting a model with data.  This is definitely something that you will be interested in doing with your own data, and this is the simplest: parameter estimation.  In the next module, we'll elaborate on this idea in much more detail.
        ```
            
    6. **BONUS**: <mark style="background-color: #e26563">PCS-2,3</mark>, <mark style="background-color: #6b9cee">MVD</mark>, <mark style="background-color: #ffda5c">TS</mark>, <mark style="background-color: #92c57a">NQP</mark> Let's say there is a paper that suggests that the value of $\lambda$ should be 0.2.  Use a Bayesian analysis (as shown in the course notes) to qualitatively assess whether this data are consistent with that result.  Perform the analysis using both a non-informative prior and an informed prior based on the result from the paper.  Show a series of figures as in the text where you have incorporated only the first time interval from your list, the first two, the first ten, etc.  Only show increments that are interesting (do not show more than 4-5 increments).  Qualitatively, how long does it take for the posterior to stop changing?  Based off these posteriors, does it seem that $\lambda = 0.2$ is consistent with your data?
        
        ```{note}
        If you can see past the daunting Bayesian component, you should notice that this problem practices something we've explored at various points: how does adding more data affect my outcomes?  Furthermore, the posing of the question hints at something you won't see until [Module 3](../Module_3/Module_3): **hypothesis testing**.  If you stick with it until then, determining whether our data are consistent with this paper will be a snap!
        ```
            
    7. **BONUS**: <mark style="background-color: #ffda5c">TS</mark> In the previous problem, you were encouraged to use all of your extracted time intervals, however, this might be overzealous as these time intervals are potentially not independent of one another!  Discuss and propose an approach for how you might work around this possible data-interdependence.  
            
    8. **BONUS**: <mark style="background-color: #92c57a">NQP</mark> Repeat either of the the previous analyses (1.5 or 1.6) and show whether the estimates of $\lambda$ differ after accounting for potential dependencies in the data.

