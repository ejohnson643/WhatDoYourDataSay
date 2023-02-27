# Worksheet 1.3: Bayes' Theorem and the Effect of Priors

As a reminder, the worksheets are designed to work in parallel to the course notes and lectures.  They are not intended to be summary assessments at the end of a module, but as **necessary content** that you consume on your way to the assignments (and eventually attacking your own data!).  As a result, if there are parts of the worksheet that you don't understand or are having trouble with, make sure to **ask for help!**

In this worksheet, you will practice implementing the concepts enumerated in the Module One course notes, particularly **the section on Bayes' Theorem**.  You should complete this worksheet in your own Jupyter notebook and try to complete **every question**.  Even if you think that you are proficient in Python coding, you should make sure that you can write working answers to all the parts of this problem - it will be useful code for later parts of the course, so your effort won't be wasted!

```{admonition} Worksheet Goals
By the end of this worksheet, you should be able to:
 - <mark style="background-color: #6b9cee">MVS</mark> <mark style="background-color: #e26563">PCS</mark> <mark style="background-color: #ffda5c">TS</mark> Implement Bayes' Theorem in Python to generate a posterior distribution for the probability of a coin flip being heads, $p$.
 - <mark style="background-color: #6b9cee">MVS</mark> <mark style="background-color: #e26563">PCS</mark> Normalize curves to convert them to PDFs
 - <mark style="background-color: #e26563">PCS</mark> Approximate the area under a curve.
```

1. Using your code from worksheet 1.1, simulate $N=100$ tosses of a coin with $p=P(\text{HEADS})=0.4$.  That is, record whether each of the $N=100$ tosses was a heads or a tails.
    
2. Let's pretend that you didn't know that $p=0.4$, and you wanted to use Bayes' Theorem to examine the likelihood that the coin's heads probability was any given value as you flip the coin.  (This is the process illustrated in Section 3.1 of the notes.)
    1. Write down Bayes' Theorem in terms of the variables in this problem.  Specifically, what are the posterior, the likelihood, and the prior dependent on?
        
        ```{note}
        Just because we're coding doesn't mean that pen and paper aren't useful!  Writing down formulas and equations in terms of quantities and variables that mean something to you can be very helpful for keeping your head on straight.
        ```
        
    2. Consider the following statements:
        1.  The likelihood that the first two coin tosses were heads given $p=0.3$ is 0.09.
        2. Before flipping any coins, the likelihood that $p=0.3$ is the same as the likelihood that $p=0.5$.
        3.  The likelihood that $p=0.3$ given that the first two coin tosses were heads is 0.0014.
        
        Each of these quantities is an example of statements that can be made with each of the three components of Bayes' Theorem (posterior, likelihood, prior).  **Match** each statement to the component of Bayes' Theorem that was used to generate it.

    3. If you are given a list of heads and tails from a coin tossing experiment.  Which quantities are *known* and which are *\textit{*unknown*: $k$, the number of heads, $N$, the number of flips, or $p$ the likelihood of heads?  Are there other quantities in this problem that we aren't considering?  Why does the statement of Bayes' Theorem at the beginning of Section 3.1 of the notes allow us to measure the *unknown* quantity?
        
        ```{note}
        Taking stock of what you do an don't know is often helpful when building code, and in solving problems in general!  Also, formulas that involve quantities that you don't know are usually the ones you need to focus on.  We're not using Bayes' Theorem just to use it; we're using it because it allows us to extract what we want out of our simulations.
        ```
    
    At this point you may be feeling that you have a handle on what the different parts of Bayes' Theorem are telling you (or you may be completely lost, that's ok!), but you have no idea how to implement it in your computer.  The rest of this worksheet is dedicated to helping you figure out how this works.
    
4. Recall that in Python, we don't manipulate variables symbolically, we always have actual *numbers* behind any variable.  This is why plotting involves assembling literally a list of $x$- and $y$-coordinates.  In this way, if we wanted to plot a distribution that represented a **uniform prior distribution** for $p=P(\text{HEADS})$, what would be our $x$- and $y$-coordinates?
    1. Make a grid of values of $p$.  What is the maximum allowed value and what is the minimum?  How can you control the resolution of this grid?  What is an appropriate function in Python for this task?
        
    2. If you were to use this grid to plot a distribution, would this grid be the $x$- or $y$-coordinates?
        
    3. What do the $y$-coordinates of a plotted probability distribution represent?  If my distribution is **uniform**, do the $y$-coordinates differ?
        
    4. Plot your $y$-coordinates vs. your $x$-coordinates.  Does your plot resemble a uniform distribution of $p$?
        
    5. Add up the area under your curve.  If you have used uniform grid-spacing, then this area will be the sum of the $y$-coordinates multiplied by your grid spacing, $\Delta x$.  If you have not used uniform spacing, you can approximate the area by summing $y_i\times \Delta x_i$ for each point $i$ in your plot.
        
    6. Divide your $y$-coordinates by an appropriate factor so that the area under your line is 1.  Justify why this makes the line represent a probability density function for $p$.
        
    7. Which part of Bayes' Theorem does this curve represent?
    
5. Consider the first 5 simulated flips of your coin, so that $N=5$.

    1. What is $k$, the number of heads?

    2. For each value of $p$ in your grid from the previous problem, what is the likelihood of observing $k$ heads in $N=5$ flips?  Save these likelihoods in an list or array.

    3. Plot these likelihoods versus the grid of values of $p$.

    4. What part of Bayes' Theorem does this curve represent?  (Hint: it's on the right-hand side of the formula.)
    
6. So now we have two likelihood arrays "on" the same grid of values of $p$.  If you deduced that these arrays represent the likelihood and prior distributions, you are correct!  Bayes' Theorem thus tells us that to get the posterior distribution, we should multiply the likelihood and prior functions; do that now.  Normalize the resulting array so that the area under the curve is 1.  You should now have the posterior distribution for $p$!  What is the most likely value for $p$?

---

This process is somewhat different and complicated, so for continued practice or if you're stuck (and to continue exploring Bayes' Theorem), download and complete the workbook [here](Worksheet_1_3_EffectOfPriors_Guide).  In this workbook you will find some useful code that will help you complete this worksheet and move on to replicating the figures in the notes and working on the assignment.

As always, if you're feeling too lost or it's not making a ton of sense.  Ask for help on the discussion boards or bring up some questions in the study sessions.  This topic in particular is a bit different from things you may have seen, so be patient with yourself!  Good luck!