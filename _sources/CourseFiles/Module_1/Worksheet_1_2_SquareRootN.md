# Worksheet 1.2: The Spread in Distributions

As a reminder, the worksheets are designed to work in parallel to the course notes and lectures. They are not intended to be summary assessments at the end of a module, but as **necessary content** that you consume on your way to the assignments (and eventually attacking your own data!). As a result, if there are parts of the worksheet that you don’t understand or are having trouble with, make sure to **ask for help!**

In this worksheet, you will practice implementing the concepts enumerated in the first two main sections of the Module One course notes, **up to the section on Bayes' Theorem**. You should complete this worksheet in your own Jupyter notebook and try to complete every question. Even if you think that you are proficient in Python coding, you should make sure that you can write working answers to all the parts of this problem - it will be useful code for later parts of the course, so your effort won’t be wasted!

```{admonition} **Worksheet Goals:**
 By the end of this worksheet, you should be able to:
 - <mark style="background-color: #6b9cee">MVS</mark> Visualize results as a function of different parameters
 - <mark style="background-color: #e26563">PCS-2</mark> <mark style="background-color: #6b9cee">MVD</mark> Calculate and visualize empirical cumulative density functions
 - <mark style="background-color: #e26563">PCS-1</mark> Calculate percentiles
 - <mark style="background-color: #e26563">PCS-2</mark> <mark style="background-color: #6b9cee">MVD</mark> Visualize the “shapes” of distributions in different ways
```

At this point in the notes, we’ve talked about flipping coins, PDFs, CDFs, and the Central Limit Theorem, but what does this actually look like in practice? In particular, you may be feeling that the CLT is somewhat theoretical, even though we presented it in somewhat physical terms. In this worksheet we will use the coin flip again to help you make these theorems and mathematical definitions more concrete.

In particular, we appeal to common sense and point out that obviously it’s possible to get 8 heads out of 10 coin flips, even from a fair coin. But how often should we expect to see a deviation of 3 heads from the mean value of 5? What if we flip 20 coins, is a deviation of 3 heads more or less likely? The CLT answers this exactly, but probably not intuitively – that is where this worksheet steps in.

Additionally, one of the bigger ideas that we hope you take away from this course is that because computation is “cheap,” you can use your computer to quickly assess how different things depend on various parameters in your problem. In the case of coin flips, it might be obvious to you how the mean and variance are going to change as you increase the number of coin flips or the probability of heads,  but if it’s not, you can use your computer to help get an answer!

So without any further ado, let’s get flipping!

---

1. Using your code from worksheet 1.1, simulate $N$ tosses of $M$ coins with $P(\text{Heads}) = p$, where $M = 50$ and $p = 0.5$.  Keep track of the number of heads, $k$, of each experiment.  You may choose the number of experiments, $N$, but make it large enough to be useful without overwhelming your computer.  
    
    ```{note}
    This problem should teach you the value in avoiding **hardcoding**.  If in the previous worksheet you had written `rands = np.random.rand(100)` instead of `rands = np.random.rand(M)` or `Heads = rands < 0.6` instead of `Heads = rands < p`, where `M` and `p` were  *variables* in your notebook, then it will be a pain to copy over your code!  Try and practice this going forward!
    
    &nbsp;
    
    Also, welcome to the world of vague parameters.  In the real world, no one tells you how many simulations to do.  We will model this throughout the course and ask you to justify what reasonable parameters might be.
    ```

2. Before writing any more code, write down a prediction for what the distribution of $k$ will look like.  Be as precise or prophetic as you want, but at the minimum you should write down an expectation for what the "shape" of the distribution will be and where the peak will lie.
    
3. Use your work from the previous worksheet to plot the distribution and calculate the mean and variance of $k$.  Examine how it did or did not conform to your expectation.
    
    ```{note}
    These two problems might seem pedantic, but they are central to being a good scientist.  Even in small situations where you are plotting distributions of simulated coin tosses, having quantitative expectations is important!  This is not just philosophical, but also practical: if you don't know what you're expecting, you will never know if something has gone wrong (or right)!
    ```

4. As in the previous assignment, consider your PDF as a list of outcomes (what are the potential outcomes in this simulation?) and the probability of observing each of those outcomes.  Using the formulas for the first and second moments
	```{math}
	    E[X] = \sum_{x \in X}xP(x)
	```
	and
	```{math}
	    Var[X] = \sum_{x \in X}(x-\mu)^2P(x),
	```
	calculate the mean and variance of your PDF. That is, **do not** use any Python package to make these calculations.  (You should however check your work with the built-in functions!)
    
	```{note}
	It's surprising how many coding errors we all make. These formulas aren't complex, and yet, when I code them up I make errors. So this task has two goals. First, getting you to practice coding, practice debugging, etc. Second, to give you an opportunity to reflect on the formulas themselves. Why do these formulas correspond to the features on a distribution that you intuitively expect?  Why is the first a formula for an *average* and the second a formula for *spread*?
	```

5. Now that we have code that generates arbitrary simulated coin tosses and we know how to calculate $E[X]$, we want to explore how the expected value depends on $M$ and $p$. 
    1. Before you do any coding, brainstorm figures that would be useful for showing how $E[X]$ depends on these parameters.
    2. Write down predictions for how $E[X]$ will depend on these parameters.  If you want to write down a mathematical relationship, also write down what this means in words.  If you are inclined to write down words, try to also use symbols to describe what you mean.
    3. Write a loop that can calculate $E[X]$ for \textbf{two} values of $M$, storing those values.  $p$ and $N$ should remain fixed to values of your choice.  You may have to run new simulations corresponding to these different parameters.
    4. Write a loop that can calculate $E[X]$ for an arbitrary number of values of $M$, storing those values. $p$ should remain fixed.
    5. Plot $E[X]$ as a function of $M$ for an *interesting* set of $M$.
    6. Replicate this process for an arbitrary (yet *interesting*) list of values of $p$.

	```{note}
	In this problem we have again asked you to slow down and be conscious about why this is an interesting thing to do.  In this case, the theory about this relationship is straightforward, but for *your data* or *your problem*, it probably won't be.  Practicing stating your expectations ahead of time and planning your figures ahead of time and testing your code iteratively is only going to help when the problems are actually hard.  Actually, I cannot emphasize how useful it is to make sure that any loops you run work **once**, then **twice**, then a few times, and **then over all iterations**.  You will waste a lot of time if you write a loop and then set it to run a million times right away.
	```
    
6. In a similar manner to the previous problem, generate a figure showing the relationship between $Var[X]$ and $M$ and $p$.  Make sure to practice setting expectations, planning your figure, and planning your code.  Based on what you see, you may want to try different ranges and resolutions of values of $M$ and $p$.
    
7. Using the code in the notes, generate and plot the empirical CDF for your simulations using $M=50$ and $p=0.5$.  What list/array in the example code makes up the $x$-coordinates and which is the $y$-coordinate?  (You should examine the type, size, and values of each of the objects created in the sample code, if you have questions, please ask!)
    1.  Using `plt.hlines`, show where the $10^{\text{th}}$, $25^{\text{th}}$, $50^{\text{th}}$, $75^{\text{th}}$, and  $90^{\text{th}}$ levels are on your figure.
    2. Eyeballing your figure, write down these percentiles.
    3. Using the CDF array, find the indices that correspond to `CDF > 0.1`.
    4. Using these indices (I hope you saved them!), find the values of $k$ that correspond to `CDF > 0.1`.  What is the smallest of these values?  How does it compare to `np.percentile(data, 10)`?
    5. Use this process to find each of the requested percentiles.  Show these percentiles as vertical lines on your figure and annotate them.
    
    ```{note}
    This is probably the first empirical CDF that you have plotted, and you should feel ok that it seemed more complicated than plotting PDFs and frequency distributions -- it definitely requires a bit more coding.  However, we cannot understate how useful eCDFs are.  As we have discussed in the notes and you might have noticed in your own plotting, PDFs can be extremely sensitive to **binning**, and as a result, using empirical PDFs to perform calculations is incredibly suspect.  Instead, we will continue to emphasize that eCDFs are **exact**; they don't average over any of your data and use all of your observations to make calculations.  In this way, you should not look at CDFs as simply a different way of looking at your data, but as a *more reliable method for making calculations*.
    ```
    
8. Make a figure showing how the median and inter-quartile range (IQR) depend on $M$ and $p$.
    
9. Make a figure showing how the ratio of the standard deviation (square root of the variance) to the mean depends on $M$.
    
10. Make a figure showing how the ratio of the distance from the $90^{\text{th}}$ percentile to the median divided by the median:
    ```{math}
        \frac{Q_{90}(k) - Q_{50}(k)}{Q_{50}(k)}
    ```
	depends on $M$.
    
11. Describe the shape of the previous two curves.  How does it seem that the number of coin tosses impacts the relationship of the standard deviation to the mean?  How does this relate to the CLT?  In particular, do you see that $\sigma/\mu\sim M^{-1/2}$?
    
    ```{note}
    The $\sim$ symbol is a shorthand notation for conveying the dependence of some quantity on a particular parameter. It certainly isn't the case that $\sigma/\mu = N^{-1/2}$, since there are prefactors and constants that sit in front of the $N^{-1/2}$ that you would need to identify to produce an equality. A very useful way to demonstrate such dependencies quantitatively is to make a log-log plot. What that means is that you take the logarithm of the x and y axes, this can be done in Python using the `plt.xscale` and `plt.yscale` functions. Why should we do this? Consider a scenario where you have data $X$ and $Y$ such that $X=\alpha Y^{\beta}$. How could we figure out what the values of $\alpha$ and $\beta$ are for your data? If you take the logarithm of the left and right hand sides of the relation above you would get $logX=log\alpha + \beta logY$, where we have used the properties of the logarithm to rearrange the equation. But what you see now is that if we were to plot logX vs logY the resulting plot should be a \textbf{line} with a y-intercept and a slope. The y-intercept will give you the value of $log\alpha$ and the slope will be $\beta$. Hence if you really wish to show that $\sigma/\mu \sim N^{-1/2}$ then you should really plot log of $\sigma/\mu$ vs log$N$ and assess the slope of the resulting straight line.  
    ```
    
12. **BONUS**: In the vein of the previous note, change the axes of your plot to be log-scaled and add a line that has $\beta = -1/2$ for comparison to your data.  What do you observe?
    
13. **BONUS**: Divide the previous two curves by $\sqrt{M}$.  What do you observe?