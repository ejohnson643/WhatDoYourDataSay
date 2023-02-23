# Worksheet 1.1: Coin Tossing

Welcome to your first worksheet of this course! The worksheets are designed to work in parallel to the course notes and lectures. They are not intended to be summary assessments at the end of a module, but as **necessary content** that you consume on your way to the assignments (and eventually attacking your own data!). As a result, if there are parts of the worksheet that you don’t understand or are having trouble with, make sure to **ask for help!**

In this worksheet, you will practice implementing the concepts enumerated in the first few sections of the Module One course notes, up to the section on cumulative density functions. You should complete this worksheet in your own Jupyter notebook and try to complete every question. Even if you think that you are proficient in Python coding, you should make sure that you can write working answers to all the parts of this problem - it will be useful code for later parts of the course, so your effort won’t be wasted!

```{admonition} **Worksheet Goals:** 
By the end of this worksheet, you should be able to: 
 - <mark style="background-color: #e26563">PCS-3</mark> Generate random numbers
 - <mark style="background-color: #e26563">PCS-2</mark> <mark style="background-color: #6b9cee">MVD</mark> Generate and plot distributions
 - <mark style="background-color: #e26563">PCS-3</mark> Perform simulations of simple experiments
 - <mark style="background-color: #e26563">PCS-2</mark> <mark style="background-color: #6b9cee">MVD</mark> Calculate the mean and variance of an experiment and display them on a figure
```

At this point in the notes, we’ve started talking about flipping coins, binomial distributions, PDFs, and moments of distributions, but what does this actually look like in practice? To start to see how your computer can be used to solve generic statistical problems, we first have to make sure that our computer can be useful in the most basic of situations. The coin flip is that basic problem that has the benefit of being useful both as a basic theoretical concept, and as a basic computational one.

One of the bigger ideas that we hope you take away from this course is that if you can program, you can use **simulation** as an integral part of experimentation and analysis. This is most obvious in the case of flipping coins. You could get a couple coins, flip them many times, record the results, and make the same figures as below, but it is much, much faster just to let the computer do the flipping for you!

So without any further ado, let’s get flipping!

---

**Please complete the following tasks:**

1. As introduced in the tutorial, the basic random number generator in Python is the `np.random.rand` function.
	1. `help()` function to read about the various arguments and keyword arguments that `np.random.rand` takes.  Describe how to use this function in your own words.
        
    2. Generate many random numbers (how many is "many"?) and print their maximum and minimum to the screen.  How many of your numbers are bigger than 0.5?  How many are bigger than 0.6?  0.95?
        
    3. Based on your generated numbers, what is the likelihood of getting a number bigger than 0.75?
        
    4. To what number would I want to compare my random numbers so that 60% of the time I get a `True` and the rest of the time I get a `False`?
        
    5. Explain in words how you can use `np.random.rand` to simulate a coin where $P(\text{Heads}) = 0.6$.

    ```{note}
    One trick that helps when you are trying to code new things, is to attempt to map out your problem **in words**.  This is known as writing **pseudo-code**, and is similar to outlining an essay.  In this problem, it might seem trivial, but as your scripts become longer and more complex, having a plan can be essential for staying on target.
    ```

2. As pointed out in the course notes, flipping one coin isn't that exciting, but flipping many coins can be very interesting!
    1. Again using $p = P(\text{Heads}) = 0.6$, flip $N=10$ coins.
    2. Count the number of heads.
    3. Write a **function** that returns the number of heads in $N$ flips of a coin with $p = P(\text{Heads})$.  That is, $N$ and $p$ are **inputs** (arguments) to your function and $k=$ the number of heads is the **output**.
    4. **Bonus:** Describe your function and write a [docstring](https://www.python.org/dev/peps/pep-0257/).  Consider the following questions: How did you choose to count the number of heads in your coin tosses?  What are other ways you could have done this?  Do you flip all your coins at once or one at a time?  Does your function check the inputs to make sure they're the correct type or value?

3. Let $Y$ be the random variable corresponding to the number of heads generated in an "experiment" of flipping $N=10$ coins, where each coin has $P(\text{Heads}) = 0.6$.  Repeat this experiment $M=100$ times, saving the outcome each time.
    
	```{note}
    If you are feeling lost, try writing down what you want to happen in words.  Pseudo-coding is especially useful when you are trying to work out *what information* you will need to be saving, and then *how* you want to save it.  For example, if you want to save the outputs in a list, then you will use slightly different functions than someone who wants to save the output in a `np.ndarray`.
    ```
    
4. Let's explore the data (!?) you just generated.
    1. What are the maximal and minimal numbers of heads that you observed in your experiments?
    2. How many unique values are there in your dataset (Hint: `np.unique`)?
    3. How many times do you observe 0 heads, 1 heads, 2 heads, etc.?  What would be an appropriate diagram with which to display this information?
    4. Create a figure that shows your **frequency distribution**.  Label the axes and if necessary, provide a legend or annotate your figure.  Make sure the $y$-axis is scaled and labeled so that it is easy to read off how many times you observed each number of heads.
    5. Describe your distribution qualitatively.  Estimate the mean, median, and mode by eye. 

	```{note}
    As we will continue to emphasize, making effective visualizations of your data is incredibly important.  In particular, you should always have a goal in mind of what you want to convey, and your figure should make that as clear as possible.  Just as with writing code, it can be very useful to write down *ahead of time* what you want your figures to show.
    ```

    
5. It's often useful to annotate your figures with relevant calculations or quantities.  Let's do that here.
    1. Using either `np.mean` or the formula for expected value, calculate the mean, $\mu$ of your observations.  Display this value on your figure using the `plt.vlines` function and using a `legend` or `plt.text`.
        
    2. Using either `np.median` or some other method, calculate the median of your observations and display the value on your figure.
        
    3. Calculate the mode of your observations and display the value on your figure.
        
    4. Calculate the standard deviation, $\sigma$ of your observations either by using the formula for variance or the `np.std` function.  Display $\mu\pm\sigma$ using `plt.vlines` and annotate the value of $\sigma$ in a legend or using `plt.text`.
        
    5. **Bonus:** Use the `axes.get_ylim` and `axes.set_ylim` functions to control the range of your `plt.vlines` function and to preserve the $y$-axis range of your figure.

	```{note}
    If the previous problem didn't present any figure-design challenges, this problem should.  Figure-making can be fussy work, but after some practice you'll develop a set of templates that make things faster.  Specific things you could think about are axes sizes, legend or text placement, and font sizes.  Also, annotating *widths* is somewhat difficult; you should explore different ideas if you have time.
	
    &nbsp;

	If this figure represents your first foray into illustrating a computational analysis, note how you can now *see* how the differences in your summary quantities relate to your data in slightly different ways.

    &nbsp;
	
	As a final note, you should notice that we suggested multiple ways for computing most of these quantities.  You should make sure that you can use any of these methods.  It might seem trivial for calculating a mean, but being able to replicate your results in different ways is an important part of being a computational worker.  When an result or method isn't clear, one of the first things we turn to is to see if we can get a similar conclusion via a different calculation.  You can start building that skill here.
    ```

6. As noted above, the quantity you should be working with is a *frequency distribution*, so how can we make it a **probability distribution**?
    1. Make a list of the different outcomes and the number of times each outcome is observed.  You should try to do this "by hand", but you can also look at `np.histogram` or `plt.hist`.  Make sure that both sets of outputs are *integers*.
        
    2. What is the sum of the counts list?  (This is a good sanity check that you are doing things right.)
        
    3. Based on your data, what is the probability of getting 0 heads?  1 heads?  2 heads?  That is, if I randomly grabbed one of your outcomes out of a bag, how often would I get $k=4$ heads?
        
    4. Generate a list of the *probabilities* of each of your outcomes.  Use this list to show the **PDF** of your data.  Using your PDF, replicate your earlier annotated figure.  Confirm that the sum of your probabilities is 1.
        
    5. Compare your figure to one generated by `sns.histplot` or `plt.hist`.  Are the bins, bar heights, and other features the same?  Describe any discrepancies or similarities.  In `sns.histplot`, what is the overlaid line?

