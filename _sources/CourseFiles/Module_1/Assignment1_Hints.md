# Assignment 1: Hints and guidance

<p style="text-align: center;">Written by Eric Johnson and Madhav Mani</p>

<p style="text-align: center;">Most recently compiled on March 14, 2023</p>

---

**Assignment Due Dates** (Fall 2020):

 - The *Assignment Attempt* is due on October 2, 2020 at 11:59 PM CST
 - The *Complete Assignment* is due on October 9, 2020 at 11:59 PM CST

---

Please follow the guidelines on Canvas for assignment [completion](../HowTo_AssignmentCompletion).  In particular, the completed assignment must be a PDF containing completed solutions as well as the code used to generate figures and results in the PDF.  Please use complete sentences in your solutions.  All figures should have labeled axes, legends, and appropriate annotations.  Your work should be completely your own, but if you do use other code or resources, please reference those sources.
    
This document contains hints and help for each non-bonus problem in the assignment.  If you complete all non-bonus problems at a satisfactory level ("Good" or "Novice" on the course rubric), then you will receive a "B" grade for the assignment.  If you complete the bonus questions at a satisfactory level, you can improve your grade on the assignment.

1.
	1.  There are several ways to make subplots in Python but one of the easiest is the `plt.subplots` function.  An example is shown here:
        ```{code-block}
        N_rows, N_cols = 3, 5  ## Don't hardcode!
        
        ## Here we use plt.subplots to make figure and axes objects.
        ## Note the 'figsize' keyword can be useful to change overall figure shape.
        fig, axes = plt.subplots(N_rows, N_cols, figsize=(12, 7))
        
        ## We can then access each subplot in the array of axes:
        ax1_1 = axes[0, 0]  ## We can assign each subplot its own variable
        _ = ax1_1.scatter(np.random.rand(100), np.random.rand(100), color='red')
        _ = ax1_1.scatter(np.random.rand(100), np.random.rand(100), color='blue')
        
        ## We can slice the array and use plotting methods directly.
        _ = axes[2, 4].bar(np.arange(10), np.random.randn(10), color='indigo')
        
        ## We can tell distplot to plot on a specific subplot.
        _ = sns.distplot(np.random.randint(12, size=100), ax=axes[1, 3],
                         color='khaki')
        ```

   		To get theoretical PDFs and CDFs, the `scipy.stats` module is recommended ([tutorial](https://docs.scipy.org/doc/scipy/reference/tutorial/stats.html).  An example using the [exponential distribution](https://en.wikipedia.org/wiki/Exponential_distribution) is shown here:
            
        ```{code-block}
        ## import the package
        import scipy.stats as st
        
        ## Define a few parameter sets for the exponential function
        lambda_1 = 1.0
        lambda_2 = 3.0
        
        ## We can create scipy.stats.expon *objects*
        ## We specify the exponential lambda parameter with the 'scale' keyword.
        expon_1 = st.expon(scale=lambda_1)
        expon_2 = st.expon(scale=lambda_2)
        
        ## Set up a grid on which we'll plot these PDFs and CDFs
        x_grid = np.linspace(0, 10, 101)
        
        ## We use the pdf/cdf *methods* with 'x_grid' as input.
        PDF_1 = expon_1.pdf(x_grid)
        CDF_1 = expon_1.cdf(x_grid)
        
        PDF_2 = expon_2.pdf(x_grid)
        CDF_2 = expon_2.cdf(x_grid)
        
        ## Make some axes and plot!
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        
        _ = ax.plot(x_grid, PDF_1, '-b', label='PDF 1')
        _ = ax.plot(x_grid, CDF_1, '--b', label='CDF 1')
        
        _ = ax.plot(x_grid, PDF_2, '-r', label='PDF 2')
        _ = ax.plot(x_grid, CDF_2, '--r',
                    label=r'$\lambda = $'+ f"{lambda_2}")  ## Feel free to ask about this label!
        
        _ = ax.legend()
        ```
            
    2. Again using the `scipy.stats` module, we can generate random samples from specified distributions.  As an example, we will continue to use the `expon` objects from the previous hint.
            
        ```{code-block}
        ## We'll need this to make eCDFs
        from collections import Counter
    
        N_samps = 1000  ## Don't hardcode!  Make a variable!
    
        ## The 'rvs' method generates rANDOM vARIABLEs
        exp_rvs_1 = expon_1.rvs(size=N_samps)
        exp_rvs_2 = expon_2.rvs(size=N_samps)
        
        ## We can make eCDFs as in the notes:
        counts_1 = Counter(exp_rvs_1)
        vals_1 = np.msort(list(counts_1.keys()))
        eCDF_1 = np.cumsum(np.array([counts_1[ii] for ii in vals_1]))
        eCDF_1 = eCDF_1 / eCDF_1[-1]
        
        ## You might want to wrap this up into a function!
        counts_2 = Counter(exp_rvs_1)
        vals_2 = np.msort(list(counts_2.keys()))
        eCDF_2 = np.cumsum(np.array([counts_2[ii] for ii in vals_2]))
        eCDF_2 = eCDF_2 / eCDF_2[-1]
        
        ## Make a figure and axes and plot!
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        
        ## What other plotting routines could we use here?
        _ = ax.hist(exp_rvs_1, color='b', bins=bins, alpha=0.5, normed=True, 
                    label='PDF 1')
        ## I just learned about the 'step' plotting function!  Why is this appropriate here?
        _ = ax.step(vals_1, eCDF_1, '-b', label='eCDF 1')
        
        ## Notice the use of the 'alpha' and 'normed' keywords.  What happens if we don't use them?
        _ = ax.hist(exp_rvs_2, color='r', bins=bins, alpha=0.5, normed=True, 
                    label='PDF 2')
        _ = ax.step(vals_2, eCDF_2, '-r',
                    label=r'$\lambda = $'+ f"{lambda_2}")
        
        _ = ax.legend()
        ```
            
    3. To calculate the mean, standard deviation, and median of data (your random variables), you can use numpy's `mean`, `std`, and `median` functions.  The IQR is the difference between the $75^{th}$ and $25^{th}$ percentiles, so you can use `np.percentile` to calculate it. 
            
        For many `scipy.stats` objects, there are `mean` and `std` methods built in.  For statistics based on percentiles of distributions (median and IQR), you can use the `ppf` method (the Percentile Point Function) which is the *inverse* of the CDF - you provide a percentile (y-coordinate) and it tells you the corresponding value on the CDF (x-coordinate).  Example usage is shown below
            
        ```{code-block}
        ## Here we get the mean and std of the exponential from earlier
        mu_1 = expon_1.mean()
        sig_1 = expon_1.std()
        print(f"The mean of the first exponential is {mu_1:.2g}")
        print(f"The std dev of the first exponential is {sig_1:.2g}")
        
        ## We can get the 25th, 50th, and 75th percentiles using 'ppf'
        fifty_1 = expon_1.ppf(0.5)
        twentyfive_1 = expon_1.ppf(0.25)
        seventyfive_1 = expon_1.ppf(0.75)
        print(f"The 25th, 50th, and 75th percentiles are {twentyfive_1:.2g}, "+
              f"{fifty_1:.2g}, and {seventyfive_1:.2g}")
        ```

2. 
	1. Re-purpose your coin flipping code from the first worksheet so that instead of returning the number of heads, you return the number of heads minus the number of tails after each flip.  Note that now we want a *trajectory*, so if we're taking 100 steps, that corresponds to flipping 100 coins and we need to know where the walker is after *each flip*.  (How big of a list/array is this?)
            
    2. What's stopping you from putting your coin-flipping-now-random-walker code in a loop like we did in the first worksheet?  If you want to think about *vectorizing* your code, you can, but it's not necessary.  What is the shape of the data you'll be plotting; that is, what size array/list do you need to allocate?  Also, for plotting, what will be on the $x$-axis and what will be on the $y$-axis?  Make sure to label this correctly!\\
            
    3. If you saved your trajectories in an array, how can you use the `axis` keyword in `np.mean` and `np.std` to calculate the mean and standard deviation after each step?  For plotting, make your calculations *obvious*!  We should not have to squint to see your results.  For describing the curves, you can use comparisons, e.g. "the mean looks like a quadratic funciton," or you can use qualitative language, e.g. "the standard deviation decreases with step-size, slowly at first, then more quickly."
            
    4. If you have saved your trajectories in an array, how can you get the $144^{th}$ step?  For your description of the CLT, why is the mean of i.i.d. random variables always normally distributed?  What about a *mean* is similar to the random walkers' positions?
            
    5. See the comments from 2.3 on how to describe curves.  You should also give a description of what *changes* between the plots.  Your figure should be very easy to understand.


3. 
	1. The `np.loadtxt` function will be useful for reading the data in.  You will need to know *where* the data set is in your computer, so you may want to start organizing a folder for this assignment into which you can place the data.  For plotting, recall that you don't need to supply an $x$-coordinate for `plt.plot`, it will assume you want to use `x=np.arange(len(data))`.  Try removing the line between points and using a small marker for the data.  Try changing the range of the $x$-axis (`ax.set_xlim`) to "zoom in."  Try chucking the data into some other plotting functions that you've been using.
            
    2. See the hint from 1.3.
            
    3. Is there a value in between the two behaviors that you can use as a threshold, so that being greater than the threshold is spinning one direction and smaller is spinning in the other?  Consider the following pseudocode: start a counter at 0 and determine whether the first data point is greater than your threshold.  Loop through the data, if the next datum is still greater than the threshold, increment the counter, else, store the counter's value in a list and then restart the counter at zero.  Continue checking the data, and as long as it remains below the threshold, increment the counter, else, store the counter and then restart.  Continue until all the data are examined.
            
    4. See problem 1.2 and 1.3 for hints.