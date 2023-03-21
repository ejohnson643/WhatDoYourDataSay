# Assignment 2: Hints and Guidance

<p style="text-align: center;">Written by Eric Johnson and Madhav Mani</p>

<p style="text-align: center;">Most recently compiled on March 14, 2023</p>

---

**Assignment Due Dates** (Fall 2020):

 - The *Assignment Attempt* is due on October 16, 2020 at 11:59PM CST
 - The *Complete Assignment* is due on October 23, 2020 at 11:59PM CST

---

Please follow the guidelines on Canvas for assignment [completion](../HowTo_AssignmentCompletion).  In particular, the completed assignment must be a PDF containing completed solutions as well as the code used to generate figures and results in the PDF.  Please use complete sentences in your solutions.  All figures should have labeled axes, legends, and appropriate annotations.  Your work should be completely your own, but if you do use other code or resources, please reference those sources.
    
This document contains hints and help for each non-bonus problem in the assignment.  If you complete all non-bonus problems at a satisfactory level ("Good" or "Novice" on the course rubric), then you will receive a "B" grade for the assignment.  If you complete the bonus questions at a satisfactory level, you can improve your grade on the assignment.

1. **Poisson Rate Estimation:**
	1. **Maximum Likelihood Estimation and Confidence Intervals:**
		1.  This problem is meant to be absurd.  The numbers will be odd.  This is meant to illustrate the pitfalls of modeling everything like a normal distribution and hoping it works out.
                
        2. If you're getting a little lost here, try and persevere into part (1.2).  That is, do your best to implement these formulas, and try and analyze them from the perspective of "if a collaborator presented me with these numbers, would I believe them?"  Then hopefully part (1.2) will help illustrate why this sub-problem and the next are interesting.  [Worksheet 2.1](./Worksheet_2_1_Bootstrapping) should also be useful.
                
        3. See the previous hint, but also take a look at [Worksheet 2.1](./Worksheet_2_1_Bootstrapping)!
                
        4. This process should be similar to what you did in Problem 2 and 3.6 in the previous assignment in terms of looping over different sample sizes.  Based on what we think about the time-dependence of the samples, should you use `np.random.choice` to select these intervals?
                
            Also, how can you illustrate how an *interval* changes with $N$?  Do you have an expectation for how the size of the interval will change based on the CLT?  How can you evaluate "how good" an approximation is?  We want you to be quantitative, so how can you quantify the "closeness" of the exact and approximate intervals?
            
    2. **Visualizing Confidence Intervals:**

    	1. If you're having trouble implementing any formula, make sure to ask for help!  Once you've got a "stable" implementation of these formulas, it might be useful to wrap it up in a function for further use...
                
        2. You get to pick the value of $N_{exp}$.  Your output is a distribution of $\hat{\lambda}$.  Will it be more useful to have a *ton* of experiments or just a few?  Make sure not to hardcode this number, and if you're confused, try a couple of values - making sure to plot each time!  Use the value of $N_{exp}$ that helps you solve the problem easiest! 
                
            Also, feel free to discuss these sub-problems in the context of one figure that combines them all.  That is, you don't need to keep remaking the figure with just one thing added, you can do it all at once in one figure and then discuss what each sub-problem adds separately.
                
        3. See problem 2.4. from the previous assignment for how you can overlay normal distributions.  The syntax for using `scipy.stats` is to do `scipy.stats.norm.pdf(xGrid, loc=mu, scale=sigma)` to make the PDF.
                
        4. Here `tauBar` is the average value of the $\tau$ intervals.  $N$ is the number of intervals in the estimate.  In this particular problem, how many intervals are you using to calculate the estimates in the $N_{exp}$ experiments?  (It's not 1372!)
                
        5. If you're stumped, please ask questions!  This is an excellent topic for office hours.
                
        6. The CDFs of the theoretical curves can be gotten by changing the `pdf` to `cdf` in the above code snippets.

    3. **Maximum A Posteriori Estimation and Credible Intervals:**
    	1. The syntax for a gamma distribution in `scipy.stats` is `st.gamma(shape, scale=1/rate)`.  You will then be plotting a curve that is a *function* of $\lambda$, so how can you choose a suitable set of values over which to look at this function?  The previous assignment's solutions may help!
                
            Also, in terms of discussion, consider what the shapes of these curves mean in terms of where you expect $\lambda$ to lie *before* you have data.
                
        2. Again, the syntax for a gamma distribution is `st.gamma(shape, scale=1/rate)`.  What are the rate and shape parameters for the posterior distribution?  Once you have a gamma distribution object or an array of posterior probabilities, there are many ways that you can find the mean (look at the next question!), median, and mode.
                
        3. You can work on this theoretically or computationally, depending on what makes more sense to you.  This output can be as simple as two curves, although you may want to wrap it up with the next sub-problem for clarity and brevity.
                
        4. If you're having trouble with the `scipy.stats` functions, please ask.
    
    4.  You are using your bootstrapped samples to generate estimates of the *variance* of $\tau$.  Generating a bootstrapped sample can be done using `np.random.choice(data, replace=True, size=N_data)`.    
    

2. **Linear Regression 1**
        
    1. **Generate Data:**
            
        1. You are free to choose the $x_i$ however you want.  You can use `np.random.randn` to generate standard normal random variables (which you'll have to adjust to have the appropriate variance!) or an appropriate `scipy.stats.norm`.  Similarly, how can you get uniform numbers going from $-\sqrt{N}$ to $\sqrt{N}$?
                
        2. Consider [Worksheet 2.2](./Worksheet_2_2_OLS_LinReg) for examples of linear regression functions in Python.  We'll discuss this in office hours as well. 
                
        3. Once you have a model that has been fit, how can you use it to generate new predictions?  Will it be easier to tell if the residuals are normally distributed using a PDF or CDF?  The last question is purely qualitative.
            
    2. **Estimating Regression Coefficients**
    	1.  You already have some experience with this from the first problem, but now there is a wrinkle in that you need to calculate $s_y$ and $s_x$ before you can multiply by the $t$-distribution percentiles.  Don't try to make this calculation any more streamlined than you can understand.  If you need to make an array `yHat` for your predictions, then do `yDiff = y-yHat`, then do `yDiffSq = yDiff**2`, and so on, *that's perfectly ok*.  Getting the formula implemented is more important than super concise code.  It's worth noting that the rubric doesn't prioritize "concise" or "efficient" code, but "understandable and interpretable" code.
                
            Make sure that the intervals you recover make sense.  Do they contain your regression coefficients?  Are they too large or small?  Check your code by running it a few times with different random data.
                
        2. This problem is tricky.  As a refresher, Equation 29 in the notes says that the posterior distribution for a regression coefficient is normally distributed with variance given by
            ```{math}
	        \Delta_{\beta} = \sigma^2\left(
	            X^TX + \delta_{\beta}^{-2}\mathbb{I}_{2}
	        \right)^{-1}
            ```
            and mean given by
            ```{math}
            \vec{\eta}_{\beta} = \frac{\Delta_{\beta}}{\sigma^2}\left(
                X^T\vec{y} + \frac{\vec{\mu}_{\beta}}{\delta_{\beta}^2}
            \right).
            ```
            These are somewhat complicated formulas, and even if you are used to working with vectors and matrices on paper, it may be difficult to see how to implement these formulas in Python.  You can use `np.dot(A, B)` as an inner product to execute the matrix multiplication $AB$ and `np.linalg.inv` to find the inverse of matrices.
            
            So then if you're given $\vec{\mu}_{\beta} = \hat{\beta}_{OLS}$ from the previous problem, $\delta_{\beta} = 10$, and $\sigma$ estimated from the residuals, you can implement the formulas by chaining together Python commands.  For example, $X^TX$ can be found with `np.dot(X.T, X)`, and you can make a $2\times2$ identity matrix $\mathbb{I}_{2}$ in several ways.
            
            If you follow the formulas and use vector operations in Python the mean and variance of the posterior will be a vector and matrix, respectively.  Thus you have to take the components corresponding to $\beta_1$, which will be the second element in $\vec{\eta}_{\beta}$ and the second row, second column element of $\Delta_{\beta}$, and put them into `st.norm` as we have been practicing.
                
        3. Refer to [Worksheet 2.3](./Worksheet_2_3_Boot_LinReg) if you want more guidance here.  We'll hopefully discuss this on Monday, and if not, Worksheet guides will be posted in time to be useful.  Of particular importance, bootstrapping multivariate data sets means that we resample entire *samples* as units.  We want to generate a new dataset from the observations we already have, not create new observations by mixing and matching covariate $x_{12}$ with response $y_{4}$.  How then can you use `np.random.choice` (or another appropriate function) to keep rows together while bootstrapping?  Please post about this on the discussion thread with your concerns or ideas. 
                
        4. Either a PDF or a CDF will be suitable here.  You only need to compare the intervals, but part (2) did contain an entire distribution if you want to show it.
    
    4. How does the Central Limit Theorem come into play here?
    

3. **BONUS: Linear Regression II**
    
	There are no written hints for bonus problems.  If you'd like some help, please ask or come to office hours!


4. **Fitting a Thermodynamic Model**

	1. The [linked notebook](https://github.com/ejohnson643/WhatDoYourDataSay/blob/main/CourseFiles/Module_2/Resources/Assignment2_Fall2020_Q4_Help.ipynb) should be sufficient to guide you through the process for getting the desired information from the images.  You may want to print and examine the structure of `newStrainInfo` at the end of that notebook (you may need to remind yourself how dictionaries work!).  Recall that you can access the *keys* of a dictionary using `myDict.keys()` - hopefully the keys are intuitive enough that you can determine what you're looking at.  In particular, the YFP is extracted on a *per-cell* basis, so based on the output of the functions in the notebook, can you determine how many values you expect for each cell's YFP array?

    	This last question has to do with the biology.  Refer to the notes what it means for a molecule to be called a *repressor* and ask yourself whether you expect more or less fluorescence if there is more or less repressor in the cells.  Recall also that the `Delta` strain has *no repressors*.

    2. Can you find values for fold change in the `newStrainInfo` dictionary?  How might you be able to extract them from the dictionary into a more useful format, like a list or array?  The rest of this problem should be similar to the work you did in problem 2.