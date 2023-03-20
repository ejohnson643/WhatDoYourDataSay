# Assignment 2

<p style="text-align: center;">Written by Eric Johnson and Madhav Mani</p>

<p style="text-align: center;">Most recently compiled on March 14, 2023</p>

---

**Assignment Due Dates** (Fall 2020):

 - The *Assignment Attempt* is due on October 16, 2020 at 11:59PM CST
 - The *Complete Assignment* is due on October 23, 2020 at 11:59PM CST

---

Please follow the guidelines for assignment [attempts](../HowTo_AssignmentAttempt) and [completion](../HowTo_AssignmentCompletion). In particular, the attempt must be a Jupyter notebook or PDF that clearly enumerates the problem attempts, and the completed assignment must be code *and* a PDF containing completed solutions. Please use complete sentences in your solutions. All figures should have labeled axes, legends, and appropriate annotations.

Please read the *entire assignment* before getting too deep into any one problem so that you can properly allocate your time and questions.

## Learning Objectives:

The module learning objectives assessed in this assignment are that a student will be able to:

- <mark style="background-color: #e26563">PCS-1</mark>: Calculate estimates and intervals (E\&I) using theoretical formulas.
- <mark style="background-color: #e26563">PCS-2</mark>: Calculate E\&I using bootstrapping.
- <mark style="background-color: #e26563">PCS-3</mark>: Perform OLS linear regression.
- <mark style="background-color: #e26563">PCS-4</mark>: Perform cross-validation to estimate model error.
- <mark style="background-color: #ffda5c">TS-1</mark>: Understand and discuss methods for making parameter E\&I.
- <mark style="background-color: #ffda5c">TS-2</mark>: Understand and discuss OLS assumptions in model fitting.
- <mark style="background-color: #6b9cee">MVD</mark>: Illustrate uncertainty in E&I from data.
- <mark style="background-color: #92c57a">NQP-1</mark>: Compare and critique different methods for calculating E&I and determine optimal methods for your data.
- <mark style="background-color: #92c57a">NQP-2</mark>: Assess whether a given model is under- or over-fit and describe how to improve the model or fitting method.

You can learn more about where to study and practice these skills in the [curriculum alignment table](./Module_2).

---

1. **Poisson Rate Estimation:** Consider the bacterial chemotaxis data from the first assignment [here](https://github.com/ejohnson643/WhatDoYourDataSay/blob/main/CourseFiles/Module_2/Resources/omega.txt).  Generate lists of the time intervals, $\tau^+$ and $\tau^-$, that the bacteria spent rotating in the positive and negative directions, respectively.  Feel free to use your work from Assignment 1 or the Assignment 1 Solutions (with proper attribution!).
        
    In Assignment 1, you examined the distributions of $\tau^+$ and $\tau^-$ and calculated some summary statistics.  In particular, the bonus problem 1.5 noted that both $\tau^+$ and $\tau^-$ are *exponentially distributed* with a rate parameter, $\lambda$.  In 1.6 you explored the posterior distribution for $\lambda$ under the assumption that $\tau$ are exponentially distributed, and started to think about whether $\lambda=0.2$ was consistent with the data.  After reading the course notes and watching the video lectures, you should now see that this is a *parameter estimation* problem, which we will now perform more carefully.
        
    Specifically, in this problem you will examine three different methods for estimating $\lambda$: maximum likelihood estimation, Bayesian estimation, and bootstrapping.  You will examine how these estimates depend on the number of samples in the data, and how the different estimates relate to each other.
        
    1. **Maximum Likelihood Estimation and Confidence Intervals:** As pointed out in the notes and worksheets, MLEs and Confidence Intervals involve analytically manipulating theoretical models.  In this problem, we will examine three different sets of assumptions, each a refinement on the last, to underscore what this theoretical process actually looks like in practice.
            
        In terms of the actual maximum likelihood point estimator, it can be shown that using $\lambda = \hat{\lambda}$ from the following equation
        ```{math}
        \hat{\lambda} = \frac{N}{\sum_{i=1}^N\tau_i} = \frac{1}{\bar{\tau}}
        ```
        maximizes the likelihood function for exponentially distributed data, $\tau_i$.  That is, if we have exponentially distributed intervals $\tau_i$, then setting $\lambda = \hat{\lambda}$ provides the model that has the highest likelihood of observing the data.
            
        Calculate $\hat{\lambda}$ for both sets of intervals: $\tau^+$ and $\tau^-$.
            
        To complete the estimation, we need to also give a confidence interval for $\hat{\lambda}$.  In the following sub-problems, you're going to explore how getting the theory correct can be tricky and can lead to more or less consistent results.  (Throughout this problem you can assume that we want a 95\% confidence interval, that is, $\alpha = 0.05$.)
            
        1. <mark style="background-color: #e26563">PCS-1</mark>, <mark style="background-color: #ffda5c">TS-1</mark>, <mark style="background-color: #92c57a">NQP-1</mark> If you didn't know anything about exponential distributions or confidence intervals, you may have seen that sometimes people give intervals as `mean` $\pm$ `stddev`.  Will this work for our estimate of the rate parameter, $\lambda$?  Try it out: calculate
	        ```{math}
	        I = \frac{1}{\bar{\tau}\pm \sqrt{\text{Var}[\tau]}}
	        ```
        	for both sets of intervals $\tau^+$ and $\tau^-$.  Describe these intervals.  Are they reasonable?  Are the respective MLEs, $\hat{\lambda}^+$ and $\hat{\lambda}^-$ in the intervals?
                
        2. <mark style="background-color: #e26563">PCS-1</mark>, <mark style="background-color: #ffda5c">TS-1</mark>, <mark style="background-color: #92c57a">NQP-1</mark> If you do some more reading, you might find that the formula for a confidence interval is often given as 
	        ```{math}
	        :label: eqn_NormConfInt
	        \texttt{mean} \pm\,\,z_{\alpha/2} \times\texttt{stderror},
	        ```
	        where the *standard error* is given as `stderror`$=$`stddev`$ / \sqrt{N}$, and $z_{\alpha/2}$ is the $p^{th}$ *percentile* of the standard normal distribution.  You can calculate percentiles of a normal distribution using `scipy.stats.norm.ppf(p)`, we have `mean` $=\hat{\lambda}$, so we just need `stddev` to complete the formula.  You may be tempted to use the standard deviation from the data as in the previous problem, but a quick Wikipedia search shows that the variance of exponentially distributed variables is $1/\lambda^2$, so that maybe we can use $\hat{\lambda} = $`stddev`.  Thus the interval now becomes
	        ```{math}
	        :label: eqn_ApproxConfInt
	        I = \left[
	            \hat{\lambda} + z_{\alpha/2}\frac{\hat{\lambda}}{\sqrt{N}},\qquad
	            \hat{\lambda} + z_{1-\alpha/2}\frac{\hat{\lambda}}{\sqrt{N}}
	        \right]
	        ```
	        Use this formula to calculate confidence intervals for $\lambda$ for both sets of intervals, $\tau^+$ and $\tau^-$.  Discuss these intervals.  Are they reasonable?  Is $\hat{\lambda}$ in the respective interval?  Importantly, does it make sense for the confidence interval to be symmetric?
                
        3. <mark style="background-color: #e26563">PCS-1</mark>, <mark style="background-color: #ffda5c">TS-1</mark>, <mark style="background-color: #92c57a">NQP-1</mark> It turns out that the confidence interval formula (Equation {eq}`eqn_NormConfInt`) you just implemented is actually only the formula for confidence intervals of *normally-distributed* random variables!  However, you can do a good amount of manipulation, as shown [here](https://people.missouristate.edu/songfengzheng/Teaching/MTH541/Lecture\%20notes/CI.pdf), [here](https://stats.stackexchange.com/questions/399809/95-confidence-interval-of-lambda-for-x-1-x-n-iid-exponential-with-rate?newreg=6579dec0b6a54b808a0b9cae6020c137), and of course, [here](https://en.wikipedia.org/wiki/Exponential_distribution#Confidence_intervals), that $\lambda$ is actually distributed according to a $\chi^2$ distribution (pronounced "chi-squared", where "chi" rhymes with "sky").   As a result, we can write the confidence interval *exactly* as 
	        ```{math}
	        :label: eqn_ExactConfInt
	        \left[
	            \frac{\chi^2_{2N}(\alpha / 2)}{2N\bar{\tau}},\quad
	            \frac{\chi^2_{2N}(1 - \alpha / 2)}{2N\bar{\tau}}
	        \right],
	        ```
            where $\chi^2_{k}(p)$ is the $p^{\text{th}}$ percentile of the $\chi^2$ distribution with $k$ degrees of freedom.  In Python, you can calculate this using `scipy.stats.chi2.ppf(p, df=k)`.
                
            Use this formula to calculate confidence intervals for $\lambda$ for both sets of intervals, $\tau^+$ and $\tau^-$.  Discuss these intervals.  Are they reasonable?  Is $\hat{\lambda}$ in the respective interval?  Importantly, how do they compare to the approximate intervals from the previous problem?  Does it seem that a normal approximation is "good enough"?  Are these intervals symmetric?
                
        4. <mark style="background-color: #e26563">PCS-1</mark>, <mark style="background-color: #6b9cee">MVD</mark>, <mark style="background-color: #92c57a">NQP-1</mark>  Rather than leaving the last question vague and qualitative, let's be precise and explore how good of an approximation Equation {eq}`eqn_ApproxConfInt` provides compared to the exact answer in Equation {eq}`eqn_ExactConfInt`.  To do this, randomly select $N$ intervals from either list, $\tau^+$ or $\tau^-$, calculate $\hat{\lambda}$ and the exact and approximate confidence intervals.  Using an appropriate set of values for $N$, generate figures that show both how the size of the confidence interval depends on $N$ and how the approximate and exact intervals differ from each other.  Can you determine how many samples are needed before the intervals are indistinguishable?
    
    2. **Visualizing Confidence Intervals:** To further unpack the differences in the confidence interval formulas, let's try and visualize the problem more directly.
            
        1. <mark style="background-color: #e26563">PCS-1</mark> Randomly draw $N=100$ intervals from one of the lists ($\tau^+$ or $\tau^-$) and estimate the rate parameter $\lambda$ (calculate a point estimate *and* and interval!).  Save this estimate as `lHat0` and `lConfInt0`.  Use the *exact* confidence interval formula in Equation {eq}`eqn_ExactConfInt`.
                
        2. <mark style="background-color: #e26563">PCS-1,2</mark>, <mark style="background-color: #ffda5c">TS-1</mark> Repeat the previous problem $N_{exp}$ times, where $N_{exp}$ is a large number of your choosing.  Show the distribution of estimates, $\hat{\lambda}$ and describe its shape.  Where is it centered?  On `\lHat0`?  On the value calculated in 1.a?
                
        3. <mark style="background-color: #e26563">PCS</mark>, <mark style="background-color: #ffda5c">TS-1</mark> Equation {eq}`eqn_ApproxConfInt` suggested that $\lambda$ is normally distributed.  Overlay a plot of a normal distribution with mean, $\mu = \hat{\lambda}$, and standard deviation $\sigma = \hat{\lambda}/\sqrt{N}$, where these can be calculated using all the intervals.  Qualitatively discuss how well the distribution of estimates match this curve.
                
        4.<mark style="background-color: #e26563">PCS-2</mark>, <mark style="background-color: #ffda5c">TS-1</mark> The exact confidence interval (Equation {eq}`eqn_ExactConfInt`) used the fact that $\lambda$ is $\chi^2$-distributed.  Use `scipy.stats.chi2.pdf(xCoords, df=2*N, scale=1/(2*N*tauBar))` to overlay a plot of this distribution. Qualitatively discuss how well the distribution of estimates match this curve.
                
        5. <mark style="background-color: #ffda5c">TS-1</mark> , <mark style="background-color: #6b9cee">MVD</mark>  Indicate where `lHat0` and `lConfInt0` fall on the distribution.  Determine the fraction of the $N_{exp}$ estimates that fall in `lConfInt0`.  Discuss this fraction - does it make sense to you?  (If needed, run your code several times to convince yourself!)
                
        6. <mark style="background-color: #ffda5c">TS-1</mark> , <mark style="background-color: #6b9cee">MVD</mark>  Recreate your figure using (empirical) CDFs of the estimates and the theoretical curves.  Show the $\alpha/2^{\text{th}}$ and $(1-\alpha/2)^{\text{th}}$ percentiles of the $N_{exp}$ estimates, the median of the estimates, `lHat0` and `lConfInt0`.  Comment on any new insights from this figure.
            
    3. **Maximum A Posteriori Estimation and Credible Intervals:** We'll now explore how you can make MAP estimates of the rate parameter, $\lambda$, using Bayes' Theorem.  While in the notes and worksheets we emphasize an empirical approach to this problem, here you'll explore how you can implement a more theoretical approach as well.
            
        In particular, it can be shown that if we use a [gamma distribution](https://en.wikipedia.org/wiki/Gamma_distribution) to describe our prior expectation for $\lambda$, that we can then *exactly* write the posterior distribution as a mathematical formula.  In general, prior distributions that allow for this analytic manipulation are called **conjugate priors**.
            
        A gamma distribution has the form
        ```{math}
        \mathcal{G}\left(
            x\,|\,a, b
        \right) = \frac{b^ax^{a-1}}{\Gamma(a)}e^{-bx},
        ```
        where $\Gamma(a) = (a-1)! = (a-1)\cdot(a-2)\cdot(a-3)\cdots2\cdot1$.  $a$ and $b$ are known as *shape* and *rate* parameters, respectively.
            
        For exponentially distributed random variables, $\tau_i$, the likelihood of those data given a specific rate can be written
        ```{math}
        P(\tau_i|\lambda) = \prod_{i=1}^N \lambda e^{-\lambda \tau_i} = \lambda^Ne^{\sum_{i=1}^N\tau_i}.
        ```
            
        Putting the gamma prior and likelihood function into Bayes' Theorem, we can rearrange and normalize to find that the posterior function for $\lambda$ given the data is also a gamma function!
        ```{math}
        :label: eqn_LambdaPost
        P\left(\lambda|\tau_i, a, b\right) =
        \mathcal{G}\left(\lambda|\alpha, \beta\right) = 
        \frac{\beta^{\alpha}\lambda^{\alpha-1}}{\Gamma(\alpha)}e^{-\lambda\beta},
        ```
        where 
        ```{math}
        \alpha = N + a\qquad\text{and}\qquad\beta = N\bar{\tau} + b.
        ```
        So if we then choose different $a$ and $b$ to describe a prior, we can update $\alpha$ and $\beta$ and use `scipy.stats.gamma(alpha, scale=1/beta)` to manipulate the posterior in Python.  You will explore this process on your own in the following sub-problems.
            
        1. <mark style="background-color: #6b9cee">MVD</mark> Set $a_1 = b_1 = 0.01$ and $a_2 = b_2 = 2$ and plot the *prior* distributions corresponding to these parameters.  Describe these curves qualitatively.
                
        2. <mark style="background-color: #e26563">PCS-1</mark>, <mark style="background-color: #ffda5c">TS-2</mark>, <mark style="background-color: #6b9cee">MVD</mark> Using $a_1$, $b_1$, and then $a_2$, $b_2$, use Equation {eq}`eqn_LambdaPost` to plot posterior PDFs and CDFs for $\lambda^+$ and $\lambda^-$ (using the two lists $\tau^+$ and $\tau_-$, respectively).  Calculate and annotate the mean, median, and mode on these plots.  Compare these values with the MLEs from earlier in the problem.
                
        3. <mark style="background-color: #6b9cee">MVD</mark>, <mark style="background-color: #92c57a">NQP-1</mark> The mean of a gamma distribution can be calculated as the ratio of the shape to the rate parameters ($a/b$), or in the case of our exponentially distributed data and a gamma prior, we get that the posterior mean is
            ```{math}
            \mu_{\lambda} = \frac{a + N}{b + N\bar{\tau}}
            ```
            As in 1.1.4, show how the mean changes as you change the number of samples.  Assess whether the choice of prior impacts the mean.  Annotate the MLE $\hat{\lambda} = 1/\bar{\tau}$.  Do the MLE and posterior mean agree?
                
        4. <mark style="background-color: #6b9cee">MVD</mark>, <mark style="background-color: #92c57a">NQP-1</mark> Using `scipy.stats.gamma.ppf`, annotate the median, $\alpha/2^{\text{th}}$, and $(1-\alpha/2)^{\text{th}}$ percentiles.  How many data points do you need to add before the MLE is inside the credible interval specified by these percentiles?
                
    4. <mark style="background-color: #e26563">PCS-2</mark>, <mark style="background-color: #6b9cee">MVD</mark>, <mark style="background-color: #92c57a">NQP-1</mark>  Use bootstrapping to estimate the *variance* of each of the sets of intervals $\tau^+$ and $\tau^-$.  You should report a point estimate and a confidence interval.  You are free to choose how many bootstrap resamplings to do, how to make the point estimate, and what confidence level you want to report, but you should clearly indicate your decisions.

---

2.