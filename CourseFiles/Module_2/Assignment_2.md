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

## Problem 1: Poisson Rate Estimation

Consider the bacterial chemotaxis data from the first assignment [here](https://github.com/ejohnson643/WhatDoYourDataSay/blob/main/CourseFiles/Module_2/Resources/omega.txt).  Generate lists of the time intervals, $\tau^+$ and $\tau^-$, that the bacteria spent rotating in the positive and negative directions, respectively.  Feel free to use your work from Assignment 1 or the Assignment 1 Solutions (with proper attribution!).
        
In Assignment 1, you examined the distributions of $\tau^+$ and $\tau^-$ and calculated some summary statistics.  In particular, the bonus problem 1.5 noted that both $\tau^+$ and $\tau^-$ are *exponentially distributed* with a rate parameter, $\lambda$.  In 1.6 you explored the posterior distribution for $\lambda$ under the assumption that $\tau$ are exponentially distributed, and started to think about whether $\lambda=0.2$ was consistent with the data.  After reading the course notes and watching the video lectures, you should now see that this is a *parameter estimation* problem, which we will now perform more carefully.
    
Specifically, in this problem you will examine three different methods for estimating $\lambda$: maximum likelihood estimation, Bayesian estimation, and bootstrapping.  You will examine how these estimates depend on the number of samples in the data, and how the different estimates relate to each other.
        
### 1.1: Maximum Likelihood Estimation and Confidence Intervals

As pointed out in the notes and worksheets, MLEs and Confidence Intervals involve analytically manipulating theoretical models.  In this problem, we will examine three different sets of assumptions, each a refinement on the last, to underscore what this theoretical process actually looks like in practice.

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
    `mean} \pm\,\,z_{\alpha/2} \times`stderror},
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

### 1.2: Visualizing Confidence Intervals

To further unpack the differences in the confidence interval formulas, let's try and visualize the problem more directly.

1. <mark style="background-color: #e26563">PCS-1</mark> Randomly draw $N=100$ intervals from one of the lists ($\tau^+$ or $\tau^-$) and estimate the rate parameter $\lambda$ (calculate a point estimate *and* and interval!).  Save this estimate as `lHat0` and `lConfInt0`.  Use the *exact* confidence interval formula in Equation {eq}`eqn_ExactConfInt`.

2. <mark style="background-color: #e26563">PCS-1,2</mark>, <mark style="background-color: #ffda5c">TS-1</mark> Repeat the previous problem $N_{exp}$ times, where $N_{exp}$ is a large number of your choosing.  Show the distribution of estimates, $\hat{\lambda}$ and describe its shape.  Where is it centered?  On `\lHat0`?  On the value calculated in 1.a?

3. <mark style="background-color: #e26563">PCS</mark>, <mark style="background-color: #ffda5c">TS-1</mark> Equation {eq}`eqn_ApproxConfInt` suggested that $\lambda$ is normally distributed.  Overlay a plot of a normal distribution with mean, $\mu = \hat{\lambda}$, and standard deviation $\sigma = \hat{\lambda}/\sqrt{N}$, where these can be calculated using all the intervals.  Qualitatively discuss how well the distribution of estimates match this curve.

4.<mark style="background-color: #e26563">PCS-2</mark>, <mark style="background-color: #ffda5c">TS-1</mark> The exact confidence interval (Equation {eq}`eqn_ExactConfInt`) used the fact that $\lambda$ is $\chi^2$-distributed.  Use `scipy.stats.chi2.pdf(xCoords, df=2*N, scale=1/(2*N*tauBar))` to overlay a plot of this distribution. Qualitatively discuss how well the distribution of estimates match this curve.

5. <mark style="background-color: #ffda5c">TS-1</mark> , <mark style="background-color: #6b9cee">MVD</mark>  Indicate where `lHat0` and `lConfInt0` fall on the distribution.  Determine the fraction of the $N_{exp}$ estimates that fall in `lConfInt0`.  Discuss this fraction - does it make sense to you?  (If needed, run your code several times to convince yourself!)

6. <mark style="background-color: #ffda5c">TS-1</mark> , <mark style="background-color: #6b9cee">MVD</mark>  Recreate your figure using (empirical) CDFs of the estimates and the theoretical curves.  Show the $\alpha/2^{\text{th}}$ and $(1-\alpha/2)^{\text{th}}$ percentiles of the $N_{exp}$ estimates, the median of the estimates, `lHat0` and `lConfInt0`.  Comment on any new insights from this figure.
        
### 1.3: Maximum A Posteriori Estimation and Credible Intervals

We'll now explore how you can make MAP estimates of the rate parameter, $\lambda$, using Bayes' Theorem.  While in the notes and worksheets we emphasize an empirical approach to this problem, here you'll explore how you can implement a more theoretical approach as well.

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

## Problem 2: Linear Regression 1

In this problem, you're going to explore how changing the "noise" in your data impacts your ability to do linear regression.  In this problem, you'll practice using linear regression routines and parameter estimation methods to make estimates of regression coefficients.  You will also explore how using synthetic data allows you to compare outcomes to ground truth.

### 2.1: Generate Data

1. <mark style="background-color: #6b9cee">MVD</mark> Generate two sets of $N=100$ samples, $x_i$ and $y_i$, where $y_i$ is generated from $x_i$ using the formula $y_i = -1 + x_i + \varepsilon_i$.  In the first set, use $\varepsilon\sim\mathcal{N}(0, 2)$ (the noise is normally distributed with a *variance} of 2), and in the second set use $\varepsilon\sim\mathcal{U}(-\sqrt{2}, \sqrt{2})$ (the noise is *uniformly* distributed from $-\sqrt{2}$ to $\sqrt{2}$).  Plot these two datasets.
        
2. <mark style="background-color: #e26563">PCS-2</mark>, <mark style="background-color: #ffda5c">TS-2</mark>, <mark style="background-color: #6b9cee">MVD</mark> Use linear regression methods (see Worksheet 2.2) to fit the model
    ```{math}
    \hat{y}_i = \beta_0 + \beta_1\x_i
    ```
    to both your synthetic data sets.  Compare the regression coefficients and plot the fitted model over the data.
        
3. <mark style="background-color: #ffda5c">TS-2</mark>, <mark style="background-color: #6b9cee">MVD</mark>, <mark style="background-color: #92c57a">NQP-2</mark>  Calculate the residuals of the data $r_i = \hat{y}_i - y_i$ for each of the models.  Plot the distribution of these residuals.  Are they normally distributed?  Considering your plots of the raw data before you did any fitting, could you have determined "by eye" whether the data had normally distributed noise or not?
        
### 2.2: Estimating Regression Coefficients

Select $N = 20$ of each of your generated data sets.  We're now going to generate confidence regions for $\beta_1$, the regression coefficient, in three ways: (i) a confidence interval from theory, (ii) a credible interval from a posterior distribution, and (iii) a confidence interval from bootstrapping.
    
1. <mark style="background-color: #e26563">PCS-1,3</mark> We won't discuss the theory here, but just as we learned that exponential rate parameters are $\chi^2$-distributed, it can [be shown](https://stats.stackexchange.com/questions/117406/proof-that-the-coefficients-in-an-ols-model-follow-a-t-distribution-with-n-k-d) that regression coefficients are $t$-distributed, so that the formula for their confidence interval is given by
    ```{math}
    \left[
        \hat{\beta}_1 + t_{N-2}(\alpha/2)\frac{s_y}{s_x},\qquad
        \hat{\beta}_1 + t_{N-2}(1-\alpha/2)\frac{s_y}{s_x},
    \right]
    ```
    where
    ```{math}
    s_x = \sqrt{\sum_{i=1}^N(x_i - \bar{x})^2}\,\qquad\text{and}\qquad
    s_y = \sqrt{\frac{\sum_{i=1}^N(y_i - \hat{y}_i)^2}{N-2}}.
    ```
    In these formulas, $\bar{x}$ is the mean of $x_i$, and $\hat{y}_i = \hat{\beta}_0 + \hat{\beta}_1x_i$ is the model prediction and $t_{\nu}(p)$ is the $p$\ts{th} percentile of the Student's $t$-distribution with $\nu$ degrees of freedom (`scipy.stats.t.ppf(p, nu)`).

    Calculate the 95\% confidence interval for your estimates $\beta_1$ from both data sets using this formula.

2. <mark style="background-color: #e26563">PCS-1,3</mark> Alternately, we can pose the problem in a Bayesian way.  Use the priors in the notes for $\beta$ with $\mu_{\beta} = \hat{\beta}_{OLS}$ and $\delta_{\beta} = 10$.  Use $\sigma = \sqrt{\frac{1}{N}\sum_{i=1}^Nr_i^2}$.  Then use Equation 29 from the notes and `scipy.stats.norm.ppf` to calculate 95\% credible intervals for $\beta_1$ from both data sets.

3. <mark style="background-color: #e26563">PCS-2,3</mark> Bootstrap your $N=20$ data points from each synthetic data set many times.  Use your preferred linear regression method to generate an estimate $\hat{\beta}_1$ from each bootstrapped sample.  Report the 95\% confidence interval from the distribution of bootstrapped estimates.

4. <mark style="background-color: #6b9cee">MVD</mark>, <mark style="background-color: #92c57a">NQP-2</mark> Illustrate these intervals with a relevant figure.  Discuss the similarities and differences in these intervals.  Is $\beta_1 = 0$ in any of these intervals?

### 2.3:

<mark style="background-color: #e26563">PCS</mark>, <mark style="background-color: #ffda5c">TS</mark>, <mark style="background-color: #6b9cee">MVD</mark>, <mark style="background-color: #92c57a">NQP</mark>  Choose one of the three methods for generating intervals of confidence and make a figure that demonstrates how this interval depends on the amount of data.  Use either synthetic data set.  Indicate approximately how many data points you need before you can be "sure" that $\beta_1\neq0$.  How many do you need before you're "sure" that it's not 0.9?  0.99?

## Problem 3: Linear Regression II (BONUS)

(This problem is not required for completion of the assignment.  Do not stop reading the assignment here!  There is another problem!)
    
In this problem you're going to deal with *real data*!  You're going to move past generating intervals of confidence and instead think about a model's predictive ability.    

### 3.1:

<mark style="background-color: #6b9cee">MVD</mark>, <mark style="background-color: #92c57a">NQP-2</mark> **BONUS - The Data:** Consider the dataset [here](https://github.com/ejohnson643/WhatDoYourDataSay/blob/main/CourseFiles/Module_2/Resources/abalones.csv). As described [here](https://archive.ics.uci.edu/ml/datasets/Abalone), the data consist of some measurements of [abalones](https://en.wikipedia.org/wiki/Abalone), which are a type of ocean mollusk.  Explore the data set and describe the measurements.  We're going to use the variable `rings` as our response variable; pick two of the continuous measurements (not `sex` or `rings`)  and *standardize* your response variable, $y$ and your covariates $x_1$ and $x_2$ so that they have mean = 0 and standard deviation = 1.  We'll notate the standardized quantities with a * ($y^*$, $x_1^*$, and $x_2^*$).  Plot $y^*$ vs $x_1^*$ and $y^*$ vs $x_2^*$.  State whether you expect a linear model to perform well on this problem.

### 3.2: Testing and Training Sets

1. <mark style="background-color: #e26563">PCS</mark>, <mark style="background-color: #ffda5c">TS</mark> **BONUS:** Break your data into two sets, a training set containing 90\% of the data and a test set containing  the rest.  Fit the model `rings`$\,\,= y^* = \beta_1 x_1^* + \beta_2 x_2^*$ using OLS to the training data.  Give estimates and confidence intervals for the parameters $\beta_i$ (the `rings` column is a proxy for the age of the abalone).  If you're using a specific package or formula to make these calculations, indicate it.  Using 10 distinct training-test splits, compute the average prediction error of your model on the testing data.
        
2. <mark style="background-color: #6b9cee">MVD</mark>, <mark style="background-color: #92c57a">NQP-2</mark> **BONUS:** Make a plot of $y^*$ vs $x_1^*$ and a plot of $y^*$ vs $x_2^*$.  Indicate training data in one color and testing data in another using one of your training-test splits from the previous problem.  Plot some best-fit lines on these figures.  (Since the model has two parameters, the fitting isn't of a line, but of a *plane*, so on the $y^*$ vs $x_1^*$ plot, you'll have to be creative!)
        
3. <mark style="background-color: #6b9cee">MVD</mark>, <mark style="background-color: #92c57a">NQP-2</mark> **BONUS:** For one of the training-testing splits, show the distribution of residuals from the training data and the distribution of residuals from the testing data.  Are they similar in shape?

### 3.3: Where is the error?

1. <mark style="background-color: #6b9cee">MVD</mark> **BONUS:** Make a scatter plot of $x_2^*$ vs $x_1^*$ and overlay a $10\times10$ grid that covers your data.
        
2. <mark style="background-color: #e26563">PCS-3</mark>, <mark style="background-color: #ffda5c">TS-2</mark> **BONUS:** Calculate the average prediction error of the fitted model on the test points that lie in each of the 100 grid cells using 10-fold cross-validation.  If there are no test points in the cell for a particular training-test split, don't incorporate that split into the average prediction error.
        
3. <mark style="background-color: #6b9cee">MVD</mark>, <mark style="background-color: #92c57a">NQP-2</mark> **BONUS:** Display these prediction errors with an appropriate figure.  Describe *where* in $x_2^*$-$x_1^*$ space the model seems to perform better or worse.

### 3.4 Bootstrapping

1. <mark style="background-color: #e26563">PCS-2</mark> **BONUS:** Generate many bootstrapped estimates of the regression coefficients, $\beta_1$ and $\beta_2$.  Only use the training data!
        
2. <mark style="background-color: #e26563">PCS-2</mark>, <mark style="background-color: #6b9cee">MVD</mark> **BONUS:** Plot the distributions of $\hat{\beta}_1$ and $\hat{\beta}_2$ on separate plots.  Show your confidence/credible interval from earlier in the problem as well as the $\alpha/2^{\text{th}}$ and $(1 - \alpha/2)^{\text{th}}$ percentiles of the bootstrapped distributions.
        
3. <mark style="background-color: #ffda5c">TS</mark>, <mark style="background-color: #6b9cee">MVD</mark>, <mark style="background-color: #92c57a">NQP-2</mark> **BONUS:** Make a scatter plot of the bootstrapped $\hat{\beta}_2$ vs $\hat{\beta}_1$.  Overlay the confidence/credible intervals and the  $\alpha/2^{\text{th}}$ and $(1 - \alpha/2)^{\text{th}}$ percentiles as vertical and horizontal lines.  Based on this figure, does it seem that $\beta_2$ and $\beta_1$ can be fit independently?  How would this figure look if you could?

## Problem 4: Fitting a Thermodynamic Model:

As described in the course notes, in [this 2011 paper](https://github.com/ejohnson643/WhatDoYourDataSay/blob/main/CourseFiles/Module_2/Resources/Philips_Garcia_Data/Garcia2011c.pdf) by Garcia and Phillips, the authors developed a model based on the principles of thermodynamics in order to measure the number of LacI repressors present in different strains of *E. coli*, as well as the binding energy of those repressors.  In Figure S8 of their paper, they show how given estimates for the number of repressors, $R$, they can use their model, given in Equation 5 of the paper:
```{math}
:label: eqn_Eq5
\texttt{Fold Change in YFP Fluorescence} = \left(1 + \frac{2R}{N_{NS}}e^{-\beta\Delta\varepsilon_{RD}}\right)^{-1}
```
to generate an estimate for the binding energy of those repressors, $\Delta\varepsilon_{RD}$.  As given in the notes and the paper, $N_{NS} = 5\times10^6$, is the number of Non-Specific DNA binding sites for repressors and RNA Polymerase, and $\beta = 1/k_BT$ is a shorthand factor that  contains the Boltzmann constant.
    
You may or may not be surprised to learn that a common experimental technique for measuring cellular activity is to *take pictures* of cells at specific wavelengths of light.  This measures activity because the cells have been modified so that specific proteins fluoresce at specific wavelengths.  What this looks like then is the images in [this folder](https://github.com/ejohnson643/WhatDoYourDataSay/tree/main/CourseFiles/Module_2/Resources/Philips_Garcia_Data/laci_full_set).  In each of these images, a specific strain of bacteria has been imaged at a couple wavelengths (a yellow one and a red one), and the idea is that the amount of light (number of photons) measured at those wavelengths directly corresponds to how much of a desired protein (LacI) exists in the cells or their environment.
    
### 4.1:

<mark style="background-color: #6b9cee">MVD</mark> The immediate trouble then is that bacteria don't sort themselves nicely into grids or patterns, so we have to either manually indicate where bacteria are in the images or help the computer use the images to detect the bacteria.  This might not seem worth it for this data set, but modern experiments regularly have hundreds or thousands of images, so automating this process is necessary!  (It's also more consistent!)  To help you see what this process looks like, we have prepared a [Python module](https://github.com/ejohnson643/WhatDoYourDataSay/blob/main/CourseFiles/Module_2/Resources/Problem2_4_ImageProcessingPackage.py) for processing these specific images.
        
Download [the module](https://github.com/ejohnson643/WhatDoYourDataSay/blob/main/CourseFiles/Module_2/Resources/Problem2_4_ImageProcessingPackage.py) into the same location that you put the data (and that you are running your notebook).  Use the [assistance notebook here](https://github.com/ejohnson643/WhatDoYourDataSay/blob/main/CourseFiles/Module_2/Resources/Assignment2_Fall2020_Q4_Help.ipynb) to learn how to extract the YFP values from the images.  Describe the workflow of going from images to fold-change in YFP fluorescence in that notebook in your own words.  Use an appropriate figure to show the distribution of YFP fluorescence in each of the 7 strains.  Predict which non-control strain (not `Auto` or `Delta`) contains the most repressors and which contains the least.
        
### 4.2.

<mark style="background-color: #e26563">PCS-3</mark>, <mark style="background-color: #ffda5c">TS</mark>, <mark style="background-color: #6b9cee">MVD</mark>, <mark style="background-color: #92c57a">NQP-2</mark> Using the direct measurements of the repressor copy number, $R$, that are given in Figure 4A of the paper, use least-squares regression to estimate $\beta\Delta\varepsilon_{RD}$ under the model given in Equation {eq}`eqn_Eq5`.  Make sure to report point estimates and intervals of confidence.  Plot your data and the fitted lines.  Plot lines corresponding to the edges of your intervals.  When plotting the data points, use error bars corresponding to the errors in measurement.  These are given by Figure 4A for $R$, and are assumed to be 30\% for the fold-change measurements (this is again from the paper).

Describe and interpret this result.  Does the curve increase or decrease?  Does it have an interesting shape?  Why might this be consistent or inconsistent with what you know about the underlying biology?  Finally, comment on why OLS might not provide the best estimates for the "regression coefficient," $\beta\Delta\varepsilon_{RD}$.

### 4.3: (BONUS)

<mark style="background-color: #e26563">PCS-3</mark>, <mark style="background-color: #ffda5c">TS</mark>, <mark style="background-color: #6b9cee">MVD</mark>, <mark style="background-color: #92c57a">NQP-2</mark> **BONUS:** You may have noted that the model in Equation {eq}`eqn_Eq5` (Equation 5 in the paper) is non-linear!  However, we can sometimes apply transformations or approximations to make things look linear.  In particular, it is often useful to fit the *log-transformed* model, as the logarithm "smooths" out large changes in our responses.
\begin{align*}
    \log(FC) &= -\log\left(
        1 + \frac{2\log(R)}{N_{NS}}
        e^{-\beta\Delta\varepsilon_{RD}}
    \right)
\end{align*}
Fit the log-transformed model using OLS or Bayesian Regression and compare to the earlier results.  Plot both results on a figure with log-scaled axes.

### 4.4: (BONUS)

<mark style="background-color: #ffda5c">TS-2</mark> <mark style="background-color: #92c57a">NQP-2</mark> **BONUS:** In this problem, it might not seem that bootstrapping is a useful tool, since we only have 5 data points.  Thinking especially about how we generated $FC$ and $R$ discuss ways that we could apply a bootstrapping "principle" to this fitting problem.

