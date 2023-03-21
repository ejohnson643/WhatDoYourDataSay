# Worksheet 2.1: An introdution to bootstrapping

Welcome to the first worksheet of Module 2! The worksheets are designed to work in parallel to the course notes and lectures. They are not intended to be summary assessments at the end of a module, but as **necessary content** that you consume on your way to the assignments (and eventually attacking your own data!). As a result, if there are parts of the worksheet that you don’t understand or are having trouble with, make sure to **ask for help!**


In this worksheet you will be introduced to the nuts and bolts of **bootstrapping**.  As we'll emphasize here -- and often in the course -- bootstrapping is simply the computational technique of leveraging your computer's ability to easily generate many random numbers in order to *re-sample* our data to make estimates and confidence intervals. 

Unlike ML and MAP estimation, where the heavy lifting is largely mathematical, bootstrapping is a computational method, so you'll need to practice *doing* it before it might be intuitive.  As you'll see, the gist of the method is that re-sampling our data is as close as we can get to approximating the underlying data distribution without actually going out and collecting new data.  By calculating our parameter of interest from many re-samples, we can approximate the distribution of that parameter as if we had done many experiments.

```{admonition} Worksheet Goals
By the end of this worksheet, you will have practiced the following skills:
- <mark style="background-color: #e26563">PCS-1</mark> Calculating the estimate and confidence interval of the mean using theoretical formulas.
- <mark style="background-color: #e26563">PCS-2</mark> Calculating the estimate and confidence interval of the mean using bootstrapping.
- <mark style="background-color: #ffda5c">TS-1</mark> Discussing the differences between different methods for making estimates and intervals.
- <mark style="background-color: #6b9cee">MVD</mark> Illustrating uncertainty in estimates.
- <mark style="background-color: #92c57a">NQP-1</mark> Comparing and critiquing different methods for making estimates and intervals.
```

1. Using `np.random.randn` generate a list of $N=100$  random numbers distributed according to the **standard normal distribution** (mean,$\mu = 0$ and variance, $\sigma^2 = 1$).  Save these numbers as a variable called `realData1`. Calculate the mean using `np.mean` and the 95\% confidence interval using the formula

	```{math}
	\left[
	\mu\pm z_{\alpha/2} \frac{\sigma}{\sqrt{N}}
	\right],
	```
	where $\mu$ is the sample mean, $\sigma$ is the sample standard deviation, $N$ is the number of samples, and $z_{p}$ is the $p^{\text{th}}$ percentile of the standard normal distribution (`scipy.stats.norm.ppf`).

	```{note}
	If you are concerned about a Python function, you should always use the `help(function_Im_worried_about)` utility in Jupyter and the **internet** to help you understand how to use it.  In particular, you should make sure you look-up/understand (in more or less this order):
	- The outputs of the function
	- The inputs of the function
	- Examples of usage of the function (typically near the bottom of the documentation)
	- The keyword options of the function

	&nbsp;

	In the same vein, if you think that there's something you want to do in Python, you should Google "normal random Python", for example and see what pops up.  This may sound like cheap advice, but it is actually an essential skill for a Python programmer.  As we mentioned, one of the benefits of Python is that it's so widely used and developed.  You will never take advantage of that without Googling your problems.

	&nbsp;

	Also, $z_{\alpha/2}\approx-1.96$ if $\alpha = 0.05$, just as a reference.
	```

2. **In a new cell** generate $N_{EXP} = 200$ new sets of $N=100$ normally distributed numbers.  Calculate their means.  Based on your understanding of a confidence interval, how many of these means would you expect to lie in the confidence interval you generated in problem 2?  How many of the means actually lie in your confidence interval?

	```{note}
	If this surprises you, run your code a couple times (make sure to change the seed)!  What this should hopefully start to point out is that a confidence interval is not quite what you think it is!

	&nbsp;

	Also, you should use this problem as an example of how you can use Python and your computer to *check your intuition} about theoretical concepts!  We literally do things like this all the time as a simple way to make sure our understanding of things isn't wildly off-base.
	```

3. Using your $N_{EXP}=200$ new sets of $N=100$ normally distributed numbers, calculate $N_{EXP}=200$ confidence intervals.  In how many confidence intervals did your original mean (the mean of `randData1`) lie?

	```{note}
	For reference, when I ran this, I got 172, or 86\% of intervals.  So not really anywhere near 95\%...
	```

4. Repeat the above, with $N_{EXP} = 10000$.  Then try $N = 5000$.  What can you deduce about confidence intervals?  (Hint: try [changing the seed](https://towardsdatascience.com/random-seeds-and-reproducibility-933da79446e3) in step 1 and running everything again.)

5. The true mean of the underlying distribution is $\mu=0$ because the data *actually come* from a standard normal distribution.  In how many of your confidence intervals does $\mu = 0$ lie?  (Rerun your code a few times to check that your result is "stable".)

	```{note}
	This is what makes confidence intervals so tricky!  [This website](https://rpsychologist.com/d3/CI/) animates the simulation you've just done.  When you're using the mean of `randData1` as a comparison, you can think of adding another vertical dashed line in the moving plot on the right.

	In any case, you can think of confidence intervals as being an assessment of whether your *experiment* captured the true parameter, not whether the true parameter is in your interval.
	```

6. Type `help(np.random.choice)` into a Jupyter cell and read the documentation for this function.  In particular, note the keyword arguments and describe them to yourself.  Try using the function on a few test cases of your own design to see if it works the way you want.

	```{note}
	It is absolutely our intention that by the end of this course (or worksheet) you are sick of us telling you to test functions, build code slowly, and add print statements!
	```

7. Re-sample `randData1` *with replacement* $N_{Boot} = 200$ times.  Calculate the mean of each of these re-samplings.

	```{note}
	For many of you, this is your first bootstrapping!  Congrats!  Now, make sure to check that you've done what you think you've done.  It's likely that you won't be able to check by-eye whether this has gone correctly, so what can you do?  (It certainly is likely that you won't raise any actual errors.)  This is where testing new functions on smaller problems becomes invaluable!
	```

8. Before writing any more code, write down what you think the *mean* of your bootstrapped means is.  

	```{note}
	Again, getting in the habit of forming expectations based on your conceptual understanding *before* you do things isn't just some pedantry that we're peddling.  It's literally the only way that you'll be able to learn anything from this work.  If you don't have expectations, it will just be numbers and plots and you'll never know if you're right or wrong!
	```

9. Calculate the mean of your bootstrapped means.  How does it compare to the mean of `randData1`?  What does the *distribution* of these resampled means look like?

10. Consider your array of $N_{Exp} = 200$ means generated from independent "experiments".  How does the distribution of these means compare to the distribution of resampled means?

11. Calculate the $2.5^{\text{th}}$ and $97.5^{\text{th}}$ percentiles of the resampled means.  How do they compare to the confidence interval generated in Problem 1?

	```{note} Most of this worksheet has focused on exploring calculated confidence intervals, but now we get to the real utility of bootstrapping: we can get confidence intervals *computationally*!
	```

12. Increase the number of both resamplings and experiments to 10,000 ($N_{EXP}$ and $N_{Boot}$).  How do the distributions of means change?  How does the mean of the resampled means compare to the mean of `randData1`?  How do the percentiles of the resampled means compare to the confidence interval of `randData1`?  How many experiments yield confidence intervals that contain `randData1`'s mean and/or 0?  (If you have not hardcoded $N_{Exp} = 200$ and $N_{BOOT} = 200$, this will be easy to re-run!)
