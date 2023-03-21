# Worksheet 2.3: Bootstrapping linear regression


Welcome to the third worksheet of Module 2! The worksheets are designed to work in parallel to the course notes and lectures. They are not intended to be summary assessments at the end of a module, but as **necessary content** that you consume on your way to the assignments (and eventually attacking your own data!). As a result, if there are parts of the worksheet that you don’t understand or are having trouble with, make sure to **ask for help!**


In this worksheet, you will be working with the [linked data set](https://github.com/ejohnson643/WhatDoYourDataSay/blob/main/CourseFiles/Module_2/Resources/auto-mpg.data), which is described [here](https://github.com/ejohnson643/WhatDoYourDataSay/blob/main/CourseFiles/Module_2/Resources/auto-mpg.names).


```{admonition} Worksheet Goals
By the end of this worksheet, you will have practiced the following skills:
- <mark style="background-color: #e26563">PCS-2</mark> Calculating the estimate and confidence interval of the mean using bootstrapping.
- <mark style="background-color: #e26563">PCS-3</mark> Performing OLS linear regression.
- <mark style="background-color: #ffda5c">TS-1</mark> Discussing the differences between different methods for making estimates and intervals.
- <mark style="background-color: #ffda5c">TS-2</mark> Discussing the assumptions of OLS regression.
- <mark style="background-color: #6b9cee">MVD</mark> Illustrating uncertainty in estimates.
- <mark style="background-color: #92c57a">NQP-1</mark> Comparing and critiquing different methods for making estimates and intervals.
- <mark style="background-color: #92c57a">NQP-2</mark> Assessing the quality of a model's fit and methods for improving it.
```

We will be performing estimation of the model parameters $\beta_0$, $\beta_3$, and $\beta_6$ in the model:
```{math}
x_0 = \beta_0 + \beta_3x_3 + \beta_6x_6
```

That is, we think that the first measurement in each observation depends linearly on the **4th** and **7th**.  It will be up to you to use the metadata to learn what these measurements are.

At the end of this worksheet, you should have a figure that shows the model, the data, the parameter estimates, and the spread in those parameter estimates.  This can be one panel or many, but it should be one easy-to-understand figure that has all of those components.

1. Use the `open` and `file.readlines` functions to open the data set.  How does it seem that the measurements in each line of the data are organized?  What type of data are the lines?  How can we `split` the lines and grab the measurements we want (1st, 4th, and 7th columns)?  How can we do this repeatedly for all lines of the data?  You may also have to check the data for missing or invalid values!  (If this is proving too complicated, **ask for help** and try using the `read_csv` function from the `pandas` package.)

	```{note}
	Opening data files and orienting yourself around your data can be a challenge. Spend some time trying to ensure you understand how the data in the file is organized. Sadly, there are a large number of file formats and opening them in a manner that preserves the formatting and metadata can be a pain. Make sure to spend time on ensuring you get this right by going online and searching for "opening xyz file in python". You are surely not the first one to try opening a given file type so lean on the community.
	```

2. Report the number of samples in your data set, save it as $N$, and calculate the mean, median, standard deviation, and IQR (inter-quartile range) of each of the measurements in the data set.

3. Use appropriate plots to show how the different quantities depend on each other.  If you loaded the data as a `pandas.DataFrame`, try using the `sns.pairplot` command.  Does it seem that the variables depend on each other *linearly*?

	```{note}
	The linear regression algorithm will run just as well on nonlinear data as it will on linear data; the algorithm doesn't care what the shape of the data is. It's always a good idea to look at your data to make sure your analyses make sense! Don't apply methods blindly and hope it all works out.
	```

4. Use your preferred linear regression package to fit the model.  What are the values of $\beta_0$, $\beta_3$, and $\beta_6$?  If you're using a package that returns this information, what is the confidence interval for each of the regression coefficients?

5. Generate a new set of $N$ samples by bootstrapping your data set with replacement.  Run linear regression on this new sample and report the regression coefficients.

6. Do the previous step $N_{BOOT}$ times, where $N_{BOOT}$ is a suitably large number.  Present the *distributions* of regression coefficients.  Calculate the 2.5th and 97.5th percentiles of each coefficient.  If you found a confidence interval earlier, compare it to these percentiles.  Plot the point estimates (part 4) of the regression from the data on these distributions; are they near the median, mean, or mode of the distributions?

	```{note}
	It is perhaps wise to now return to the course material to evaluate any patterns/differences that stand out to you. What did you expect? How did you expect the confidence intervals to compare to the IQR of the bootstrapped parameter distribution?
	```