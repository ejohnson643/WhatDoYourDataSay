# Worksheet 2.2: OLS Linear Regression

Welcome to the second worksheet of Module 2! The worksheets are designed to work in parallel to the course notes and lectures. They are not intended to be summary assessments at the end of a module, but as **necessary content** that you consume on your way to the assignments (and eventually attacking your own data!). As a result, if there are parts of the worksheet that you don’t understand or are having trouble with, make sure to **ask for help!**


This worksheet will really be a nuts and bolts tutorial on linear regression in Python.  There are quite a few linear regression functions - most of the main quantitative and statistical packages have one, so we'll be looking at the following functions:

- `np.polyfit`
- `scipy.stats.lingress`
- `sklearn.linear_model`
- `statsmodels.api.OLS`


```{admonition} Worksheet Goals
By the end of this worksheet, you will have practiced the following skills:
- <mark style="background-color: #e26563">PCS-3</mark> Performing OLS linear regression in Python.
- <mark style="background-color: #ffda5c">TS-2</mark> Discussing the assumptions of OLS regression.
- <mark style="background-color: #6b9cee">MVD</mark> Illustrating uncertainty in estimates.
- <mark style="background-color: #92c57a">NQP-2</mark> Assessing the quality of a model's fit to data and how it can be improved.
```

Now, you may be wondering, isn't there a "right" function to use for linear regression? Not really. The reason there are different functions is because each have different abilities. The goal here is to expose you to more than one of them so you start to see that you have options, and in order to navigate those options you have to be willing to explore and have a sense of what you want to do with the analysis. This step, from running someone elses code to do some "standard" analysis of your data, to navigating your own analyses and coding is such a powerful transition. Making this transition means that you are closer to really understanding what analysis has been, can be, done, what you think is interesting, and finally "What do your data say?"!

Recall that you can install new packages using the [Anaconda Navigator](https://docs.anaconda.com/anaconda/navigator/tutorials/manage-packages/#installing-a-package) or via the command line by typing `conda install myPackageThatIWantToInstall`.  For this worksheet, you will likely have to install `sklearn`, which is under the package `scikit-learn`, and `statsmodels`, which can be found with the same name.

1. Use the `help` function to look at the documentation for each of the 4 functions above.
    
    ```{note}
    What does "look at" mean in this context? I typically first try to understand into the conceptual aspects. What is this function essentially trying to do? This is usually indicated in a sentence or two at the very top of the documentation.  The idea is that there's no point getting into function details before knowing what it's doing!

    Then, I look into what the function requires as inputs and what it can return as an output. This often helps you understand how the function is working and can alert you to keywords that might be useful for tailoring functions to your specific needs. 
    ```

2. Create some fake linear data.  That is, create two lists of numbers $X$ and $Y$ that may or may not be linearly related.  Make sure they are not identical and also not completely unrelated or the results will not be that interesting.
    
    ```{note}
	This may seem trivial, but the ability to perform a *controlled experiment* is maybe the most important tool in all of coding, data analysis, and perhaps science. The real power of doing science in a computer is that before analysing real data with all its complexities, we can create a toy version of the problem. So, for example, if you wish to run a linear regression on your real data, first generate some fake data where you can control everything about the stochastic process that generates it. Test your code on the synthetic data by asking whether you can recover the desired properties. This is a good check that: a) your code is doing what you think it is, b) you have a solid sanity check against which to compare your data.
	```

3. Run each of the above functions and examine their output.  Answer the following for each function:
    - How can you extract the regression coefficients?
    - How can you get the residuals?  
    - Which methods report "error" estimates in the coefficients?
    - How easy is this method to use?
    - Does this method fit the intercept ($\beta_0$)?
    
    ```{note}
    These are all crucial questions to answer when you start comparing tools for your analyses. It is important to be able to navigate yourself through the conceptual and coding landscape by noting details and differences between packages and tools.
    ```
	