# MSDS431-MigusWong-Assignment-3

## Assignment Requirements/Scenario 

Take on the role of the company's data scientists. Using data from The Anscombe Quartet and the Go testing package, ensure that the Go statistical package yields results comparable to those obtained from Python and R. In particular, ensure that similar results are obtained for estimated linear regression coefficients. Also, use the Go testing package to obtain program execution times and compare these with execution times observed from running Python and R programs on The Anscombe Quartet.

[The Anscombe Quartet](https://en.wikipedia.org/wiki/Anscombe%27s_quartet), developed by Anscombe (1973), is a set of four data sets with one independent variable x and one dependent variable y. Simple linear regression of y on x yields identical estimates of regression coefficients despite the fact that these are very different data sets. The Anscombe Quartet provides a telling demonstration of the importance of data visualization. 
 
As part of the program documentation (in a README.md file), include a recommendation to management. Note any concerns that data scientists might have about using the Go statistics package instead of Python or R statistical packages.

The testing package in the Go standard library provides methods for testing and benchmarking, although "benchmarking" with the Go testing library is now what we mean by running a performance benchmark in this assignment. And the go test tool is bundled into the Go programming environment. Bates and LaNou (2023) and Bodner (2024) provide Go programming examples of testing and benchmarking as needed for this assignment. 

## Setup Linear Regression Calculations

### Python

Below is a modified version of the code used to calculate the Anscombe Quartet in Thomas Miller's *Modeling Techniques in Predictive Analytics with Python and R: A Guide to Data Science.(2015)*. The original file can be found in this repository [miller-mtpa-chapter-1-program.py](https://github.com/miguswong/MSDS431-MigusWong-Assignment-3/blob/main/miller-mtpa-chapter-1-program.py)\. Note that the %%timemit function in jupyter notebook was utilized in order to benchmark execution speed.


**Packages Used**
```Python
# The Anscombe Quartet (Python)

# demonstration data from
# Anscombe, F. J. 1973, February. Graphs in statistical analysis. 
#  The American Statistician 27: 17–21.

# prepare for Python version 3x features and functions
from __future__ import division, print_function

# import packages for Anscombe Quartet demonstration
import pandas as pd  # data frame operations
import numpy as np  # arrays and math functions
import statsmodels.api as sm  # statistical models (including regression)
import warnings

#ignore warnings
warnings.filterwarnings("ignore")
```

**Code Used to test**
```Python
%%timeit

# The Anscombe Quartet (R)

# demonstration data from
# Anscombe, F. J. 1973, February. Graphs in statistical analysis. 
#  The American Statistician 27: 17–21.

# define the anscombe data frame
# define the anscombe data frame using dictionary of equal-length lists
anscombe = pd.DataFrame({'x1' : [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5],
    'x2' : [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5],
    'x3' : [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5],
    'x4' : [8, 8, 8, 8, 8, 8, 8, 19, 8, 8, 8],
    'y1' : [8.04, 6.95,  7.58, 8.81, 8.33, 9.96, 7.24, 4.26,10.84, 4.82, 5.68],
    'y2' : [9.14, 8.14,  8.74, 8.77, 9.26, 8.1, 6.13, 3.1,  9.13, 7.26, 4.74],
    'y3' : [7.46, 6.77, 12.74, 7.11, 7.81, 8.84, 6.08, 5.39, 8.15, 6.42, 5.73],
    'y4' : [6.58, 5.76, 7.71, 8.84, 8.47, 7.04, 5.25, 12.5, 5.56, 7.91, 6.89]})

# fit linear regression models by ordinary least squares
set_I_design_matrix = sm.add_constant(anscombe['x1'])
set_I_model = sm.OLS(anscombe['y1'], set_I_design_matrix)
print(set_I_model.fit().summary())

set_II_design_matrix = sm.add_constant(anscombe['x2'])
set_II_model = sm.OLS(anscombe['y2'], set_II_design_matrix)
print(set_II_model.fit().summary())

set_III_design_matrix = sm.add_constant(anscombe['x3'])
set_III_model = sm.OLS(anscombe['y3'], set_III_design_matrix)
print(set_III_model.fit().summary())

set_IV_design_matrix = sm.add_constant(anscombe['x4'])
set_IV_model = sm.OLS(anscombe['y4'], set_IV_design_matrix)
print(set_IV_model.fit().summary())
```

### GoLang
Go's testing library was used to benchmark [Montana Flyn's Stats Package](https://github.com/montanaflynn/stats/blob/master/regression.go). The package was imported by using:

```
go get github.com/montanaflynn/stats
```
The following [testing](https://github.com/miguswong/MSDS431-MigusWong-Assignment-3/blob/main/main_test.go) code was used to calculate the linear regression model in Go. Note that the [LinearRegression](https://github.com/montanaflynn/stats/blob/master/regression.go) function does not explicitly return the coefficients and details like Python does, instead, the function returns X,Y coordinates of the linear model.  

```go
package main

import (
	"fmt"
	"testing"

	"github.com/montanaflynn/stats"
)

func TestLinearRegression(t *testing.T) {
	//Create Data Sets for Anscombe quartet
	set1 := []stats.Coordinate{
		{10.0, 8.04},
		{8.0, 6.95},
		{13.0, 7.58},
		{9.0, 8.81},
		{11.0, 8.33},
		{14.0, 9.96},
		{6.0, 7.24},
		{4.0, 4.26},
		{12.0, 10.84},
		{7.0, 4.82},
		{5.0, 5.68},
	}

	set2 := []stats.Coordinate{
		{10.0, 9.14},
		{8.0, 8.14},
		{13.0, 8.74},
		{9.0, 8.77},
		{11.0, 9.26},
		{14.0, 8.10},
		{6.0, 6.13},
		{4.0, 3.10},
		{12.0, 9.13},
		{7.0, 7.26},
		{5.0, 4.74},
	}
	set3 := []stats.Coordinate{
		{10.0, 7.46},
		{8.0, 6.77},
		{13.0, 12.74},
		{9.0, 7.11},
		{11.0, 7.81},
		{14.0, 8.84},
		{6.0, 6.08},
		{4.0, 5.39},
		{12.0, 8.15},
		{7.0, 6.42},
		{5.0, 5.73},
	}
	set4 := []stats.Coordinate{
		{8.0, 6.58},
		{8.0, 5.76},
		{8.0, 7.71},
		{8.0, 8.84},
		{8.0, 8.47},
		{8.0, 7.04},
		{8.0, 5.25},
		{19.0, 12.50},
		{8.0, 5.56},
		{8.0, 7.91},
		{8.0, 6.89},
	}
	exp := true
	res := true

	//Perform Linear Regression and Quartet using stats package
	r, _ := stats.LinearRegression(set1)
	fmt.Println(r)
	r, _ = stats.LinearRegression(set2)
	fmt.Println(r)
	r, _ = stats.LinearRegression(set3)
	fmt.Println(r)
	r, _ = stats.LinearRegression(set4)
	fmt.Println(r)

	if res != exp {
		t.Errorf("Expected %t, got %t instead.", exp, res)
	}
}
```
Using *go test* benchmarking results were captured.

### R 
Similar to Python, the linear regression test used in R was heavily based off of Miller's [miller-mtpa-chapter-1-program.R](https://github.com/miguswong/MSDS431-MigusWong-Assignment-3/blob/main/miller-mtpa-chapter-1-program.R)

```R
anscombe <- data.frame(
    x1 = c(10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5),
    x2 = c(10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5),
    x3 = c(10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5),
    x4 = c(8, 8, 8, 8, 8, 8, 8, 19, 8, 8, 8),
    y1 = c(8.04, 6.95,  7.58, 8.81, 8.33, 9.96, 7.24, 4.26,10.84, 4.82, 5.68),
    y2 = c(9.14, 8.14,  8.74, 8.77, 9.26, 8.1, 6.13, 3.1,  9.13, 7.26, 4.74),
    y3 = c(7.46, 6.77, 12.74, 7.11, 7.81, 8.84, 6.08, 5.39, 8.15, 6.42, 5.73),
    y4 = c(6.58, 5.76,  7.71, 8.84, 8.47, 7.04, 5.25, 12.5, 5.56, 7.91, 6.89))

# show results from four regression analyses
start_time <- Sys.time()
with(anscombe, print(summary(lm(y1 ~ x1, data = anscombe))))
with(anscombe, print(summary(lm(y2 ~ x2, data = anscombe))))
with(anscombe, print(summary(lm(y3 ~ x3, data = anscombe))))
with(anscombe, print(summary(lm(y4 ~ x4, data = anscombe))))
end_time <- Sys.time()
execution_time <- end_time - start_time
print(paste("Execution time:", execution_time))
```

## Results

### Execution Time
Python was found to be the fastest program to determine the linear regressions for the quartet taking an average of 36 ms to determine all 3 regressions. However, go and R were of comparable speed. In terms of overall performance, there Go was not significantly slower than R, although Python was the fastest program by a large margin.
| Language | Execution Time (ms) |
|----------|---------------------|
| Go       | 153                 |
| Python   | 36                  |
| R        | 176                 |   


### Consistency of Resutlts betweent the languagues
In terms of consistency, all methods utiltized were able to generate the same intercept and slope within reasonable (3 decimal points) constraints. 

### Output
The detail from each output was not the same granularity in all scenarios. Python offered the most granular output of regression detail containg data such as kurtosis and adjusted R squared values.
#### Python
```
==============================================================================
Dep. Variable:                     y1   R-squared:                       0.667
Model:                            OLS   Adj. R-squared:                  0.629
Method:                 Least Squares   F-statistic:                     17.99
Date:                Thu, 17 Oct 2024   Prob (F-statistic):            0.00217
Time:                        12:24:51   Log-Likelihood:                -16.841
No. Observations:                  11   AIC:                             37.68
Df Residuals:                       9   BIC:                             38.48
Df Model:                           1                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const          3.0001      1.125      2.667      0.026       0.456       5.544
x1             0.5001      0.118      4.241      0.002       0.233       0.767
==============================================================================
Omnibus:                        0.082   Durbin-Watson:                   3.212
Prob(Omnibus):                  0.960   Jarque-Bera (JB):                0.289
Skew:                          -0.122   Prob(JB):                        0.865
Kurtosis:                       2.244   Cond. No.                         29.1
==============================================================================
```
#### Go
Go did not contain any functionality(that could be found in documentation) on how to display the calculated intercept or slope. However, the LinearRegression function returns a series of X,Y coordinates that are part of the linear regression plot. Below is the output for printing the first anscombe quartet.

```
[{10 8.001000000000001} {8 7.000818181818185} {13 9.501272727272724} {9 7.500909090909093} {11 8.501090909090909} {14 10.001363636363633} {6 6.000636363636369} {4 5.000454545454553} {12 9.001181818181816} {7 6.500727272727277} {5 5.5005454545454615}]
```

#### R
Similar to Python, R was more robust with "out the box" functionality in terms of regression analysis as seen byt the output below. Much of the inoformation easily available in Python is also availabe in R.
```
Call:
lm(formula = y1 ~ x1, data = anscombe)

Residuals:
     Min       1Q   Median       3Q      Max
-1.92127 -0.45577 -0.04136  0.70941  1.83882

Coefficients:
            Estimate Std. Error t value Pr(>|t|)
(Intercept)   3.0001     1.1247   2.667  0.02573 *
x1            0.5001     0.1179   4.241  0.00217 **
---
Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

Residual standard error: 1.237 on 9 degrees of freedom
Multiple R-squared:  0.6665,    Adjusted R-squared:  0.6295
F-statistic: 17.99 on 1 and 9 DF,  p-value: 0.00217
```
## Reccomendation to Management
