# MSDS431-MigusWong-Assignment-3

## Assignment Requirements/Scenario 

Take on the role of the company's data scientists. Using data from The Anscombe Quartet and the Go testing package, ensure that the Go statistical package yields results comparable to those obtained from Python and R. In particular, ensure that similar results are obtained for estimated linear regression coefficients. Also, use the Go testing package to obtain program execution times and compare these with execution times observed from running Python and R programs on The Anscombe Quartet.

The Anscombe Quartet, developed by Anscombe (1973), is a set of four data sets with one independent variable x and one dependent variable y. Simple linear regression of y on x yields identical estimates of regression coefficients despite the fact that these are very different data sets. The Anscombe Quartet provides a telling demonstration of the importance of data visualization. 
 
As part of the program documentation (in a README.md file), include a recommendation to management. Note any concerns that data scientists might have about using the Go statistics package instead of Python or R statistical packages.

The testing package in the Go standard library provides methods for testing and benchmarking, although "benchmarking" with the Go testing library is now what we mean by running a performance benchmark in this assignment. And the go test tool is bundled into the Go programming environment. Bates and LaNou (2023) and Bodner (2024) provide Go programming examples of testing and benchmarking as needed for this assignment. 

## Code Used to test linear regression calculations

### Python Setup

Below is a modified version of the code used to calculate the Anscombe Quartet in Thomas Miller's *Modeling Techniques in Predictive Analytics with Python and R: A Guide to Data Science.(2015)*. The original file can be found in this repository [miller-mtpa-chapter-1-program.py](miller-mtpa-chapter-1-program.py)\. Note that the %%timemit function in jupyter notebook was utilized in order to benchmark execution speed.


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

### GoLang Setup
Go's testing library was used to benchmark [Montana Flyn's Stats Package](https://github.com/montanaflynn/stats/blob/master/regression.go). The package was imported by using:

```
go get github.com/montanaflynn/stats
```

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