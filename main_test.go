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
