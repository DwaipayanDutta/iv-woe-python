# README for Information Value (IV) and Weight of Evidence (WOE) Analysis

## Overview

This README provides an overview of the Information Value (IV) and Weight of Evidence (WOE) concepts, their importance in variable selection during model building, and how to implement these techniques using Python. The content is derived from a detailed blog post on [UCanAnalytics](https://ucanalytics.com/blogs/information-value-and-weight-of-evidencebanking-case/).

## Table of Contents

- [Introduction](#introduction)
- [What is Information Value (IV)?](#what-is-information-value-iv)
- [What is Weight of Evidence (WOE)?](#what-is-weight-of-evidence-woe)
- [Importance of IV and WOE](#importance-of-iv-and-woe)
- [Implementation in Python](#implementation-in-python)
  - [Feature Classes](#feature-classes)
    - [Categorical Feature Class](#categorical-feature-class)
    - [Continuous Feature Class](#continuous-feature-class)
  - [IV Calculation](#iv-calculation)
  - [Visualization](#visualization)
- [Interpreting IV Values](#interpreting-iv-values)
- [Logistic Regression with WOE](#logistic-regression-with-woe)
- [Conclusion](#conclusion)

## Introduction

Information Value (IV) and Weight of Evidence (WOE) are statistical measures used primarily in credit scoring and risk management. They help assess the predictive power of independent variables in relation to a binary dependent variable, such as loan default status.

## What is Information Value (IV)?

Information Value quantifies the predictive power of a feature by measuring how well it separates the good and bad outcomes. It is calculated based on the distribution of good and bad outcomes across different bins or categories of the feature.

### Formula for IV:

$$
IV = \sum \left( \text{Distribution Good} - \text{Distribution Bad} \right) \times WOE
$$


Where:
- **Distribution Good**: Proportion of good outcomes in a bin.
- **Distribution Bad**: Proportion of bad outcomes in a bin.
- **WOE**: Weight of Evidence for that bin.

## What is Weight of Evidence (WOE)?

Weight of Evidence is a transformation technique that converts categorical variables into numerical values. It reflects the strength of evidence provided by each category with respect to the target variable.

### Formula for WOE:

$$
WOE = \ln\left(\frac{\text{Distribution Good}}{\text{Distribution Bad}}\right)
$$


## Importance of IV and WOE

1. **Variable Selection**: IV helps in identifying which variables are useful predictors for modeling.
2. **Handling Categorical Variables**: WOE allows for the effective handling of categorical variables in logistic regression models.
3. **Interpretability**: Both IV and WOE provide clear interpretations regarding the predictive power and relationship between features and target variables.

## Implementation in Python

The implementation involves creating classes for handling categorical and continuous features, calculating IV and WOE, and visualizing results. Below are key components:

### Feature Classes

This section defines classes for handling categorical and continuous features in our dataset.

#### Categorical Feature Class

The `CategoricalFeature` class processes categorical variables by creating bins and calculating relevant statistics.

```python
import pandas as pd

class CategoricalFeature:
    def __init__(self, df, feature):
        self.df = df
        self.feature = feature

    @property
    def df_lite(self):
        df_lite = self.df.copy()
        df_lite['bin'] = df_lite[self.feature].fillna('MISSING')
        return df_lite[['bin', 'label']]
