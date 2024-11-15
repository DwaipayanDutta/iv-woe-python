# Information Value (IV) and Weight of Evidence (WOE) Analysis

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
```

#### Continuous Feature Class
ContinuousFeature class handles continuous variables by generating bins based on quantiles and ensuring that each bin has a minimum size.

```python
import pandas as pd
import scipy.stats as stats

class ContinuousFeature:
    def __init__(self, df, feature):
        self.df = df
        self.feature = feature
        self.bin_min_size = int(len(self.df) * 0.05)

    def __generate_bins(self, bins_num):
        df = self.df[[self.feature, 'label']].copy()
        df['bin'] = pd.qcut(df[self.feature], bins_num, duplicates='drop').apply(lambda x: x.left).astype(float)
        return df

    def __generate_correct_bins(self, bins_max=20):
        for bins_num in range(bins_max, 1, -1):
            df = self.__generate_bins(bins_num)
            df_grouped = pd.DataFrame(df.groupby('bin').agg({self.feature: 'count', 'label': 'sum'})).reset_index()
            r, p = stats.spearmanr(df_grouped['bin'], df_grouped['label'])
            if (
                abs(r) == 1 and  # Check if WOE for bins are monotonic
                df_grouped[self.feature].min() > self.bin_min_size and  # Check if bin size is greater than 5%
                not (df_grouped[self.feature] == df_grouped['label']).any()  # Check if number of good and bad is not equal to 0
            ):
                break
        return df

    @property
    def df_lite(self):
        df_lite = self.__generate_correct_bins()
        # Handle missing values without inplace assignment
        df_lite['bin'].fillna('MISSING', inplace=True)
        return df_lite[['bin', 'label']]
```

#### IV Calculation
The IV class calculates Information Value based on the defined features.

```python
import numpy as np

class IV:
    @staticmethod
    def __perc_share(df, group_name):
        return df[group_name] / df[group_name].sum()

    def __calculate_perc_share(self, feat):
        df = feat.df_lite.groupby('bin').agg({feat.target_column: ['count', 'sum']}).reset_index()
        df.columns = [feat.feature, 'count', 'good']
        df['bad'] = df['count'] - df['good']
        return df

    def __calculate_woe(self, feat):
        df = self.__calculate_perc_share(feat)
        
        # Calculate percentages while avoiding division by zero
        total_good = df['good'].sum()
        total_bad = df['bad'].sum()

        # Avoid division by zero by adding a small value (epsilon)
        epsilon = 1e-10
        
        # Calculate WOE safely
        with np.errstate(divide='ignore', invalid='ignore'):
            df['perc_good'] = (df['good'] + epsilon) / (total_good + epsilon)
            df['perc_bad'] = (df['bad'] + epsilon) / (total_bad + epsilon)
            df['woe'] = np.log(df['perc_good'] / df['perc_bad'])
        
        return df

    def calculate_iv(self, feat):
        iv_df = self.__calculate_woe(feat)

        # Calculate IV safely while avoiding NaN values in WOE calculation
        iv_df['iv'] = (iv_df['perc_good'] - iv_df['perc_bad']) * iv_df['woe']
        
        return iv_df, iv_df['iv'].sum()

    @staticmethod
    def interpretation(iv):
        if iv < 0.02:
            return 'useless'
        elif iv < 0.1:
            return 'weak'
        elif iv < 0.3:
            return 'medium'
        elif iv < 0.5:
            return 'strong'
        else:
            return 'suspicious'
```

#### Visualization
This section provides functions to visualize IV and WOE values
```python
import matplotlib.pyplot as plt
import seaborn as sns

def draw_iv(feat):
    iv_df, iv_value = feat.calculate_iv()
    fig, ax = plt.subplots(figsize=(10,6))
    sns.barplot(x=feat.feature, y='woe', data=iv_df, palette='viridis')
    ax.set_title('WOE visualization for: ' + feat.feature)
    plt.show()
```
# Interpreting IV Values

The interpretation of IV values can be summarized as follows:

| Information Value | Predictive Power                |
|-------------------|----------------------------------|
| < 0.02            | Useless for prediction            |
| 0.02 to 0.1       | Weak predictor                   |
| 0.1 to 0.3        | Medium predictor                 |
| 0.3 to 0.5        | Strong predictor                 |
| > 0.5             | Suspicious or too good to be true|

## Logistic Regression with WOE

Once WOE values are calculated, they can be used as independent variables in logistic regression models, enhancing model performance by providing a linear relationship with the log odds.

### Example Logistic Regression Output

| Predictor         | Coefficient | Std Error | z-value | p-value |
|-------------------|-------------|-----------|---------|---------|
| Constant          | -3.66223    | 0.0263162 | -139.16 | 0       |
| WOE Age           | -1          | 0.0796900 | -12.55  | 0       |
