# main libraries
import os
import pandas as pd
import numpy as np

# visualization libraries/modules
import matplotlib.pyplot as plt
import seaborn as sns

from datetime import datetime
import requests

# for learning & filling-in missing values
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


'''function for retrieving the merged sales dataframe'''
def get_sales_df():
    filename = "merged_sales.csv"
    if os.path.isfile(filename):
        df = pd.read_csv(filename)

        # let's print the shape
        print(f'df shape: {df.shape}')

        return df


'''function for cleaning the sales dataframe'''
def clean_sales_df(df):

    # dropiing unneeded upc columns
    df = df.drop(columns = ["item_upc12", "item_upc14"])

    # convert the sale date to proper DateTime
    df["sale_date"] = pd.to_datetime(df["sale_date"], infer_datetime_format=True)

    # updating column data types
    df[["item", "store", "store_zipcode"]] = df[["item", "store", "store_zipcode"]].astype(str)

    # setting date to index
    df = df.set_index("sale_date").rename_axis(None).sort_index()

    # isolating and creating a month and day of the week column 
    df["month"] = df.index.strftime("%B")

    # creating a day of the week column
    df["month_and_day"] = df.index.strftime("%A")

    # creating a "sales total" column 
    df["total_sales"] = df["sale_amount"] * df["item_price"]

    # let's print the shape
    print(f'df shape: {df.shape}')

    # return the cleaned* df
    return df


'''function for retrieving the merged sales dataframe'''
def get_energy_df():
    filename = "german_energy.csv"
    if os.path.isfile(filename):
        return pd.read_csv(filename, index_col=[0])


'''function for cleaning the german enery dataframe'''
def clean_energy_df(df):
    
    # changing data column to pd.datetime type
    df["Date"] = pd.to_datetime(df["Date"], infer_datetime_format=True)

    # setting the date column as index
    df = df.set_index("Date").sort_index()

    # adding a year column to the df
    df["year"] = df.index.strftime("%Y")

    # adding a month column
    df["month"] = df.index.strftime("%B")

    return df


'''function that uses sklearn's iterative imputer to learn from continuous values
and imputes the missing/null values in the energy dataframe'''
def fill_energy_nulls(df):
    # creating a day of the year column for ml null imputer process
    df["day_of_the_year"] = df.index.dayofyear

    # initializing the imputer
    imputer = IterativeImputer(
            missing_values = np.nan, \
            skip_complete = True, \
            random_state = 123)

    # fitting the imputer
    imputed = imputer.fit(
        df[[
        'Consumption',
        'Wind',
        'Solar',
        'Wind+Solar',
        'day_of_the_year']])

    # predicting/transforming values from learned values
    df_imputed = imputed.transform(
    df[[
        'Consumption',
        'Wind',
        'Solar',
        'Wind+Solar',
        'day_of_the_year']])

    # creating a new df of learned values
    df_imputed = pd.DataFrame(df_imputed, index = df.index)

    # adding the values where nulls exist
    df[['Consumption','Wind', 'Solar','Wind+Solar', 'day_of_the_year']] = df_imputed

    # returning the new df
    return df