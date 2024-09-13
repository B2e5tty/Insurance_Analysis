import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

class EDA:
    def __init__(self, dataframe: pd.DataFrame):
        self.df = dataframe

    def overview(self):
        print("Shape of the dataframe:")
        print(f"{self.df.shape}\n")

        print("Datatypes of each columns:")
        print(f"{self.df.dtypes}\n")

        print("Information on the data:")
        print(f"{self.df.info()}\n")

        print("Describe the numerical column statistics:")
        print(f"{self.df.describe()}\n")

        print("The first five rows of the data:")
        print(f"{self.df.head(5)}\n")

        print("The bottom five rows of the date:")
        print(f"{self.df.tail(5)}\n")

    def columns_with_null(self):
        for col in self.df.columns:
            if self.df[col].isnull().sum() > 0:
                print(col)

        

                     


