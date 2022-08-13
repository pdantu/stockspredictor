import warnings
warnings.simplefilter(action='ignore', category=Warning)
from fileinput import filename
import pandas as pd 
import numpy as np
import os
from os import listdir
import glob



path = os.getcwd()
symbols = []
def main():
    
    f_list = loop(path)
    
    d_list = []
    for name in f_list:
        df = pd.read_csv('{0}/metrics/{1}'.format(path,name))
        print('Processing: ', name)
        d_list = process(d_list,df,name)

    newDF = pd.concat(d_list)
    newDF.to_csv('test.csv')

def find_csv_filenames( path_to_dir, suffix=".csv" ):
    filenames = listdir(path_to_dir)
    return [ filename for filename in filenames if filename.endswith( suffix ) ]

def loop(path):
    path += "/metrics"
    filenames = find_csv_filenames(path)
    return filenames


def process(d_list,df,sector):
    sector = sector[:sector.find("-")]
    df = df[(df['Beta'] < 0.5) & (df['Current Ratio'] > 1) & (df['PEG Ratio'] < 0.5)  ]
    df.rename(columns={df.columns[0]:"Symbol"}, inplace=True)
    df.insert(0, "Sector", sector)
    d_list.append(df)
    return d_list
if __name__ == "__main__":
    main()