
import pandas as pd

def add_index_line(file_name):
    df = pd.read_csv(file_name, header=None)
    df = df[[1,0]]
    df.to_csv(file_name+".indexed", sep='\t')

add_index_line('./data/test.ft.csv')
