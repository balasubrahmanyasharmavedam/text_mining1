# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 01:02:37 2018

@author: subra
"""

import time
import datetime
import pandas as pd

import os
import tkinter as tk
from tkinter.filedialog import askopenfilename

start_time = time.time()




def import_csv_data():
    global v
    global df
    csv_file_path = askopenfilename(filetypes =[('csv file','*.csv')])
    #print(csv_file_path)
    v.set(csv_file_path)
    df = pd.read_csv(csv_file_path)

root = tk.Tk()
root.title("Layer_4(prototype)")
root.geometry("320x50")
tk.Label(root, text='File Path').grid(row=0, column=0)

v = tk.StringVar()
path = tk.StringVar()
entry = tk.Entry(root, textvariable=v).grid(row=0, column=1)

tk.Button(root, text='Browse Data Set',command=import_csv_data).grid(row=4, column=0)

tk.Button(root, text='Start Binning',command=root.destroy).grid(row=4, column=2)

root.mainloop()

tf = time.time()
ft = datetime.datetime.fromtimestamp(tf).strftime('%Y-%m-%d_%H_%M_%S')
file_path = "E:\\Darwin\\{}\\".format(ft)
directory = os.path.dirname(file_path)
if not os.path.exists(directory):
    os.makedirs(directory)
    os.makedirs("E:\\Darwin\\{}\\Pre-processing\\Layer-4".format(ft))
path_d='E:\Darwin\{}\Pre-processing\Layer-4'.format(ft)

df1=df.copy()

threshold = round((5/len(df.index))*100) # Remove items less than or equal to threshold
 
for col in df:
    if (df[col].dtype=='object'):
        count = df[col].value_counts()
        vals_to_remove = count[count <= threshold].index.values
        mrge=df[col].value_counts()[len(df[col].value_counts())-2]
        assign=count[count == mrge].index.values
        df[col].loc[df[col].isin(vals_to_remove)] = assign[0]



    


ts = time.time()

st = datetime.datetime.fromtimestamp(ts).strftime('%Y%m%d_%H_%M_%S')
#print( st)

df.to_csv(path_d + '/' + 'StratBinned_{}.csv'.format(st))
df1.to_csv(path_d + '/' + 'NotBinned_{}.csv'.format(st))


print("--- %s seconds ---" % (time.time() - start_time))

