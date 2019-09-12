# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 10:06:35 2019

@author: tancetti
"""

import pandas as pd
import os
import datetime
import matplotlib.pyplot as plt
import scipy
import math

def readCsv(folder):
    l_files=[]
    
    for file in os.listdir(folder):
        l_files.append(file)
        
    # select last filename in list with recent measures
    filename=l_files[-1]
    path=folder+'/'+filename
    df_meas=pd.read_csv(path,sep=';',encoding="ISO-8859-1")
    df_meas=df_meas.astype('str')
    
    # convert from str to float
    for col in df_meas.columns:
        if col != "Timestamp":
            df_meas=df_meas.apply(lambda x: x.str.replace(',','.'))       
    for col in df_meas.columns:
        if col != "Timestamp":
            df_meas[col] = pd.to_numeric(df_meas[col], errors='coerce')
        
    # convert dates from str to datetime
    l_dates=[] 
    for t in df_meas['Timestamp'].values.tolist():
        l_dates.append(datetime.datetime.strptime(t, '%Y-%m-%d %H:%M:%S'))
    df_meas['Timestamp']=l_dates
    
    # set columns labels
    l_col=['Timestamp']
    for col in df_meas.columns:
        if col != "Timestamp":
            pos=[]
            for i in range(0,len(col)):
                if col[i]=='.':
                    pos.append(i)
            um=col[-4:]
            l_col.append(col[pos[0]+1:pos[1]]+um)
    df_meas.columns=l_col
    
    df_meas=df_meas.set_index("Timestamp")
   
    return(df_meas)

def zeroRead(df_meas,zero_datetime):
    # set zero date
    zero_date=zero_datetime.date()
    
    l_dates=[]
    for index,row in df_meas.iterrows():
        if index.date() == zero_date:
            l_dates.append(index)
    
    s_zero=df_meas.loc[l_dates].mean()
    
    return(s_zero)
    
def deltaCalc(df_meas,s_zero,zero_datetime):
    df_delta=pd.DataFrame()
    
    for col in df_meas.columns:
        delta=df_meas[col]-s_zero[col]
        df_delta[col]=delta
    
    df_delta=df_delta.loc[(df_delta.index>zero_datetime)]
    return(df_delta)
    
# calculate Paerson index
def PearsonCorr(df_delta):  
    
    # fill na values
    df_delta_fill=df_delta.fillna(method='ffill')
    df_delta_fill=df_delta_fill.drop(df_delta_fill.index[0])
    
    # no sfasamento
    d_coeff_Pear_s0={}
    for i in range(1,len(df_delta_fill.columns),2):
        T_col=df_delta_fill.columns[i-1]
        data_col=df_delta_fill.columns[i]
        
        data = df_delta_fill[data_col].values.tolist()
        T=df_delta_fill[T_col].values.tolist()
        
        c=scipy.stats.pearsonr(T, data)

        d_coeff_Pear_s0[data_col]=c
        
    l_c=[]
    for key in d_coeff_Pear_s0.keys():
        l_c.append((1+d_coeff_Pear_s0[key][0])**2)
        s_0=math.sqrt(sum(l_c)/len(l_c))
    
    # sfasamento 1 h
    d_coeff_Pear_s1={}
    for i in range(1,len(df_delta_fill.columns),2):
        T_col=df_delta_fill.columns[i-1]
        data_col=df_delta_fill.columns[i]
        
        data = df_delta_fill[data_col].values.tolist()
        data_s1=data[1:]
        T=df_delta_fill[T_col].values.tolist()
        T_s1=T[:(len(T)-1)]
        
        c_1=scipy.stats.pearsonr(T_s1, data_s1)

        d_coeff_Pear_s1[data_col]=c_1
    
    l_c=[]
    for key in d_coeff_Pear_s1.keys():
        l_c.append((1+d_coeff_Pear_s1[key][0])**2)
        s_1=math.sqrt(sum(l_c)/len(l_c))
            
    # sfasamento 2 h
    d_coeff_Pear_s2={}
    for i in range(1,len(df_delta_fill.columns),2):
        T_col=df_delta_fill.columns[i-1]
        data_col=df_delta_fill.columns[i]
        
        data = df_delta_fill[data_col].values.tolist()
        data_s2=data[2:]
        T=df_delta_fill[T_col].values.tolist()
        T_s2=T[:(len(T)-2)]
        
        c_2=scipy.stats.pearsonr(T_s2, data_s2)

        d_coeff_Pear_s2[data_col]=c_2
        
    l_c=[]
    for key in d_coeff_Pear_s2.keys():
        l_c.append((1+d_coeff_Pear_s2[key][0])**2)
        s_2=math.sqrt(sum(l_c)/len(l_c))
    
    return(d_coeff_Pear_s1)
    
   
def graphDelta(df_delta,d_coeff_Pear):

    for i in range(1,len(df_delta.columns),2):
        col_data=df_delta.columns[i]
        col_temp=df_delta.columns[i-1]
        t=df_delta[col_data].index
        data = df_delta[col_data].values
        
        fig, ax1 = plt.subplots()
        color0 = 'tab:blue'
        ax1.set_xlabel('time')
        ax1.set_ylabel('displacement [mm]')
        ax1.plot(t, data, color=color0, label=col_data,linewidth=1)
        ax1.tick_params(axis='y',labelsize=10,labelrotation=0)
        ax1.tick_params(axis='x',labelsize=10,labelrotation=-90)
        ax1.set_ylim([-5,5])
        
        textstr='c_Pear='+str(round(d_coeff_Pear[col_data][0],2))
        
        ax1.text(0.11, 0.85,'%s'%(textstr),
                 transform=ax1.transAxes,fontsize=10,verticalalignment='top')
        
        plt.legend(loc='upper left', bbox_to_anchor=(0.1, 0.95))
        
        plt.grid(True, which='major',axis='both', linestyle='--',dashes=[10, 10], linewidth=0.5)

        data_temp = df_delta[col_temp].values
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        color2 = 'tab:grey'
        ax2.set_ylabel('delta T [°C]') 
        ax2.plot(t, data_temp, color=color2, label="TEMPERATURE", linewidth=1,alpha=0.4)
        ax2.tick_params(axis='y',labelsize=10,labelrotation=0)
        ax2.set_ylim([-20,20])
        
        fig.tight_layout()
        fig.set_size_inches((16,9))
        fig.canvas.set_window_title(col_data)
        
        fig.show()
        
        fname=col_data
        fig.savefig(fname, dpi=300, facecolor='w', edgecolor='w',
                    orientation='landscape', papertype=None, format=None,
                    transparent=False, bbox_inches='tight', pad_inches=None,
                    frameon=None, metadata=None)
        
def main():
    folder="./download"
    zero_datetime=datetime.datetime(2019, 8, 3, 0, 0)
    df_meas=readCsv(folder)
    s_zero=zeroRead(df_meas,zero_datetime)
    df_delta=deltaCalc(df_meas,s_zero,zero_datetime)
    d_coeff_Pear=PearsonCorr(df_delta)
    graphDelta(df_delta,d_coeff_Pear)
        
# call the main function
if __name__ == "__main__":
    main()