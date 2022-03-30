#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 08:24:55 2022

@author: emmastanley
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import pickle

#########################
#   Define Functions    #
#########################

#functions to determine classification errors by demographic attribute
#TCR = true classification rate, fraction of males and females correctly identified
#TMR = true male classification rate, fraction of males correctly identified
#TFR = true female classification rate, fraction of females correctly identified

#true classificaition rate - males and females 
def get_TCR(df, label):
    label_df = df.loc[df[label] == 1] #slice the dataframe to only contain attribute of interest
    TP = (label_df.TP.values == 1).sum() #number of TP 
    TN = (label_df.TN.values == 1).sum() #number of TN
    
    TC = TP + TN
    
    TCR = TC/len(label_df) #TCR for all subjects with label

    return TCR


#true classification rate by sex- TMR and TFR
def get_TSR(df, label):
    
    label_df = df.loc[df[label] == 1] #slice the dataframe to only conain attribute of interest
    
    TM = (label_df.TP.values == 1).sum() #number of corr. classified males
    TF = (label_df.TN.values == 1).sum() #number of corr. classified females
    
    males = len(label_df.loc[label_df['M'] == 1]) #number of total males
    females = len(label_df.loc[label_df['F'] == 1]) #number of total females
    
    TMR = TM/males
    TFR = TF/females


    return TMR, TFR


#true classification rate by intersection of two labels (eg. label1 = race, label2 = class)
def get_intersection_TCR(df, label1, label2):
    label1_df = df.loc[df[label1] == 1] #slice the dataframe to only contain first attribute of interest
    label2_df = label1_df.loc[label1_df[label2] == 1]
    
    TP = (label2_df.TP.values == 1).sum() #number of TP 
    TN = (label2_df.TN.values == 1).sum() #number of TN
    
    TC = TP + TN
    
    TCR = TC/len(label2_df) #TCR for all subjects with labels

    return TCR


#true classification rate sex with intersection of two labels (eg. label1 = race, label2 = class)
def get_intersection_TSR(df, label1, label2):
    
    label1_df = df.loc[df[label1] == 1] #slice the dataframe to only contain first attribute of interest
    label2_df = label1_df.loc[label1_df[label2] == 1]
    
    TM = (label2_df.TP.values == 1).sum() #number of corr. classified males
    TF = (label2_df.TN.values == 1).sum() #number of corr. classified females
    
    males = len(label2_df.loc[label2_df['M'] == 1]) #number of total males
    females = len(label2_df.loc[label2_df['F'] == 1]) #number of total females
    
    TMR = TM/males
    TFR = TF/females

    return TMR, TFR


def getFoldFairness(df, fold_no):
    
    col_name = 'preds_fold' + str(fold_no) #col with predictions for that fold
    df = df[~df[col_name].isnull()] #df with values for that fold 
    
    #create one-hot encoded columns for TP, TN, FP, FN
    #for sex classification M => positive , F => negative 
    df['TP'] = df.apply(lambda row: 1 if ((row['M'] == 1) & (row[col_name]==1)) else 0, axis=1)
    df['TN'] = df.apply(lambda row: 1 if ((row['M']== 0) & (row[col_name] ==0)) else 0, axis=1)
    df['FP'] = df.apply(lambda row: 1 if ((row['M'] == 0) & (row[col_name] ==1)) else 0, axis=1)
    df['FN'] = df.apply(lambda row: 1 if ((row['M'] == 1) & (row[col_name] ==0)) else 0, axis=1)
        
    
    TCR_white = get_TCR(df, 'white_only')
    TCR_black = get_TCR(df, 'black_only')
    TCR_uc = get_TCR(df, 'upper_class')
    TCR_mc = get_TCR(df, 'middle_class')
    TCR_lc = get_TCR(df, 'lower_class')

    TCR_white_uc = get_intersection_TCR(df, 'white_only', 'upper_class')
    TCR_white_mc = get_intersection_TCR(df, 'white_only', 'middle_class')
    TCR_white_lc = get_intersection_TCR(df, 'white_only', 'lower_class')
    TCR_black_uc = get_intersection_TCR(df, 'black_only', 'upper_class')
    TCR_black_mc = get_intersection_TCR(df, 'black_only', 'middle_class')
    TCR_black_lc = get_intersection_TCR(df, 'black_only', 'lower_class')
    
    
    TMR_white, TFR_white = get_TSR(df, 'white_only')
    TMR_black, TFR_black = get_TSR(df, 'black_only')
    TMR_uc, TFR_uc = get_TSR(df, 'upper_class')
    TMR_mc, TFR_mc = get_TSR(df, 'middle_class')
    TMR_lc, TFR_lc = get_TSR(df, 'lower_class')
    
    TMR_white_uc, TFR_white_uc = get_intersection_TSR(df, 'white_only', 'upper_class')
    TMR_white_mc, TFR_white_mc = get_intersection_TSR(df, 'white_only', 'middle_class')
    TMR_white_lc, TFR_white_lc = get_intersection_TSR(df, 'white_only', 'lower_class')
    
    TMR_black_uc, TFR_black_uc = get_intersection_TSR(df, 'black_only', 'upper_class')
    TMR_black_mc, TFR_black_mc = get_intersection_TSR(df, 'black_only', 'middle_class')
    TMR_black_lc, TFR_black_lc = get_intersection_TSR(df, 'black_only', 'lower_class')
    
    
    #compile list of metrics 
    TCR_list = [TCR_white, TCR_black, TCR_uc, TCR_mc, TCR_lc]
    TMR_list = [TMR_white, TMR_black, TMR_uc, TMR_mc, TMR_lc]
    TFR_list = [TFR_white, TFR_black, TFR_uc, TFR_mc, TFR_lc]
    
    #compile list of metrics for intersections
    TCR_ix_list = [TCR_white_uc, TCR_white_mc, TCR_white_lc, TCR_black_uc, TCR_black_mc, TCR_black_lc]
    TMR_ix_list = [TMR_white_uc, TMR_white_mc, TMR_white_lc, TMR_black_uc, TMR_black_mc, TMR_black_lc]
    TFR_ix_list = [TFR_white_uc, TFR_white_mc, TFR_white_lc, TFR_black_uc, TFR_black_mc, TFR_black_lc]
    
    return df, TCR_list, TMR_list, TFR_list, TCR_ix_list, TMR_ix_list, TFR_ix_list



def plots(df, label):
    
    #plot true classification rate for race
    sns.stripplot(x=label, y="TCR", data=df, size=4, palette='hls', linewidth=0, dodge=True,  alpha=0.5)
    sns.pointplot(x=label, y="TCR", data=df, size=4, palette='hls', linewidth=0, join=False, dodge=0.5,
                          capsize=.2, ci='sd', errwidth=1, markers='x')
    plt.grid()
    plt.title('True Classification Rate', fontsize=14)
    plt.ylim([0.4, 1.2])
    plt.xticks(rotation=45)
    plt.show()
    # plt.savefig('/Users/emmastanley/Documents/BME/Research/ABCD_sexclassification_2022/exp5b/TCR.png', dpi=300)
    # plt.close()
    
    
    #plot true male classification rate for race
    sns.stripplot(x=label, y="TMR", data=df, size=4, palette='hls', linewidth=0, dodge=True,  alpha=0.5)
    sns.pointplot(x=label, y="TMR", data=df, size=4, palette='hls', linewidth=0, join=False, dodge=0.5,
                          capsize=.2, ci='sd', errwidth=1, markers='x')
    plt.grid()
    plt.title('True Male Classification Rate', fontsize=14)
    plt.ylim([0.4, 1.2])
    plt.xticks(rotation=45)
    plt.show()
    # plt.savefig('/Users/emmastanley/Documents/BME/Research/ABCD_sexclassification_2022/exp5b/TMR.png', dpi=300)
    # plt.close()
    
    #plot true female classification rate for race
    sns.stripplot(x=label, y="TFR", data=df, size=4, palette='hls', linewidth=0, dodge=True,  alpha=0.5)
    sns.pointplot(x=label, y="TFR", data=df, size=4, palette='hls', linewidth=0, join=False, dodge=0.5,
                          capsize=.2, ci='sd', errwidth=1, markers='x')
    plt.grid()
    plt.title('True Female Classification Rate',  fontsize=14)
    plt.ylim([0.4, 1.2])
    plt.xticks(rotation=45)
    plt.show()
    # plt.savefig('/Users/emmastanley/Documents/BME/Research/ABCD_sexclassification_2022/exp5b/TFR.png', dpi=300)
    # plt.close()
    return 


def mean_std(df, label, group, metric):
    data = df.loc[df[label]==group]
    mean = data[metric].mean()
    std = data[metric].std()
    print('{}: {}, Metric: {}\nMean = {:0.3f}, Std = {:0.4f}\n '.format(label, group, metric, mean, std))
    return

#%% load dataframes from each fold and merge prediction columns

num_folds = 5
exp_title = 'exp12b'


save_dir = '/Users/emmastanley/Documents/BME/Research/ABCD_sexclassification_2022/' + exp_title + '/'



data_1 = pd.read_csv(save_dir + exp_title + '_fold1_df.csv', index_col = 0)
data_2 = pd.read_csv(save_dir + exp_title + '_fold2_df.csv', index_col = 0)
data_3 = pd.read_csv(save_dir + exp_title + '_fold3_df.csv', index_col = 0)
data_4 = pd.read_csv(save_dir + exp_title + '_fold4_df.csv', index_col = 0)
data_5 = pd.read_csv(save_dir + exp_title + '_fold5_df.csv', index_col = 0)



#merge all predictions columns in one dataframe

df_demo = data_1
df_demo['preds_fold1'] = df_demo['preds_fold1'].str[1].astype(float) #get rid of brackets + convert to numeric

df_demo['preds_fold2'] = data_2['preds_fold2'].str[1].astype(float) #get rid of brackets + convert to numeric
df_demo['preds_fold3'] = data_3['preds_fold3'].str[1].astype(float) #get rid of brackets + convert to numeric
df_demo['preds_fold4'] = data_4['preds_fold4'].str[1].astype(float) #get rid of brackets + convert to numeric
df_demo['preds_fold5'] = data_5['preds_fold5'].str[1].astype(float) #get rid of brackets + convert to numeric




#%%
#######################################
#   Compile Fairness Data & Analyze   #
#######################################

#get data by fold
df_1, TCR_list_1, TMR_list_1, TFR_list_1, TCR_ix_list_1, TMR_ix_list_1, TFR_ix_list_1 = getFoldFairness(df_demo, 1)
df_2, TCR_list_2, TMR_list_2, TFR_list_2, TCR_ix_list_2, TMR_ix_list_2, TFR_ix_list_2 = getFoldFairness(df_demo, 2)
df_3, TCR_list_3, TMR_list_3, TFR_list_3, TCR_ix_list_3, TMR_ix_list_3, TFR_ix_list_3 = getFoldFairness(df_demo, 3)
df_4, TCR_list_4, TMR_list_4, TFR_list_4, TCR_ix_list_4, TMR_ix_list_4, TFR_ix_list_4 = getFoldFairness(df_demo, 4)
df_5, TCR_list_5, TMR_list_5, TFR_list_5, TCR_ix_list_5, TMR_ix_list_5, TFR_ix_list_5 = getFoldFairness(df_demo, 5)


#get lists for TCR per fold
TCR_white_all = [item[0] for item in [TCR_list_1, TCR_list_2, TCR_list_3, TCR_list_4, TCR_list_5]]
TCR_black_all = [item[1] for item in [TCR_list_1, TCR_list_2, TCR_list_3, TCR_list_4, TCR_list_5]]
TCR_uc_all = [item[2] for item in [TCR_list_1, TCR_list_2, TCR_list_3, TCR_list_4, TCR_list_5]]
TCR_mc_all = [item[3] for item in [TCR_list_1, TCR_list_2, TCR_list_3, TCR_list_4, TCR_list_5]]
TCR_lc_all = [item[4] for item in [TCR_list_1, TCR_list_2, TCR_list_3, TCR_list_4, TCR_list_5]]

TCR_white_uc_all = [item[0] for item in [TCR_ix_list_1, TCR_ix_list_2, TCR_ix_list_3, TCR_ix_list_4, TCR_ix_list_5]]
TCR_white_mc_all = [item[1] for item in [TCR_ix_list_1, TCR_ix_list_2, TCR_ix_list_3, TCR_ix_list_4, TCR_ix_list_5]]
TCR_white_lc_all = [item[2] for item in [TCR_ix_list_1, TCR_ix_list_2, TCR_ix_list_3, TCR_ix_list_4, TCR_ix_list_5]]
TCR_black_uc_all = [item[3] for item in [TCR_ix_list_1, TCR_ix_list_2, TCR_ix_list_3, TCR_ix_list_4, TCR_ix_list_5]]
TCR_black_mc_all = [item[4] for item in [TCR_ix_list_1, TCR_ix_list_2, TCR_ix_list_3, TCR_ix_list_4, TCR_ix_list_5]]
TCR_black_lc_all = [item[5] for item in [TCR_ix_list_1, TCR_ix_list_2, TCR_ix_list_3, TCR_ix_list_4, TCR_ix_list_5]]


#get lists for TMR per fold
TMR_white_all = [item[0] for item in [TMR_list_1, TMR_list_2, TMR_list_3, TMR_list_4, TMR_list_5]]
TMR_black_all = [item[1] for item in [TMR_list_1, TMR_list_2, TMR_list_3, TMR_list_4, TMR_list_5]]
TMR_uc_all = [item[2] for item in [TMR_list_1, TMR_list_2, TMR_list_3, TMR_list_4, TMR_list_5]]
TMR_mc_all = [item[3] for item in [TMR_list_1, TMR_list_2, TMR_list_3, TMR_list_4, TMR_list_5]]
TMR_lc_all = [item[4] for item in [TMR_list_1, TMR_list_2, TMR_list_3, TMR_list_4, TMR_list_5]]

TMR_white_uc_all = [item[0] for item in [TMR_ix_list_1, TMR_ix_list_2, TMR_ix_list_3, TMR_ix_list_4, TMR_ix_list_5]]
TMR_white_mc_all = [item[1] for item in [TMR_ix_list_1, TMR_ix_list_2, TMR_ix_list_3, TMR_ix_list_4, TMR_ix_list_5]]
TMR_white_lc_all = [item[2] for item in [TMR_ix_list_1, TMR_ix_list_2, TMR_ix_list_3, TMR_ix_list_4, TMR_ix_list_5]]
TMR_black_uc_all = [item[3] for item in [TMR_ix_list_1, TMR_ix_list_2, TMR_ix_list_3, TMR_ix_list_4, TMR_ix_list_5]]
TMR_black_mc_all = [item[4] for item in [TMR_ix_list_1, TMR_ix_list_2, TMR_ix_list_3, TMR_ix_list_4, TMR_ix_list_5]]
TMR_black_lc_all = [item[5] for item in [TMR_ix_list_1, TMR_ix_list_2, TMR_ix_list_3, TMR_ix_list_4, TMR_ix_list_5]]



#get lists for TFR per fold
TFR_white_all = [item[0] for item in [TFR_list_1, TFR_list_2, TFR_list_3, TFR_list_4, TFR_list_5]]
TFR_black_all = [item[1] for item in [TFR_list_1, TFR_list_2, TFR_list_3, TFR_list_4, TFR_list_5]]
TFR_uc_all = [item[2] for item in [TFR_list_1, TFR_list_2, TFR_list_3, TFR_list_4, TFR_list_5]]
TFR_mc_all = [item[3] for item in [TFR_list_1, TFR_list_2, TFR_list_3, TFR_list_4, TFR_list_5]]
TFR_lc_all = [item[4] for item in [TFR_list_1, TFR_list_2, TFR_list_3, TFR_list_4, TFR_list_5]]



TFR_white_uc_all = [item[0] for item in [TFR_ix_list_1, TFR_ix_list_2, TFR_ix_list_3, TFR_ix_list_4, TFR_ix_list_5]]
TFR_white_mc_all = [item[1] for item in [TFR_ix_list_1, TFR_ix_list_2, TFR_ix_list_3, TFR_ix_list_4, TFR_ix_list_5]]
TFR_white_lc_all = [item[2] for item in [TFR_ix_list_1, TFR_ix_list_2, TFR_ix_list_3, TFR_ix_list_4, TFR_ix_list_5]]
TFR_black_uc_all = [item[3] for item in [TFR_ix_list_1, TFR_ix_list_2, TFR_ix_list_3, TFR_ix_list_4, TFR_ix_list_5]]
TFR_black_mc_all = [item[4] for item in [TFR_ix_list_1, TFR_ix_list_2, TFR_ix_list_3, TFR_ix_list_4, TFR_ix_list_5]]
TFR_black_lc_all = [item[5] for item in [TFR_ix_list_1, TFR_ix_list_2, TFR_ix_list_3, TFR_ix_list_4, TFR_ix_list_5]]




folds = [i for i in range(num_folds)]
race_list = ['white','black']
class_list =  ['upper_class', 'middle_class', 'lower_class']
ix_list = ['white_uc', 'white_mc', 'white_lc', 'black_uc', 'black_mc', 'black_lc']


#dataframe for comparing just race subgroups
idx_r = pd.MultiIndex.from_product([race_list, folds],
                                 names=['Race', 'Fold'])

race_df = pd.DataFrame('-', idx_r, ['TCR', 'TMR', 'TFR'])
race_df['TCR'] = TCR_white_all + TCR_black_all
race_df['TMR'] = TMR_white_all + TMR_black_all
race_df['TFR'] = TFR_white_all + TFR_black_all
race_df = race_df.reset_index()

plots(race_df, 'Race')



#dataframe for comparing just class subgroups
idx_c = pd.MultiIndex.from_product([class_list, folds],
                                 names=['Class', 'Fold'])

class_df = pd.DataFrame('-', idx_c, ['TCR', 'TMR', 'TFR'])
class_df['TCR'] = TCR_uc_all + TCR_mc_all + TCR_lc_all 
class_df['TMR'] = TMR_uc_all + TMR_mc_all + TMR_lc_all 
class_df['TFR'] = TFR_uc_all + TFR_mc_all + TFR_lc_all 
class_df = class_df.reset_index()

plots(class_df, 'Class')


#dataframe for comparing just intersection subgroups
idx_ix = pd.MultiIndex.from_product([ix_list, folds],
                                 names=['Race x Class', 'Fold'])

ix_df = pd.DataFrame('-', idx_ix, ['TCR', 'TMR', 'TFR'])
ix_df['TCR'] = TCR_white_uc_all + TCR_white_mc_all + TCR_white_lc_all + TCR_black_uc_all + TCR_black_mc_all + TCR_black_lc_all
ix_df['TMR'] = TMR_white_uc_all + TMR_white_mc_all + TMR_white_lc_all + TMR_black_uc_all + TMR_black_mc_all + TMR_black_lc_all
ix_df['TFR'] = TFR_white_uc_all + TFR_white_mc_all + TFR_white_lc_all + TFR_black_uc_all + TFR_black_mc_all + TFR_black_lc_all
ix_df = ix_df.reset_index()

plots(ix_df, 'Race x Class')





df_demo.to_csv(save_dir + exp_title + '_all_folds.csv')
race_df.to_csv(save_dir + exp_title + '_race_outcomes.csv')
class_df.to_csv(save_dir + exp_title + '_class_outcomes.csv')
ix_df.to_csv(save_dir + exp_title + '_race_class_outcomes.csv')
