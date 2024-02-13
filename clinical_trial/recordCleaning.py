# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 14:45:08 2021

@author: yujia
"""

import pandas as pd

original_df=pd.read_csv("cleanerDF.csv")
df = original_df.drop(labels=["BriefSummary","Condition","InterventionType","LocationCountry"],axis=1)

df["AdverseEffectsorDeath"] = df["AdverseEffectsorDeath"]/df['EnrollmentCount']

df['DurationDays'] = pd.cut(df['DurationDays'], bins = [-1,366,1097,2190,7000], 
                            labels = ['shortLen', 'relatively_shortLen', 
                                      'medianLen', 'longLen'])

df['EnrollmentCount'] = pd.cut(df['EnrollmentCount'], bins = [0,101,301,1001,3001,90000], 
                            labels = ['smallEnrol', 'relatively_smallEnrol', 'median_Enrol', 
                                      'largeEnrol','very_largeEnrol'])

df['AdverseEffectsorDeath'] = pd.cut(df['AdverseEffectsorDeath'], 
                                     bins = [-1,0,0.0005,0.01,0.1,295], 
                            labels = ['noAD', 'lowAD', 'medianAD', 'highAD','very_highAD'])

df.to_csv("ARMCleaned.csv", index=False,encoding='utf-8-sig')
