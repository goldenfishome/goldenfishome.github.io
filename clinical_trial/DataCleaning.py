# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 00:19:05 2021

@author: yujia
"""

import pandas as pd
import re


################################################
## Read all raw data from csv files
# save the names of csv files into a list
all_filenames = [f"rawData{i}.csv" for i in range(1,9)] 

#read all files and merge them into one dataframe
combined_csv = pd.concat([pd.read_csv(f,skiprows=(10),na_values=' ') for f in all_filenames ])

#export merged dataframe to a csv file
combined_csv.to_csv( "combined_csv.csv", index=False, encoding='utf-8-sig')

# read the merged dataframe and save into a variable
myDF = pd.read_csv("combined_csv.csv")

# drop the row with StudyType == 'Observational'
interventionStudy= myDF["StudyType"] == 'Interventional'
myDF = myDF[interventionStudy]
#print(myDF.shape)

# save to csv file
myDF.to_csv( "myDF.csv", index=False, encoding='utf-8-sig')

# check the data type for each column
#myDF.dtypes

# rewrite the values in HealthyVolunteers as 'Accept'
myDF.loc[myDF['HealthyVolunteers']=='Accepts Healthy Volunteers','HealthyVolunteers'] = 'Accept'


################################################
## standarize the StartDate and CompletionDate columns 
# standarize the StartDate columns
myDF['StartDate'] = pd.to_datetime(myDF['StartDate'],errors='coerce')

# standarize the CompletionDate columns
myDF['CompletionDate'] = pd.to_datetime(myDF['CompletionDate'],errors='coerce')

# calculate the days between StartDate & CompletionDate,
# save the days as integer in new columns of DurationDays
myDF.insert(8,'DurationDays',(myDF['CompletionDate']-myDF['StartDate']).dt.days)

################################################
## Deduplicate the InterventionType and create new column ArmNumber
myDF['InterventionType'] = myDF['InterventionType'].str.split(pat='|')
myDF['ArmNumber'] = myDF['InterventionType'].str.len()

# deduplicate the InterventionType column
myDF['InterventionType'] = myDF['InterventionType'].apply(set)
myDF['InterventionType'] = myDF['InterventionType'].apply(list)

a = lambda x: ",".join(x)
myDF['InterventionType'] = myDF['InterventionType'].apply(a)

################################################
## Fill nan in Gender column by modes
myDF['Gender'].value_counts()
myDF['Gender']=myDF['Gender'].fillna("All")


################################################
## Fill nan in DesignPrimaryPurpose column based on crosstab with InterventionType
# DesignPrimaryPurpose column has 68 nan
myDF['DesignPrimaryPurpose'].isnull().sum()
# fill NA by finding the relationship to InterventionType
crossTabDF1=pd.crosstab(myDF['InterventionType'],myDF['DesignPrimaryPurpose'])

myDF["DesignPrimaryPurpose"][myDF['InterventionType']=='Biological']=myDF["DesignPrimaryPurpose"][myDF['InterventionType']=='Biological'].fillna("Prevention")
myDF["DesignPrimaryPurpose"][myDF['InterventionType']=='Behavioral']=myDF["DesignPrimaryPurpose"][myDF['InterventionType']=='Behavioral'].fillna("Prevention")
myDF["DesignPrimaryPurpose"][myDF['InterventionType']=='Biological,Procedure']=myDF["DesignPrimaryPurpose"][myDF['InterventionType']=='Biological,Procedure'].fillna("Prevention")
myDF["DesignPrimaryPurpose"]=myDF["DesignPrimaryPurpose"].fillna("Treatment")


################################################
## Standardize the MaximumAge and MinimumAge columns in new AgeGroups column
# fill NA as NoLimt in both MaximumAge and MinimumAge columns
myDF['MaximumAge']=myDF['MaximumAge'].fillna('100')
myDF['MinimumAge']=myDF['MinimumAge'].fillna('0')

# change string to number
## if have minutes, days, weeks, months == 0
myDF['MaximumAge']=myDF['MaximumAge'].str.replace(r'[0-9]*\sWeeks?',"0",regex=True)
myDF['MaximumAge']=myDF['MaximumAge'].str.replace(r'[0-9]*\sDays?',"0",regex=True)
myDF['MaximumAge']=myDF['MaximumAge'].str.replace(r'[0-9]*\sMonths?',"0",regex=True)
myDF['MaximumAge']=myDF['MaximumAge'].str.replace(r'[0-9]*\sHours?',"0",regex=True)
myDF['MaximumAge']=myDF['MaximumAge'].str.replace(r'[0-9]*\sMinutes?',"0",regex=True)


# drop the unit for MaximumAge column
myDF['MaximumAge']=myDF['MaximumAge'].str.replace(r'\D','')
myDF['MaximumAge']=pd.to_numeric(myDF['MaximumAge'])

myDF['MinimumAge']=myDF['MinimumAge'].str.replace(r'[0-9]*\sWeeks?',"0",regex=True)
myDF['MinimumAge']=myDF['MinimumAge'].str.replace(r'[0-9]*\sDays?',"0",regex=True)
myDF['MinimumAge']=myDF['MinimumAge'].str.replace(r'[0-9]*\sMonths?',"0",regex=True)
myDF['MinimumAge']=myDF['MinimumAge'].str.replace(r'[0-9]*\sHours?',"0",regex=True)
myDF['MinimumAge']=myDF['MinimumAge'].str.replace(r'[0-9]*\sMinutes?',"0",regex=True)

# drop the unit for MinimumAge column
myDF['MinimumAge']=myDF['MinimumAge'].str.replace(r'\D','')
myDF['MinimumAge']=pd.to_numeric(myDF['MinimumAge'])


# change number to category
myDF['MaximumAge'] = pd.cut(myDF['MaximumAge'], bins = [-1,5,17,64,151], labels = ['infants', 'childTeen', 'adults', 'elderly'])
myDF['MinimumAge'] = pd.cut(myDF['MinimumAge'], bins = [-1,5,17,64,151], labels = ['infants', 'childTeen', 'adults', 'elderly'])

myDF['AgeGroups'] = myDF['MaximumAge'].str.cat(myDF['MinimumAge'],sep="-")

myDF['AgeGroups']=myDF['AgeGroups'].str.replace('adults-adults',"adults",regex=False)
myDF['AgeGroups']=myDF['AgeGroups'].str.replace('infants-infants',"infants",regex=False)
myDF['AgeGroups']=myDF['AgeGroups'].str.replace('childTeen-childTeen',"childTeen",regex=False)
myDF['AgeGroups']=myDF['AgeGroups'].str.replace('elderly-elderly',"elderly",regex=False)
myDF['AgeGroups']=myDF['AgeGroups'].str.replace('elderly-infants',"allAges",regex=False)


################################################
# drop columns
myDF=myDF.drop(['MaximumAge','MinimumAge'],axis='columns')
myDF=myDF.drop(['Rank','Condition','StudyType','InterventionName'],axis='columns')


################################################
# deduplicate the LocationCountry column into set
myDF['LocationCountry'] = myDF['LocationCountry'].str.split(pat='|')
f = lambda x: set(x) if isinstance(x,list) else x
myDF['LocationCountry']=myDF['LocationCountry'].apply(f)
m = lambda x: list(x) if isinstance(x,set) else x
myDF['LocationCountry'] = myDF['LocationCountry'].apply(m)

a = lambda x: ",".join(x) if isinstance(x,list) else x
myDF['LocationCountry'] = myDF['LocationCountry'].apply(a)


# fill NA as SingleArmNA in DesignAllocation column
myDF['DesignAllocation']=myDF['DesignAllocation'].fillna('SingleArmNA')


# fill nan in DesignInterventionModel column based on crosstab with DesignInterventionModel
myDF['DesignInterventionModel']=myDF['DesignInterventionModel'].fillna("Parallel Assignment")
myDF['DesignInterventionModel'].isnull().sum()

################################################
# manage columns of EventGroupDeathsNumAffected and EventGroupSeriousNumAffected    
myDF['EventGroupDeathsNumAffected'] = myDF['EventGroupDeathsNumAffected'].str.split(pat='|')
b = lambda x: [int(i) for i in x] if isinstance(x,list) else x
myDF['EventGroupDeathsNumAffected'] = myDF['EventGroupDeathsNumAffected'].apply(b)
total = lambda x: sum(x) if isinstance(x,list) else x
myDF['EventGroupDeathsNumAffected'] = myDF['EventGroupDeathsNumAffected'].apply(total)

myDF['EventGroupSeriousNumAffected'] = myDF['EventGroupSeriousNumAffected'].str.split(pat='|')
myDF['EventGroupSeriousNumAffected'] = myDF['EventGroupSeriousNumAffected'].apply(b)
myDF['EventGroupSeriousNumAffected'] = myDF['EventGroupSeriousNumAffected'].apply(total)

#complete_trial = myDF[myDF['OverallStatus']=='Completed']

# export the cleaned dataframe to csv file
myDF.to_csv( "clearDF.csv", index=False, encoding='utf-8-sig')





