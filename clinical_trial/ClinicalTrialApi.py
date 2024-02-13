# -*- coding: utf-8 -*-
"""
Created on Sat Sep 11 21:38:20 2021

@author: yujia
"""

import requests

# data values from 1-1000
url1 = "https://clinicaltrials.gov/api/query/study_fields?expr=bacterial+infection&fields=OverallStatus%2CBriefSummary%2CCondition%2CHealthyVolunteers%2CStudyType%2CStartDate%2CCompletionDate%2CInterventionName%2CInterventionType%2CPhase%2CGender%2CMaximumAge%2CMinimumAge%2CDesignPrimaryPurpose%2CDesignInterventionModel%2CLocationCountry%2CDesignAllocation%2CEnrollmentCount%2CEventGroupDeathsNumAffected%2CEventGroupSeriousNumAffected&min_rnk=1&max_rnk=1000&fmt=csv"

# data values from 1001-2000
url2 = "https://clinicaltrials.gov/api/query/study_fields?expr=bacterial+infection&fields=OverallStatus%2CBriefSummary%2CCondition%2CHealthyVolunteers%2CStudyType%2CStartDate%2CCompletionDate%2CInterventionName%2CInterventionType%2CPhase%2CGender%2CMaximumAge%2CMinimumAge%2CDesignPrimaryPurpose%2CDesignInterventionModel%2CLocationCountry%2CDesignAllocation%2CEnrollmentCount%2CEventGroupDeathsNumAffected%2CEventGroupSeriousNumAffected&min_rnk=1001&max_rnk=2000&fmt=csv"

# data values from 2001-3000
url3 = "https://clinicaltrials.gov/api/query/study_fields?expr=bacterial+infection&fields=OverallStatus%2CBriefSummary%2CCondition%2CHealthyVolunteers%2CStudyType%2CStartDate%2CCompletionDate%2CInterventionName%2CInterventionType%2CPhase%2CGender%2CMaximumAge%2CMinimumAge%2CDesignPrimaryPurpose%2CDesignInterventionModel%2CLocationCountry%2CDesignAllocation%2CEnrollmentCount%2CEventGroupDeathsNumAffected%2CEventGroupSeriousNumAffected&min_rnk=2001&max_rnk=3000&fmt=csv"

# data values from 3001-4000
url4 = "https://clinicaltrials.gov/api/query/study_fields?expr=bacterial+infection&fields=OverallStatus%2CBriefSummary%2CCondition%2CHealthyVolunteers%2CStudyType%2CStartDate%2CCompletionDate%2CInterventionName%2CInterventionType%2CPhase%2CGender%2CMaximumAge%2CMinimumAge%2CDesignPrimaryPurpose%2CDesignInterventionModel%2CLocationCountry%2CDesignAllocation%2CEnrollmentCount%2CEventGroupDeathsNumAffected%2CEventGroupSeriousNumAffected&min_rnk=3001&max_rnk=4000&fmt=csv"

# data values from 4001-5000
url5 = "https://clinicaltrials.gov/api/query/study_fields?expr=bacterial+infection&fields=OverallStatus%2CBriefSummary%2CCondition%2CHealthyVolunteers%2CStudyType%2CStartDate%2CCompletionDate%2CInterventionName%2CInterventionType%2CPhase%2CGender%2CMaximumAge%2CMinimumAge%2CDesignPrimaryPurpose%2CDesignInterventionModel%2CLocationCountry%2CDesignAllocation%2CEnrollmentCount%2CEventGroupDeathsNumAffected%2CEventGroupSeriousNumAffected&min_rnk=4001&max_rnk=5000&fmt=csv"

# data values from 5001-6000
url6 = "https://clinicaltrials.gov/api/query/study_fields?expr=bacterial+infection&fields=OverallStatus%2CBriefSummary%2CCondition%2CHealthyVolunteers%2CStudyType%2CStartDate%2CCompletionDate%2CInterventionName%2CInterventionType%2CPhase%2CGender%2CMaximumAge%2CMinimumAge%2CDesignPrimaryPurpose%2CDesignInterventionModel%2CLocationCountry%2CDesignAllocation%2CEnrollmentCount%2CEventGroupDeathsNumAffected%2CEventGroupSeriousNumAffected&min_rnk=5001&max_rnk=6000&fmt=csv"

# data values from 6000-7000
url7 = "https://clinicaltrials.gov/api/query/study_fields?expr=bacterial+infection&fields=OverallStatus%2CBriefSummary%2CCondition%2CHealthyVolunteers%2CStudyType%2CStartDate%2CCompletionDate%2CInterventionName%2CInterventionType%2CPhase%2CGender%2CMaximumAge%2CMinimumAge%2CDesignPrimaryPurpose%2CDesignInterventionModel%2CLocationCountry%2CDesignAllocation%2CEnrollmentCount%2CEventGroupDeathsNumAffected%2CEventGroupSeriousNumAffected&min_rnk=6001&max_rnk=7000&fmt=csv"

# data values from 7000-7688
url8 = "https://clinicaltrials.gov/api/query/study_fields?expr=bacterial+infection&fields=OverallStatus%2CBriefSummary%2CCondition%2CHealthyVolunteers%2CStudyType%2CStartDate%2CCompletionDate%2CInterventionName%2CInterventionType%2CPhase%2CGender%2CMaximumAge%2CMinimumAge%2CDesignPrimaryPurpose%2CDesignInterventionModel%2CLocationCountry%2CDesignAllocation%2CEnrollmentCount%2CEventGroupDeathsNumAffected%2CEventGroupSeriousNumAffected&min_rnk=7000&max_rnk=7688&fmt=csv"

urls = [url1,url2,url3,url4,url5,url6,url7,url8]

file_number = 1
for url in urls:
    """store each API data into sepate files"""
    file_name = f"rawData{file_number}.csv"
    file_number += 1
    file_object = open(file_name,'w',encoding='utf-8') # create empty file
    response = requests.get(url) # get all info from url, store all contents get from page
    file_object.write(response.text) # store the data onto file 
    file_object.close()
    
    



