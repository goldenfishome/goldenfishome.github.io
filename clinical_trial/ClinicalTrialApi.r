library("httr")
library("jsonlite")

# data values from 1-1000
url1 <- "https://clinicaltrials.gov/api/query/study_fields?expr=bacterial+infection&fields=NCTId%2CBriefTitle%2CCondition%2CHealthyVolunteers%2CStudyType%2CStartDate%2CCompletionDate%2CInterventionName%2CInterventionType%2CPhase%2CGender%2CMaximumAge%2CMinimumAge%2CDesignPrimaryPurpose%2CDesignInterventionModel%2CLocationCountry%2CDesignAllocation%2CEnrollmentCount%2CEventGroupDeathsNumAffected%2CEventGroupSeriousNumAffected&min_rnk=1&max_rnk=1000&fmt=csv"

# data values from 1001-2000
url2 <- "https://clinicaltrials.gov/api/query/study_fields?expr=bacterial+infection&fields=NCTId%2CBriefTitle%2CCondition%2CHealthyVolunteers%2CStudyType%2CStartDate%2CCompletionDate%2CInterventionName%2CInterventionType%2CPhase%2CGender%2CMaximumAge%2CMinimumAge%2CDesignPrimaryPurpose%2CDesignInterventionModel%2CLocationCountry%2CDesignAllocation%2CEnrollmentCount%2CEventGroupDeathsNumAffected%2CEventGroupSeriousNumAffected&min_rnk=1001&max_rnk=2000&fmt=csv"

# data values from 2001-3000
url3 <- "https://clinicaltrials.gov/api/query/study_fields?expr=bacterial+infection&fields=NCTId%2CBriefTitle%2CCondition%2CHealthyVolunteers%2CStudyType%2CStartDate%2CCompletionDate%2CInterventionName%2CInterventionType%2CPhase%2CGender%2CMaximumAge%2CMinimumAge%2CDesignPrimaryPurpose%2CDesignInterventionModel%2CLocationCountry%2CDesignAllocation%2CEnrollmentCount%2CEventGroupDeathsNumAffected%2CEventGroupSeriousNumAffected&min_rnk=2001&max_rnk=3000&fmt=csv"

# data values from 3001-4000
url4 <- "https://clinicaltrials.gov/api/query/study_fields?expr=bacterial+infection&fields=NCTId%2CBriefTitle%2CCondition%2CHealthyVolunteers%2CStudyType%2CStartDate%2CCompletionDate%2CInterventionName%2CInterventionType%2CPhase%2CGender%2CMaximumAge%2CMinimumAge%2CDesignPrimaryPurpose%2CDesignInterventionModel%2CLocationCountry%2CDesignAllocation%2CEnrollmentCount%2CEventGroupDeathsNumAffected%2CEventGroupSeriousNumAffected&min_rnk=3001&max_rnk=4000&fmt=csv"

# data values from 4001-5000
url5 <- "https://clinicaltrials.gov/api/query/study_fields?expr=bacterial+infection&fields=NCTId%2CBriefTitle%2CCondition%2CHealthyVolunteers%2CStudyType%2CStartDate%2CCompletionDate%2CInterventionName%2CInterventionType%2CPhase%2CGender%2CMaximumAge%2CMinimumAge%2CDesignPrimaryPurpose%2CDesignInterventionModel%2CLocationCountry%2CDesignAllocation%2CEnrollmentCount%2CEventGroupDeathsNumAffected%2CEventGroupSeriousNumAffected&min_rnk=4001&max_rnk=5000&fmt=csv"

# data values from 5001-6000
url6 <- "https://clinicaltrials.gov/api/query/study_fields?expr=bacterial+infection&fields=NCTId%2CBriefTitle%2CCondition%2CHealthyVolunteers%2CStudyType%2CStartDate%2CCompletionDate%2CInterventionName%2CInterventionType%2CPhase%2CGender%2CMaximumAge%2CMinimumAge%2CDesignPrimaryPurpose%2CDesignInterventionModel%2CLocationCountry%2CDesignAllocation%2CEnrollmentCount%2CEventGroupDeathsNumAffected%2CEventGroupSeriousNumAffected&min_rnk=5001&max_rnk=6000&fmt=csv"

# data values from 6000-6455
url7 <- "https://clinicaltrials.gov/api/query/study_fields?expr=bacterial+infection&fields=NCTId%2CBriefTitle%2CCondition%2CHealthyVolunteers%2CStudyType%2CStartDate%2CCompletionDate%2CInterventionName%2CInterventionType%2CPhase%2CGender%2CMaximumAge%2CMinimumAge%2CDesignPrimaryPurpose%2CDesignInterventionModel%2CLocationCountry%2CDesignAllocation%2CEnrollmentCount%2CEventGroupDeathsNumAffected%2CEventGroupSeriousNumAffected&min_rnk=6001&max_rnk=6455&fmt=csv"

urls <- list(url1,url2,url3,url4,url5,url6,url7)

file_num <- 1
for (url in urls) {
  ClinicalTrialAPI_Call<-httr::GET(url)
  MYDF<-httr::content(ClinicalTrialAPI_Call) ## Print to a file
  FileName <- paste("trialdata",file_num,".csv",sep = "")
  file_num <- file_num+1
  ClinicalFile <- file(FileName)
  #write.csv(MYDF,ClinicalFile)## Write data to file
}


