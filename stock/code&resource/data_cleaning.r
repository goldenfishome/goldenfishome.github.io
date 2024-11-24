---
title: "Data Cleaning"
output: html_document
date: "2023-03-30"
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
```

```{r}
vaccine <- read.csv('data/vaccine.csv', skip = 2)[c('Date','Total.Doses.Administered.Daily')]
colnames(vaccine)[2] <- 'total_doses'
write.csv(vaccine, 'data/vaccine_clean.csv', row.names = FALSE)
```

```{r}
covid <- read.csv('data/weekly_covid.csv', skip = 2)[c('Date','Weekly.Cases')]

write.csv(covid, 'data/covid.csv', row.names = FALSE)
```

