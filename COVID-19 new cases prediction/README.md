# Modeling COVID-19 new cases using twitter and google mobility trends  
 
## Introduction:  
Traditionally, public health is monitored by aggregating statistic obtained from healthcare providers. Such models are costly, slow, and maybe biased. Recently, several studies have been successfully used digital media and social network services like Google Flu Trends, Google search data, and Twitter data to reduce the latency and improve the overall effectiveness of public health monitoring for Influenza surveillance.    

Here, I applied machine learning techniques to study the relationships between global twitter trends, Google Maps mobility reports and number of daily COVID-19 new cases.

## Data description:  
**1.	Twitter data**     
Dataset of tweets is acquired from the Twitter Stream related to COVID-19 chatter from http://www.panacealab.org/covid19/. This dataset contains the top 1000 frequent terms, the top 1000 bigrams, and the top 1000 trigrams in all languages. Daily top 100 bigrams tweets in English were selected for this study. 
  
**2.	Google Maps mobility reports**    
Community Mobility Reports from Google Maps are created with aggregated, anonymized sets of data from users who have turned on the Location History setting. The dataset displays the percentage changes from baseline in visits to retail and recreation places, grocery and pharmacy shops, parks, transit stations, workplaces and residence location. Besides residence mobility, the remaining locations show the same trend of mobility change. In order to decrease the dimension of dataset, I selected transit station mobility as their representative together with the residential data for this study. This dataset is obtained from https://www.google.com/covid19/mobility/
  
**3.	COVID-19 new cases**    
The website “Our World in Data” collects the statistics on the coronavirus pandemic  for every country in the world. For this study, I focused on global daily new cases of COVID-19. All information is updated daily at https://ourworldindata.org/coronavirus. 
  

## Duration of the study:  
March 22nd to May 4th, 2020

## Project design:
**Part 1:** Create binary classification model to label the type of daily tweets 
**Part 2:** Create multiple linear regression model to observe the relationship between twitter trends, global mobility reports and number of COVID-19 new cases     
**Part 3:** Attempt to use long short-term memory networks to predict COVID-19 new cases in advance   
