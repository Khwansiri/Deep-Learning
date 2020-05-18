# Data wrangling: 45 days (2203 - 0405)
# 1. Global daily covid19 new cases
# 2. Global daily mobility in transit station and resident
# 3. Global daily tweets


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sb

from wordcloud import WordCloud
#---------------------------------------------------------------------------------------------#

# Data wrangling: Covid19
covid_raw = pd.read_csv("E:\DataScience\DSTI\Artificial Neural Networks\Project\Data\Covid\covid_updated_2203to1105.csv")
covid = covid_raw.iloc[:44,:]
date = covid.loc[:,"Date"]
total_case = covid.iloc[:,1]
new_case = covid.iloc[:,2]
total_death = covid.iloc[:,3]
new_death = covid.iloc[:,4]

# Data frame
df_covid_raw = pd.DataFrame(np.array([date, total_case, new_case, total_death, new_death]))            
df_covid = df_covid_raw.T
df_covid.rename(columns={0:"Date", 1:"Total Cases", 2:"New Cases", 3:"Total Deaths", 4: "New Deaths"}, inplace = True)


# Subplot to see big picture
fig, axs  = plt.subplots(2,2)
axs[0,0].plot(total_case)
axs[0,0].set_title("Covid-19 Total Cases")
axs[0,1].plot(new_case)
axs[0,1].set_title("Covid-19 New Cases")
axs[1,0].plot(total_death)
axs[1,0].set_title("Covid-19 Total Death")
axs[1,1].plot(new_death)
axs[1,1].set_title("Covid-19 New Death")


# New Cases plot
plt.plot(date,new_case)
plt.ylabel("Cases", fontsize = 30)
plt.title("Covid-19 New Cases", weight='heavy', fontsize = 30)
plt.xticks(rotation = 45)
plt.show()


#---------------------------------------------------------------------------------------------#

# Data wrangling: Mobility
mobility_raw = pd.read_csv("E:\DataScience\DSTI\Artificial Neural Networks\Project\Data\Mobility\Global_Mobility_updated_45days.csv")
mobility = mobility_raw.iloc[:44,:]
mob_transit = mobility.iloc[:,1]
mob_resident = mobility.iloc[:,2]


# Data frame
df_mobility_raw = pd.DataFrame(np.array([mob_transit, mob_resident]))            
df_mobility = df_mobility_raw.T
df_mobility.rename(columns={0:"Transit Station Mobility",1:"Resident Mobility"}, inplace = True)


# Subplot to see big picture
fig, axs  = plt.subplots(2)
axs[0].plot(mob_transit)
axs[0].set_title("Mobility: Transit Station")
axs[1].plot(mob_resident)
axs[1].set_title("Mobility:Resident")



# With date + share x axis
fig=plt.figure()
ax = fig.add_subplot(111)    # The big subplot
ax1 = plt.subplot(211)
ax2 = plt.subplot(212)

# Turn off axis lines and ticks of the big subplot
ax.spines['top'].set_color('none')
ax.spines['bottom'].set_color('none')
ax.spines['left'].set_color('none')
ax.spines['right'].set_color('none')
ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)

# Set common labels
#fig.text(0.06, 0.5, 'Percent change from baseline', ha='center', va='center', rotation='vertical')
fig.text(1, 2, 'Percent change from baseline', ha='center', va='center', rotation='vertical')

ax1.plot(date,mob_transit)
ax2.plot(date,mob_resident)

ax1.set_title("Transit Station Mobility", weight='heavy', fontsize = 30)
ax2.set_title("Resident Mobility", weight='heavy', fontsize = 30)

ax1.get_shared_x_axes().join(ax1, ax2)
ax1.set_xticklabels([])

plt.xticks(rotation=45)
plt.show()




#---------------------------------------------------------------------------------------------#

# Data wrangling: Twitter
bi_2203_raw = pd.read_excel("E:\DataScience\DSTI\Artificial Neural Networks\Project\Data\Tweets_Ref_100Labeled_ANN\Bi_2203_ANN.xlsx",
							names=["Text","Count","Label"])         
bi_2203 = bi_2203_raw.iloc[:100,:]  
other_bi_2203 = bi_2203.loc[bi_2203["Label"] == 0,"Count"].sum()
covid_bi_2203 = bi_2203.loc[bi_2203["Label"] != 0,"Count"].sum()

    
bi_2303_raw = pd.read_excel("E:\DataScience\DSTI\Artificial Neural Networks\Project\Data\Tweets_Ref_100Labeled_ANN\Bi_2303_ANN.xlsx", 
                            names=["Text","Count","Label"])        
bi_2303 = bi_2303_raw.iloc[:100,:] 
other_bi_2303 = bi_2303.loc[bi_2303["Label"] == 0,"Count"].sum()
covid_bi_2303 = bi_2303.loc[bi_2303["Label"] != 0,"Count"].sum()


bi_2403_raw = pd.read_excel("E:\DataScience\DSTI\Artificial Neural Networks\Project\Data\Tweets_Ref_100Labeled_ANN\Bi_2403_ANN.xlsx", 
                            names=["Text","Count","Label"])        
bi_2403 = bi_2403_raw.iloc[:100,:] 
other_bi_2403 = bi_2403.loc[bi_2403["Label"] == 0,"Count"].sum()
covid_bi_2403 = bi_2403.loc[bi_2403["Label"] != 0,"Count"].sum()


bi_2503_raw = pd.read_excel("E:\DataScience\DSTI\Artificial Neural Networks\Project\Data\Tweets_Ref_100Labeled_ANN\Bi_2503_ANN.xlsx", 
                            names=["Text","Count","Label"])        
bi_2503 = bi_2503_raw.iloc[:100,:] 
other_bi_2503 = bi_2503.loc[bi_2503["Label"] == 0,"Count"].sum()
covid_bi_2503 = bi_2503.loc[bi_2503["Label"] != 0,"Count"].sum()


bi_2603_raw = pd.read_excel("E:\DataScience\DSTI\Artificial Neural Networks\Project\Data\Tweets_Ref_100Labeled_ANN\Bi_2603_ANN.xlsx", 
                            names=["Text","Count","Label"])        
bi_2603 = bi_2603_raw.iloc[:100,:] 
other_bi_2603 = bi_2603.loc[bi_2603["Label"] == 0,"Count"].sum()
covid_bi_2603 = bi_2603.loc[bi_2603["Label"] != 0,"Count"].sum()


bi_2703_raw = pd.read_excel("E:\DataScience\DSTI\Artificial Neural Networks\Project\Data\Tweets_Ref_100Labeled_ANN\Bi_2703_ANN.xlsx", 
                            names=["Text","Count","Label"])        
bi_2703 = bi_2703_raw.iloc[:100,:] 
other_bi_2703 = bi_2703.loc[bi_2703["Label"] == 0,"Count"].sum()
covid_bi_2703 = bi_2703.loc[bi_2703["Label"] != 0,"Count"].sum()


bi_2803_raw = pd.read_excel("E:\DataScience\DSTI\Artificial Neural Networks\Project\Data\Tweets_Ref_100Labeled_ANN\Bi_2803_ANN.xlsx", 
                            names=["Text","Count","Label"])        
bi_2803 = bi_2803_raw.iloc[:100,:] 
other_bi_2803 = bi_2803.loc[bi_2803["Label"] == 0,"Count"].sum()
covid_bi_2803 = bi_2803.loc[bi_2803["Label"] != 0,"Count"].sum()


bi_2903_raw = pd.read_excel("E:\DataScience\DSTI\Artificial Neural Networks\Project\Data\Tweets_Ref_100Labeled_ANN\Bi_2903_ANN.xlsx", 
                            names=["Text","Count","Label"])        
bi_2903 = bi_2903_raw.iloc[:100,:] 
other_bi_2903 = bi_2903.loc[bi_2903["Label"] == 0,"Count"].sum()
covid_bi_2903 = bi_2903.loc[bi_2903["Label"] != 0,"Count"].sum()


bi_3003_raw = pd.read_excel("E:\DataScience\DSTI\Artificial Neural Networks\Project\Data\Tweets_Ref_100Labeled_ANN\Bi_3003_ANN.xlsx", 
                            names=["Text","Count","Label"])        
bi_3003 = bi_3003_raw.iloc[:100,:] 
other_bi_3003 = bi_3003.loc[bi_3003["Label"] == 0,"Count"].sum()
covid_bi_3003 = bi_3003.loc[bi_3003["Label"] != 0,"Count"].sum()


bi_3103_raw = pd.read_excel("E:\DataScience\DSTI\Artificial Neural Networks\Project\Data\Tweets_Ref_100Labeled_ANN\Bi_3103_ANN.xlsx", 
                            names=["Text","Count","Label"])        
bi_3103 = bi_3103_raw.iloc[:100,:] 
other_bi_3103 = bi_3103.loc[bi_3103["Label"] == 0,"Count"].sum()
covid_bi_3103 = bi_3103.loc[bi_3103["Label"] != 0,"Count"].sum()


bi_0104_raw = pd.read_excel("E:\DataScience\DSTI\Artificial Neural Networks\Project\Data\Tweets_Ref_100Labeled_ANN\Bi_0104_ANN.xlsx", 
                            names=["Text","Count","Label"])        
bi_0104 = bi_0104_raw.iloc[:100,:] 
other_bi_0104 = bi_0104.loc[bi_0104["Label"] == 0,"Count"].sum()
covid_bi_0104 = bi_0104.loc[bi_0104["Label"] != 0,"Count"].sum()


bi_0204_raw = pd.read_excel("E:\DataScience\DSTI\Artificial Neural Networks\Project\Data\Tweets_Ref_100Labeled_ANN\Bi_0204_ANN.xlsx", 
                            names=["Text","Count","Label"])        
bi_0204 = bi_0204_raw.iloc[:100,:] 
other_bi_0204 = bi_0204.loc[bi_0204["Label"] == 0,"Count"].sum()
covid_bi_0204 = bi_0204.loc[bi_0204["Label"] != 0,"Count"].sum()


bi_0304_raw = pd.read_excel("E:\DataScience\DSTI\Artificial Neural Networks\Project\Data\Tweets_Ref_100Labeled_ANN\Bi_0304_ANN.xlsx", 
                            names=["Text","Count","Label"])        
bi_0304 = bi_0304_raw.iloc[:100,:] 
other_bi_0304 = bi_0304.loc[bi_0304["Label"] == 0,"Count"].sum()
covid_bi_0304 = bi_0304.loc[bi_0304["Label"] != 0,"Count"].sum()


bi_0404_raw = pd.read_excel("E:\DataScience\DSTI\Artificial Neural Networks\Project\Data\Tweets_Ref_100Labeled_ANN\Bi_0404_ANN.xlsx", 
                            names=["Text","Count","Label"])        
bi_0404 = bi_0404_raw.iloc[:100,:] 
other_bi_0404 = bi_0404.loc[bi_0404["Label"] == 0,"Count"].sum()
covid_bi_0404 = bi_0404.loc[bi_0404["Label"] != 0,"Count"].sum()


bi_0504_raw = pd.read_excel("E:\DataScience\DSTI\Artificial Neural Networks\Project\Data\Tweets_Ref_100Labeled_ANN\Bi_0504_ANN.xlsx", 
                            names=["Text","Count","Label"])        
bi_0504 = bi_0504_raw.iloc[:100,:] 
other_bi_0504 = bi_0504.loc[bi_0504["Label"] == 0,"Count"].sum()
covid_bi_0504 = bi_0504.loc[bi_0504["Label"] != 0,"Count"].sum()


bi_0604_raw = pd.read_excel("E:\DataScience\DSTI\Artificial Neural Networks\Project\Data\Tweets_Ref_100Labeled_ANN\Bi_0604_ANN.xlsx", 
                            names=["Text","Count","Label"])        
bi_0604 = bi_0604_raw.iloc[:100,:] 
other_bi_0604 = bi_0604.loc[bi_0604["Label"] == 0,"Count"].sum()
covid_bi_0604 = bi_0604.loc[bi_0604["Label"] != 0,"Count"].sum()


bi_0704_raw = pd.read_excel("E:\DataScience\DSTI\Artificial Neural Networks\Project\Data\Tweets_Ref_100Labeled_ANN\Bi_0704_ANN.xlsx", 
                            names=["Text","Count","Label"])        
bi_0704 = bi_0704_raw.iloc[:100,:] 
other_bi_0704 = bi_0704.loc[bi_0704["Label"] == 0,"Count"].sum()
covid_bi_0704 = bi_0704.loc[bi_0704["Label"] != 0,"Count"].sum()


bi_0804_raw = pd.read_excel("E:\DataScience\DSTI\Artificial Neural Networks\Project\Data\Tweets_Ref_100Labeled_ANN\Bi_0804_ANN.xlsx", 
                            names=["Text","Count","Label"])        
bi_0804 = bi_0804_raw.iloc[:100,:] 
other_bi_0804 = bi_0804.loc[bi_0804["Label"] == 0,"Count"].sum()
covid_bi_0804 = bi_0804.loc[bi_0804["Label"] != 0,"Count"].sum()


bi_0904_raw = pd.read_excel("E:\DataScience\DSTI\Artificial Neural Networks\Project\Data\Tweets_Ref_100Labeled_ANN\Bi_0904_ANN.xlsx", 
                            names=["Text","Count","Label"])        
bi_0904 = bi_0904_raw.iloc[:100,:] 
other_bi_0904 = bi_0904.loc[bi_0904["Label"] == 0,"Count"].sum()
covid_bi_0904 = bi_0904.loc[bi_0904["Label"] != 0,"Count"].sum()


bi_1004_raw = pd.read_excel("E:\DataScience\DSTI\Artificial Neural Networks\Project\Data\Tweets_Ref_100Labeled_ANN\Bi_1004_ANN.xlsx", 
                            names=["Text","Count","Label"])        
bi_1004 = bi_1004_raw.iloc[:100,:] 
other_bi_1004 = bi_1004.loc[bi_1004["Label"] == 0,"Count"].sum()
covid_bi_1004 = bi_1004.loc[bi_1004["Label"] != 0,"Count"].sum()


bi_1104_raw = pd.read_excel("E:\DataScience\DSTI\Artificial Neural Networks\Project\Data\Tweets_Ref_100Labeled_ANN\Bi_1104_ANN.xlsx", 
                            names=["Text","Count","Label"])        
bi_1104 = bi_1104_raw.iloc[:100,:] 
other_bi_1104 = bi_1104.loc[bi_1104["Label"] == 0,"Count"].sum()
covid_bi_1104 = bi_1104.loc[bi_1104["Label"] != 0,"Count"].sum()


bi_1204_raw = pd.read_excel("E:\DataScience\DSTI\Artificial Neural Networks\Project\Data\Tweets_Ref_100Labeled_ANN\Bi_1204_ANN.xlsx", 
                            names=["Text","Count","Label"])        
bi_1204 = bi_1204_raw.iloc[:100,:] 
other_bi_1204 = bi_1204.loc[bi_1204["Label"] == 0,"Count"].sum()
covid_bi_1204 = bi_1204.loc[bi_1204["Label"] != 0,"Count"].sum()


bi_1304_raw = pd.read_excel("E:\DataScience\DSTI\Artificial Neural Networks\Project\Data\Tweets_Ref_100Labeled_ANN\Bi_1304_ANN.xlsx", 
                            names=["Text","Count","Label"])        
bi_1304 = bi_1304_raw.iloc[:100,:] 
other_bi_1304 = bi_1304.loc[bi_1304["Label"] == 0,"Count"].sum()
covid_bi_1304 = bi_1304.loc[bi_1304["Label"] != 0,"Count"].sum()


bi_1404_raw = pd.read_excel("E:\DataScience\DSTI\Artificial Neural Networks\Project\Data\Tweets_Ref_100Labeled_ANN\Bi_1404_ANN.xlsx", 
                            names=["Text","Count","Label"])        
bi_1404 = bi_1404_raw.iloc[:100,:] 
other_bi_1404 = bi_1404.loc[bi_1404["Label"] == 0,"Count"].sum()
covid_bi_1404 = bi_1404.loc[bi_1404["Label"] != 0,"Count"].sum()


bi_1504_raw = pd.read_excel("E:\DataScience\DSTI\Artificial Neural Networks\Project\Data\Tweets_Ref_100Labeled_ANN\Bi_1504_ANN.xlsx", 
                            names=["Text","Count","Label"])        
bi_1504 = bi_1504_raw.iloc[:100,:] 
other_bi_1504 = bi_1504.loc[bi_1504["Label"] == 0,"Count"].sum()
covid_bi_1504 = bi_1504.loc[bi_1504["Label"] != 0,"Count"].sum()


bi_1604_raw = pd.read_excel("E:\DataScience\DSTI\Artificial Neural Networks\Project\Data\Tweets_Ref_100Labeled_ANN\Bi_1604_ANN.xlsx", 
                            names=["Text","Count","Label"])        
bi_1604 = bi_1604_raw.iloc[:100,:] 
other_bi_1604 = bi_1604.loc[bi_1604["Label"] == 0,"Count"].sum()
covid_bi_1604 = bi_1604.loc[bi_1604["Label"] != 0,"Count"].sum()


bi_1704_raw = pd.read_excel("E:\DataScience\DSTI\Artificial Neural Networks\Project\Data\Tweets_Ref_100Labeled_ANN\Bi_1704_ANN.xlsx", 
                            names=["Text","Count","Label"])        
bi_1704 = bi_1704_raw.iloc[:100,:] 
other_bi_1704 = bi_1704.loc[bi_1704["Label"] == 0,"Count"].sum()
covid_bi_1704 = bi_1704.loc[bi_1704["Label"] != 0,"Count"].sum()


bi_1804_raw = pd.read_excel("E:\DataScience\DSTI\Artificial Neural Networks\Project\Data\Tweets_Ref_100Labeled_ANN\Bi_1804_ANN.xlsx", 
                            names=["Text","Count","Label"])        
bi_1804 = bi_1804_raw.iloc[:100,:] 
other_bi_1804 = bi_1804.loc[bi_1804["Label"] == 0,"Count"].sum()
covid_bi_1804 = bi_1804.loc[bi_1804["Label"] != 0,"Count"].sum()
 

bi_1904_raw = pd.read_excel("E:\DataScience\DSTI\Artificial Neural Networks\Project\Data\Tweets_Ref_100Labeled_ANN\Bi_1904_ANN.xlsx", 
                            names=["Text","Count","Label"])        
bi_1904 = bi_1904_raw.iloc[:100,:] 
other_bi_1904 = bi_1904.loc[bi_1904["Label"] == 0,"Count"].sum()
covid_bi_1904 = bi_1904.loc[bi_1904["Label"] != 0,"Count"].sum()


bi_2004_raw = pd.read_excel("E:\DataScience\DSTI\Artificial Neural Networks\Project\Data\Tweets_Ref_100Labeled_ANN\Bi_2004_ANN.xlsx", 
                            names=["Text","Count","Label"])        
bi_2004 = bi_2004_raw.iloc[:100,:] 
other_bi_2004 = bi_2004.loc[bi_2004["Label"] == 0,"Count"].sum()
covid_bi_2004 = bi_2004.loc[bi_2004["Label"] != 0,"Count"].sum()


bi_2104_raw = pd.read_excel("E:\DataScience\DSTI\Artificial Neural Networks\Project\Data\Tweets_Ref_100Labeled_ANN\Bi_2104_ANN.xlsx", 
                            names=["Text","Count","Label"])        
bi_2104 = bi_2104_raw.iloc[:100,:] 
other_bi_2104 = bi_2104.loc[bi_2104["Label"] == 0,"Count"].sum()
covid_bi_2104 = bi_2104.loc[bi_2104["Label"] != 0,"Count"].sum()


bi_2204_raw = pd.read_excel("E:\DataScience\DSTI\Artificial Neural Networks\Project\Data\Tweets_Ref_100Labeled_ANN\Bi_2204_ANN.xlsx",
                            names=["Text","Count","Label"])        
bi_2204 = bi_2204_raw.iloc[:100,:]  
other_bi_2204 = bi_2204.loc[bi_2204["Label"] == 0,"Count"].sum()
covid_bi_2204 = bi_2204.loc[bi_2204["Label"] != 0,"Count"].sum()

    
bi_2304_raw = pd.read_excel("E:\DataScience\DSTI\Artificial Neural Networks\Project\Data\Tweets_Ref_100Labeled_ANN\Bi_2304_ANN.xlsx", 
                            names=["Text","Count","Label"])        
bi_2304 = bi_2304_raw.iloc[:100,:] 
other_bi_2304 = bi_2304.loc[bi_2304["Label"] == 0,"Count"].sum()
covid_bi_2304 = bi_2304.loc[bi_2304["Label"] != 0,"Count"].sum()


bi_2404_raw = pd.read_excel("E:\DataScience\DSTI\Artificial Neural Networks\Project\Data\Tweets_Ref_100Labeled_ANN\Bi_2404_ANN.xlsx", 
                            names=["Text","Count","Label"])        
bi_2404 = bi_2404_raw.iloc[:100,:] 
other_bi_2404 = bi_2404.loc[bi_2404["Label"] == 0,"Count"].sum()
covid_bi_2404 = bi_2404.loc[bi_2404["Label"] != 0,"Count"].sum()


bi_2504_raw = pd.read_excel("E:\DataScience\DSTI\Artificial Neural Networks\Project\Data\Tweets_Ref_100Labeled_ANN\Bi_2504_ANN.xlsx", 
                            names=["Text","Count","Label"])        
bi_2504 = bi_2504_raw.iloc[:100,:] 
other_bi_2504 = bi_2504.loc[bi_2504["Label"] == 0,"Count"].sum()
covid_bi_2504 = bi_2504.loc[bi_2504["Label"] != 0,"Count"].sum()


bi_2604_raw = pd.read_excel("E:\DataScience\DSTI\Artificial Neural Networks\Project\Data\Tweets_Ref_100Labeled_ANN\Bi_2604_ANN.xlsx", 
                            names=["Text","Count","Label"])        
bi_2604 = bi_2604_raw.iloc[:100,:] 
other_bi_2604 = bi_2604.loc[bi_2604["Label"] == 0,"Count"].sum()
covid_bi_2604 = bi_2604.loc[bi_2604["Label"] != 0,"Count"].sum()


bi_2704_raw = pd.read_excel("E:\DataScience\DSTI\Artificial Neural Networks\Project\Data\Tweets_Ref_100Labeled_ANN\Bi_2704_ANN.xlsx", 
                            names=["Text","Count","Label"])        
bi_2704 = bi_2704_raw.iloc[:100,:] 
other_bi_2704 = bi_2704.loc[bi_2704["Label"] == 0,"Count"].sum()
covid_bi_2704 = bi_2704.loc[bi_2704["Label"] != 0,"Count"].sum()


bi_2804_raw = pd.read_excel("E:\DataScience\DSTI\Artificial Neural Networks\Project\Data\Tweets_Ref_100Labeled_ANN\Bi_2804_ANN.xlsx", 
                            names=["Text","Count","Label"])        
bi_2804 = bi_2804_raw.iloc[:100,:] 
other_bi_2804 = bi_2804.loc[bi_2804["Label"] == 0,"Count"].sum()
covid_bi_2804 = bi_2804.loc[bi_2804["Label"] != 0,"Count"].sum()


bi_2904_raw = pd.read_excel("E:\DataScience\DSTI\Artificial Neural Networks\Project\Data\Tweets_Ref_100Labeled_ANN\Bi_2904_ANN.xlsx", 
                            names=["Text","Count","Label"])        
bi_2904 = bi_2904_raw.iloc[:100,:] 
other_bi_2904 = bi_2904.loc[bi_2904["Label"] == 0,"Count"].sum()
covid_bi_2904 = bi_2904.loc[bi_2904["Label"] != 0,"Count"].sum()


bi_3004_raw = pd.read_excel("E:\DataScience\DSTI\Artificial Neural Networks\Project\Data\Tweets_Ref_100Labeled_ANN\Bi_3004_ANN.xlsx", 
                            names=["Text","Count","Label"])  
bi_3004 = bi_3004_raw.iloc[:100,:] 
other_bi_3004 = bi_3004.loc[bi_3004["Label"] == 0,"Count"].sum()
covid_bi_3004 = bi_3004.loc[bi_3004["Label"] != 0,"Count"].sum()


bi_0105_raw = pd.read_excel("E:\DataScience\DSTI\Artificial Neural Networks\Project\Data\Tweets_Ref_100Labeled_ANN\Bi_0105_ANN.xlsx", 
                            names=["Text","Count","Label"])        
bi_0105 = bi_0105_raw.iloc[:100,:] 
other_bi_0105 = bi_0105.loc[bi_0105["Label"] == 0,"Count"].sum()
covid_bi_0105 = bi_0105.loc[bi_0105["Label"] != 0,"Count"].sum()


bi_0205_raw = pd.read_excel("E:\DataScience\DSTI\Artificial Neural Networks\Project\Data\Tweets_Ref_100Labeled_ANN\Bi_0205_ANN.xlsx", 
                            names=["Text","Count","Label"])        
bi_0205 = bi_0205_raw.iloc[:100,:] 
other_bi_0205 = bi_0205.loc[bi_0205["Label"] == 0,"Count"].sum()
covid_bi_0205 = bi_0205.loc[bi_0205["Label"] != 0,"Count"].sum()


bi_0305_raw = pd.read_excel("E:\DataScience\DSTI\Artificial Neural Networks\Project\Data\Tweets_Ref_100Labeled_ANN\Bi_0305_ANN.xlsx", 
                            names=["Text","Count","Label"])        
bi_0305 = bi_0305_raw.iloc[:100,:] 
other_bi_0305 = bi_0305.loc[bi_0305["Label"] == 0,"Count"].sum()
covid_bi_0305 = bi_0305.loc[bi_0305["Label"] != 0,"Count"].sum()


bi_0405_raw = pd.read_excel("E:\DataScience\DSTI\Artificial Neural Networks\Project\Data\Tweets_Ref_100Labeled_ANN\Bi_0405_ANN.xlsx", 
                            names=["Text","Count","Label"])        
bi_0405 = bi_0405_raw.iloc[:100,:] 
other_bi_0405 = bi_0405.loc[bi_0405["Label"] == 0,"Count"].sum()
covid_bi_0405 = bi_0405.loc[bi_0405["Label"] != 0,"Count"].sum()



# Data frame for bi word
bi_2203to2603 = pd.concat([bi_2203, bi_2303, bi_2403, bi_2503, bi_2603])
bi_2703to3103 = pd.concat([bi_2703, bi_2803, bi_2903, bi_3003, bi_3103])
bi_0104to0504 = pd.concat([bi_0104, bi_0204, bi_0304, bi_0404, bi_0504])
bi_0604to1004 = pd.concat([bi_0604, bi_0704, bi_0804, bi_0904, bi_1004])
bi_1104to1504 = pd.concat([bi_1104, bi_1204, bi_1304, bi_1404, bi_1504])
bi_1604to2004 = pd.concat([bi_1604, bi_1704, bi_1804, bi_1904, bi_2004])
bi_2104to2504 = pd.concat([bi_2104, bi_2204, bi_2304, bi_2404, bi_2504])
bi_2604to3004 = pd.concat([bi_2604, bi_2704, bi_2804, bi_2904, bi_3004])
bi_0105to0405 = pd.concat([bi_0105, bi_0205, bi_0305, bi_0405])



#bi_train = pd.concat([bi_2203to2603, bi_2703to3103,bi_0104to0504, bi_0604to1004, bi_1104to1504,
#                   bi_1604to20045])
#bi_train["Label"].replace({2:1, 3:1},inplace = True)
#bi_train.to_csv('E:\DataScience\DSTI\Artificial Neural Networks\Project\Final\Tweets_word_for_Train.csv', index=False)



bi_all = pd.concat([bi_2203to2603, bi_2703to3103,bi_0104to0504, bi_0604to1004, bi_1104to1504,
                    bi_1604to2004, bi_2104to2504,bi_2604to3004, bi_0105to0405])
bi_all["Label"].replace({2:1, 3:1},inplace = True)
bi_all.to_csv('E:\DataScience\DSTI\Artificial Neural Networks\Project\Final\Tweets_word_45days.csv', index=False)

# Data frame for bi count
other_list_bi = [other_bi_2203, other_bi_2303, other_bi_2403, other_bi_2503, other_bi_2603,
              other_bi_2703, other_bi_2803, other_bi_2903, other_bi_3003, other_bi_3103,
              other_bi_0104, other_bi_0204, other_bi_0304, other_bi_0404, other_bi_0504,
              other_bi_0604, other_bi_0704, other_bi_0804, other_bi_0904, other_bi_1004,
              other_bi_1104, other_bi_1204, other_bi_1304, other_bi_1404, other_bi_1504,
              other_bi_1604, other_bi_1704, other_bi_1804, other_bi_1904, other_bi_2004,
              other_bi_2104, other_bi_2204, other_bi_2304, other_bi_2404, other_bi_2504,
              other_bi_2604, other_bi_2704, other_bi_2804, other_bi_2904, other_bi_3004,
              other_bi_0105, other_bi_0205, other_bi_0305, other_bi_0405]

covid_list_bi = [covid_bi_2203, covid_bi_2303, covid_bi_2403, covid_bi_2503, covid_bi_2603,
              covid_bi_2703, covid_bi_2803, covid_bi_2903, covid_bi_3003, covid_bi_3103,
              covid_bi_0104, covid_bi_0204, covid_bi_0304, covid_bi_0404, covid_bi_0504,
              covid_bi_0604, covid_bi_0704, covid_bi_0804, covid_bi_0904, covid_bi_1004,
              covid_bi_1104, covid_bi_1204, covid_bi_1304, covid_bi_1404, covid_bi_1504,
              covid_bi_1604, covid_bi_1704, covid_bi_1804, covid_bi_1904, covid_bi_2004,
              covid_bi_2104, covid_bi_2204, covid_bi_2304, covid_bi_2404, covid_bi_2504,
              covid_bi_2604, covid_bi_2704, covid_bi_2804, covid_bi_2904, covid_bi_3004,
              covid_bi_0105, covid_bi_0205, covid_bi_0305, covid_bi_0405]



df_twitter_bi_raw = pd.DataFrame(np.array([other_list_bi, covid_list_bi]))            
df_twitter_bi = df_twitter_bi_raw.T
df_twitter_bi.rename(columns={0:"Others Tweets", 1:"Covid-19 Tweets", 2:"Quarantine Tweets"}, inplace = True)


# Plot others tweets vs covid tweets: Daily
df_twitter_bi.plot(kind = "bar")
plt.show()

# Plot others tweets vs covid tweets: Total
fig=plt.figure()
ax = fig.add_subplot(111)    # The big subplot
ax1 = plt.subplot(211)
ax2 = plt.subplot(212)

# Turn off axis lines and ticks of the big subplot
ax.spines['top'].set_color('none')
ax.spines['bottom'].set_color('none')
ax.spines['left'].set_color('none')
ax.spines['right'].set_color('none')
ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)

# Set common labels
fig.text(0.06, 0.5, 'Count', ha='center', va='center', rotation='vertical')

ax1.plot(date,df_twitter_bi["Others Tweets"])
ax2.plot(date,df_twitter_bi["Covid-19 Tweets"])

ax1.set_title("Others Tweets", weight='heavy', fontsize = 30)
ax2.set_title("Covid-19 Tweets", weight='heavy', fontsize = 30)

ax1.get_shared_x_axes().join(ax1, ax2)
ax1.set_xticklabels([])

plt.xticks(rotation=45)
plt.show()



#---------------------------------------------------------------------------------------------#

# Data frame for all feature
df_feature_vs_covid = pd.concat([df_covid, df_mobility, df_twitter_bi],axis = 1)
df_feature_vs_covid.to_csv('E:\DataScience\DSTI\Artificial Neural Networks\Project\Final\Feature_vs_Covid_ANN.csv', index=False)


# Word Cloud for top 50 popular words of all Tweets during these 45 days
top500 = bi_all.nlargest(500,["Count"])                   # select top 500 words among these 45 days 
aggregation_functions = {"Count":"sum", "Text":"first"}   # sum count of the same word
unique_top500 = top500.groupby(top500["Text"]).aggregate(aggregation_functions)
unique_top500.sort_values(by="Count", axis = 0, ascending=False, inplace = True)  # sort word base on count
# Generate word cloud // Top 50   
wordcloud = WordCloud(max_font_size=50, max_words=50, 
                      background_color="black", collocations=False).generate(str(unique_top500))
# plot wordcloud
plt.figure(figsize=(18, 8))
plt.imshow(wordcloud)
plt.axis("off")
plt.show()


# Plot top 5 popular bi term of others tweets
other_word500 = top500.loc[top500["Label"] == 0]
other_word500_rank = other_word500.groupby(other_word500["Text"]).aggregate(aggregation_functions)
other_word500_rank.sort_values(by="Count", axis = 0, ascending=True, inplace = True)  # sort word base on count
other_word10 = other_word500_rank.iloc[-5:,:]

other_count = other_word10.iloc[:,0]
other_count_list = other_count.values.tolist()
other_word = other_word10.iloc[:,1]
other_word_list = other_word.values.tolist()

plt.barh(other_word_list,other_count_list,  align="center")

# Plot top 5 popular bi term of covid tweets
covid_word500 = top500.loc[top500["Label"] == 1]
covid_word500_rank = covid_word500.groupby(covid_word500["Text"]).aggregate(aggregation_functions)
covid_word500_rank.sort_values(by="Count", axis = 0, ascending=True, inplace = True)  # sort word base on count
covid_word10 = covid_word500_rank.iloc[-5:,:]

covid_count = covid_word10.iloc[:,0]
covid_count_list = covid_count.values.tolist()
covid_word = covid_word10.iloc[:,1]
covid_word_list = covid_word.values.tolist()

plt.barh(covid_word_list,covid_count_list,  align="center")


# Pairwise correlation between features
df_bi_cor = df_feature_vs_covid.iloc[:,[2,5, 6, 7, 8]]
bi_cor = df_bi_cor.corr(method = "pearson")
fig = plt.figure(figsize = (15,15))
sb.heatmap(bi_cor, vmax = .8, square = True, annot=True)
plt.show()



# Plot: Features vs Covid new cases
ax = plt.gca()
df_feature_vs_covid.plot(kind = "line", x = "Date", y = "New Cases", label="Covid19_New Cases", ax=ax)
df_feature_vs_covid.plot(kind = "line", x = "Date", y = "Covid19 Tweets",label="Twitter_Covid", linestyle = "dashed",  ax=ax)
df_feature_vs_covid.plot(kind = "line", x = "Date", y = "Others Tweets", label="Twitter_Others Topic",  linestyle = "dashed", ax=ax)
df_feature_vs_covid.plot(kind = "line", x = "Date", y = "Transit Station Mobility", label="Mobility",  linestyle = "dashed", ax=ax)  # Note that we have to normalize the unit before make model
df_feature_vs_covid.plot(kind = "line", x = "Date", y = "Resident Mobility", label="Stay Home",  linestyle = "dashed", ax=ax)        # Note that we have to normalize the unit before make model
# Should present in subplot as above because mobility infos have dif scale






