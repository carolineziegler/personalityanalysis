#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


inpPath = "C:/CarolineZiegler/Studium_DCU/8. Semester/Data Analytics for Marketing Applications/Pair Assignments/"
demographics = pd.read_csv(inpPath + "demographics.csv", delimiter =  ",", header = 0)
demographics


# In[3]:


demographics.isna().sum()


# In[4]:


round(demographics.describe(),2)


# In[5]:


demographics["Age (in years)"].unique()


# In[6]:


demographics["Age (in years)"].value_counts()


# In[7]:


demographics["Gender"].value_counts()


# In[8]:


demographics["Seat row in class"].unique()


# In[9]:


demographics_is_na = demographics.iloc[:5, :].copy()
demographics_is_na


# In[10]:


demographics_no_na = demographics.iloc[5:, :].copy()
demographics_no_na


# In[11]:


plt.scatter(demographics_no_na["Old Dublin postcode (0 if outside Dublin)"], demographics_no_na["Daily travel to DCU (in km, 0 if on-campus)"])
plt.title("Relation between Postcode and Daily Travel")
plt.xlabel("Postcode")
plt.ylabel("Daily Travel")
plt.show()


# In[12]:


demographics_no_na_pd = demographics_no_na["Daily travel to DCU (in km, 0 if on-campus)"].unique()
np.set_printoptions(suppress=True, precision =2)
demographics_no_na_pd


# In[13]:


demographics_no_na[demographics_no_na["Old Dublin postcode (0 if outside Dublin)"] == 0]["Old Dublin postcode (0 if outside Dublin)"].count()


# In[14]:


demographics_no_na_post_dist = demographics_no_na[["Daily travel to DCU (in km, 0 if on-campus)", "Old Dublin postcode (0 if outside Dublin)"]]
demographics_no_na_post_dist.groupby("Old Dublin postcode (0 if outside Dublin)").mean()


# In[15]:


demographics_no_na_post_dist.groupby("Old Dublin postcode (0 if outside Dublin)").max()


# In[16]:


demographics_no_na_post_dist.groupby("Old Dublin postcode (0 if outside Dublin)").std()


# In[17]:


demographics_is_na.iloc[1,2] = 11.60
demographics_is_na


# In[18]:


# import libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score

# Split the data columnwise into the independt variables (x) and the dependent variable (y)
xDf = demographics_no_na[['Age (in years)','Average year 1 exam result (as %)']]
yDf = demographics_no_na['CAO Points (100 to 600)']

# Split between train and test set
X_train, X_test, y_train, y_test = train_test_split(xDf, yDf, test_size=0.3, random_state=0)

#fitting the model
reg_lin = LinearRegression().fit(X_train, y_train)

#getting the accuracy
print(reg_lin.score(X_test, y_test))
print(reg_lin.coef_)
print(reg_lin.predict([[21,66]]))


# In[19]:


demographics_is_na.iloc[0,1] = 491
demographics_is_na


# In[20]:


# Split the data columnwise into the independt variables (x) and the dependent variable (y)
xDf = demographics_no_na[['Age (in years)', 'CAO Points (100 to 600)']]
yDf = demographics_no_na['Average year 1 exam result (as %)']

# Split between train and test set
X_train, X_test, y_train, y_test = train_test_split(xDf, yDf, test_size=0.3, random_state=0)

#fitting the model
reg_lin = LinearRegression().fit(X_train, y_train)

#getting the accuracy
print(reg_lin.score(X_test, y_test))
print(reg_lin.coef_)
print(reg_lin.predict([[20,600]]))


# In[21]:


demographics_is_na.iloc[2,3] = 70.5
demographics_is_na


# In[22]:


# Split the data columnwise into the independent variables (x) and the dependent variable (y)
xDf = demographics_no_na[['Age (in years)', 'CAO Points (100 to 600)', "Average year 1 exam result (as %)"]]
yDf = demographics_no_na['Seat row in class']

# Split between train and test set
X_train, X_test, y_train, y_test = train_test_split(xDf, yDf, test_size=0.3, random_state=0)

#fitting the model
reg_lin = LinearRegression().fit(X_train, y_train)

#getting the accuracy
print(reg_lin.score(X_test, y_test))
print(reg_lin.coef_)
print(reg_lin.predict([[19,543,71]]))


# In[23]:


#no meaningful correlation which was not expected in any other way
np.random.randint(1,12)


# In[84]:


demographics_is_na.iloc[3,4] = 7
demographics_is_na


# In[85]:


#because we cannot predict the age here we use the median to fill it in
demographics_is_na.iloc[4,0] = 21
demographics_is_na


# In[86]:


normalised_demographics = pd.concat([demographics_is_na, demographics_no_na], axis = 0)
normalised_demographics


# In[27]:


inpPath2 = "C:/CarolineZiegler/Studium_DCU/8. Semester/Data Analytics for Marketing Applications/Pair Assignments/"
personalities = pd.read_csv(inpPath2 + "personalities.csv", delimiter =  ",", header = 0)
personalities


# In[28]:


personalities.rename(columns = {"Last 4 digits of your mobile (same as on previous form)": "Last 4 digits of your mobile (0000 to 9999)", 
                        "Your rating for EXTRAVERSION (vs. introversion)":"Extraversion", 
                        "Your rating for INTUITION (vs. observation)": "Intuition", 
                        "Your rating for THINKING (vs. feeling)":"Thinking", 
                        "Your rating for JUDGING (vs. prospecting)":"Judging", 
                        "Your rating for ASSERTIVE (vs. turbulent)":"Assertive",
                        }, inplace = True)
personalities


# In[29]:


personalities[personalities["Last 4 digits of your mobile (0000 to 9999)"] == 4397]


# In[30]:


personalities.drop(53, axis =0, inplace = True)
personalities


# In[31]:


personalities[personalities["Last 4 digits of your mobile (0000 to 9999)"] == 4397]


# In[32]:


demo_perso = normalised_demographics.merge(personalities, on = "Last 4 digits of your mobile (0000 to 9999)", how = "inner")
demo_perso


# In[33]:


personalities["Last 4 digits of your mobile (0000 to 9999)"].unique()


# In[34]:


personalities["Last 4 digits of your mobile (0000 to 9999)"].value_counts()


# In[35]:


normalised_demographics["Last 4 digits of your mobile (0000 to 9999)"].unique()


# In[36]:


common_lst = []
for el in normalised_demographics['Last 4 digits of your mobile (0000 to 9999)'].unique():
    if el in personalities['Last 4 digits of your mobile (0000 to 9999)'].unique():
        common_lst.append(el)
print(len(common_lst))


# In[37]:


print(common_lst)


# In[38]:


personalities[personalities["Last 4 digits of your mobile (0000 to 9999)"] == 1699]


# In[39]:


personalities[personalities["Last 4 digits of your mobile (0000 to 9999)"] == 1462]


# In[40]:


personalities.drop(89, axis =0, inplace = True)
personalities


# In[41]:


personalities.drop(97, axis =0, inplace = True)
personalities


# In[42]:


demo_perso = normalised_demographics.merge(personalities, on = "Last 4 digits of your mobile (0000 to 9999)", how = "inner")
demo_perso


# In[43]:


demo_perso_outer = normalised_demographics.merge(personalities, on = "Last 4 digits of your mobile (0000 to 9999)", how = "outer")
demo_perso_outer


# In[44]:


demo_perso[["Gender","Extraversion","Intuition","Thinking","Judging","Assertive"]].groupby("Gender").mean()


# In[45]:


men_women_pers_data = demo_perso[["Gender","Extraversion","Intuition","Thinking","Judging","Assertive"]]
men_pers_data = men_women_pers_data[men_women_pers_data['Gender']== 'Male']
women_pers_data = men_women_pers_data[men_women_pers_data['Gender']== 'Female']

men_pers_data


# In[46]:


from scipy import stats


# In[47]:


# Sample data
men_extra = men_pers_data["Extraversion"]
women_extra = women_pers_data["Extraversion"]

# Perform independent t-test
t_statistic, p_value = stats.ttest_ind(men_extra, women_extra)

# Print the results
print("t-statistic:", t_statistic)
print("p-value:", p_value)


# In[48]:


# Sample data
men_extra = men_pers_data["Intuition"]
women_extra = women_pers_data["Intuition"]

# Perform independent t-test
t_statistic, p_value = stats.ttest_ind(men_extra, women_extra)

# Print the results
print("t-statistic:", t_statistic)
print("p-value:", p_value)


# In[49]:


# Sample data
men_extra = men_pers_data["Thinking"]
women_extra = women_pers_data["Thinking"]

# Perform independent t-test
t_statistic, p_value = stats.ttest_ind(men_extra, women_extra)

# Print the results
print("t-statistic:", t_statistic)
print("p-value:", p_value)


# In[50]:


# Sample data
men_extra = men_pers_data["Judging"]
women_extra = women_pers_data["Judging"]

# Perform independent t-test
t_statistic, p_value = stats.ttest_ind(men_extra, women_extra)

# Print the results
print("t-statistic:", t_statistic)
print("p-value:", p_value)


# In[51]:


# Sample data
men_extra = men_pers_data["Assertive"]
women_extra = women_pers_data["Assertive"]

# Perform independent t-test
t_statistic, p_value = stats.ttest_ind(men_extra, women_extra)

# Print the results
print("t-statistic:", t_statistic)
print("p-value:", p_value)


# In[52]:


demo_perso[demo_perso["Last 4 digits of your mobile (0000 to 9999)"] == 2785]


# In[53]:


round(demo_perso.describe(),2)


# In[54]:


# Split the data columnwise into the independt variables (x) and the dependent variable (y)
xDf = demo_perso[['Weight (in kg)', 'Shoe size']]
yDf = demo_perso['Height (in cm)']

# Split between train and test set
X_train, X_test, y_train, y_test = train_test_split(xDf, yDf, test_size=0.3, random_state=0)

#fitting the model
reg_lin = LinearRegression().fit(X_train, y_train)

#getting the accuracy
print(reg_lin.score(X_test, y_test))
print(reg_lin.coef_)
#print(reg_lin.predict([[20,600]]))


# In[55]:


demo_perso["Total Siblings"] = demo_perso["Number of older siblings"] + demo_perso["Number of younger siblings"]
demo_perso


# In[56]:


# Split the data columnwise into the independt variables (x) and the dependent variable (y)
xDf = demo_perso[['Total Siblings']]
yDf = demo_perso['Average year 1 exam result (as %)']

# Split between train and test set
X_train, X_test, y_train, y_test = train_test_split(xDf, yDf, test_size=0.3, random_state=0)

#fitting the model
reg_lin = LinearRegression().fit(X_train, y_train)

#getting the accuracy
print(reg_lin.score(X_test, y_test))
print(reg_lin.coef_)


# In[57]:


# Split the data columnwise into the independt variables (x) and the dependent variable (y)
xDf = demo_perso[['Number of older siblings', "Number of younger siblings"]]
yDf = demo_perso['Average year 1 exam result (as %)']

# Split between train and test set
X_train, X_test, y_train, y_test = train_test_split(xDf, yDf, test_size=0.3, random_state=0)

#fitting the model
reg_lin = LinearRegression().fit(X_train, y_train)

#getting the accuracy
print(reg_lin.score(X_test, y_test))
print(reg_lin.coef_)


# In[58]:


plt.scatter(demo_perso["Seat row in class"], demo_perso["Average year 1 exam result (as %)"])
plt.title("Relation between Seat Row and Year 1 Exam Results")
plt.xlabel("Seat Row")
plt.ylabel("Year 1 Exam Results")
plt.show()


# In[59]:


corr_matrix = demo_perso.corr()
corr_matrix


# In[60]:


tpllst = []
for i in range(0,18):
    for j in range(0,18):
        if corr_matrix.iloc[i,j]>0.6 and corr_matrix.iloc[i,j]<1.0:
            tpllst.append((corr_matrix.index[i], corr_matrix.index[j], corr_matrix.iloc[i,j]))
tpllst


# In[61]:


demo_perso["Eye colour"].unique()


# In[62]:


demo_perso["Hair colour"].unique()


# In[63]:


demo_perso["Star sign"].unique()


# In[64]:


len(demo_perso["Star sign"].unique())


# In[65]:


demo_perso["Star sign"].value_counts()


# In[66]:


#For the hair colour and eye colour, labe encoding can be used to be ablre to perform analysis with the data using bot of them with an ordinal scale from light to dark 


# In[67]:


encoded_dict = {'Blonde': 0, 'Red': 1, 'Brown': 2, "Black":3}
demo_perso["Hair Colour Encoded"] = demo_perso["Hair colour"].map(encoded_dict)
demo_perso


# In[68]:


encoded_dict2 = {'Blue': 0, 'Green': 1, 'Brown': 2}
demo_perso["Eye Colour Encoded"] = demo_perso["Eye colour"].map(encoded_dict2)
demo_perso


# In[69]:


# Split the data columnwise into the independt variables (x) and the dependent variable (y)
xDf = demo_perso[['Hair Colour Encoded']]
yDf = demo_perso['Eye Colour Encoded']

# Split between train and test set
X_train, X_test, y_train, y_test = train_test_split(xDf, yDf, test_size=0.3, random_state=0)

#fitting the model
reg_lin = LinearRegression().fit(X_train, y_train)

#getting the accuracy
print(reg_lin.score(X_test, y_test))
print(reg_lin.coef_)


# In[70]:


demo_perso[['Hair Colour Encoded', 'Eye Colour Encoded']].corr()


# In[71]:


# import libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score

# Split the data columnwise into the independt variables (x) and the dependent variable (y)
xDf = demo_perso[["Extraversion", "Intuition", "Thinking", "Judging", "Assertive"]]
yDf = demo_perso['CAO Points (100 to 600)']

# Split between train and test set
X_train, X_test, y_train, y_test = train_test_split(xDf, yDf, test_size=0.3, random_state=0)

#fitting the model
reg_lin = LinearRegression().fit(X_train, y_train)

#getting the accuracy
print(reg_lin.score(X_test, y_test))
print(reg_lin.coef_)


# In[72]:


round(demo_perso[['Star sign', "Extraversion", "Intuition", "Thinking", "Judging", "Assertive"]].groupby("Star sign").mean(),2)


# In[73]:


demo_perso[['Star sign', "Extraversion", "Intuition", "Thinking", "Judging", "Assertive"]].groupby("Star sign").max()


# In[74]:


taur_extra = demo_perso[demo_perso["Star sign"] == "Taurus"]["Extraversion"]
taur_extra


# In[75]:


libra_extra = demo_perso[demo_perso["Star sign"] == "Libra"]["Extraversion"]
libra_extra


# In[76]:


# Perform independent t-test
t_statistic, p_value = stats.ttest_ind(taur_extra, libra_extra)

# Print the results
print("t-statistic:", t_statistic)
print("p-value:", p_value)


# In[77]:


pices_int = demo_perso[demo_perso["Star sign"] == "Pices"]["Intuition"]
libra_int = demo_perso[demo_perso["Star sign"] == "Libra"]["Intuition"]

# Perform independent t-test
t_statistic, p_value = stats.ttest_ind(pices_int, libra_int)

# Print the results
print("t-statistic:", t_statistic)
print("p-value:", p_value)


# In[78]:


gemini_think = demo_perso[demo_perso["Star sign"] == "Gemini"]["Thinking"]
sagi_think = demo_perso[demo_perso["Star sign"] == "Sagittarius"]["Thinking"]

# Perform independent t-test
t_statistic, p_value = stats.ttest_ind(gemini_think, sagi_think)

# Print the results
print("t-statistic:", t_statistic)
print("p-value:", p_value)


# In[79]:


capri_judg = demo_perso[demo_perso["Star sign"] == "Capricorn"]["Judging"]
libra_judg = demo_perso[demo_perso["Star sign"] == "Libra"]["Judging"]

# Perform independent t-test
t_statistic, p_value = stats.ttest_ind(capri_judg, libra_judg)

# Print the results
print("t-statistic:", t_statistic)
print("p-value:", p_value)


# In[80]:


virgo_asser = demo_perso[demo_perso["Star sign"] == "Virgo"]["Assertive"]
sagi_asser = demo_perso[demo_perso["Star sign"] == "Sagittarius"]["Assertive"]

# Perform independent t-test
t_statistic, p_value = stats.ttest_ind(virgo_asser, sagi_asser)

# Print the results
print("t-statistic:", t_statistic)
print("p-value:", p_value)


# In[81]:


plt.plot(taur_extra, marker = "s", color = "green", linestyle = "-", linewidth = 1.5, label = "Taurus")
plt.plot(libra_extra, marker = "s", color = "orange", linestyle = "-", linewidth = 1.5, label = "Libra")
plt.title("Extraversion Taurus and Libra")
plt.xlabel("Observations Index")
plt.ylabel("Personality Score")
plt.legend()


# In[82]:


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

# Plot the first line graph on the first subplot
ax1.plot(range(0,10), taur_extra, color='green', label='Taurus')
ax1.set_xlabel('Observations')
ax1.set_ylabel('Personality Score')
ax1.set_title('Taurus Extraversion')
ax1.legend()
ax1.set_ylim(0,100)
ax1.set_xticks([])

# Plot the second line graph on the second subplot
ax2.plot(range(0,5), libra_extra, color='orange', label='Libra')
ax2.set_xlabel('Observations')
ax2.set_ylabel('Personality Score')
ax2.set_title('Libra Extraversion')
ax2.legend()
ax2.set_ylim(0,100)
ax2.set_xticks([])

plt.tight_layout()

plt.show()


# In[83]:


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

# Plot the first line graph on the first subplot
ax1.plot(range(0,7), gemini_think, color='blue', label='Gemini')
ax1.set_xlabel('Observations')
ax1.set_ylabel('Personality Score')
ax1.set_title('Gemini Extraversion')
ax1.legend()
ax1.set_ylim(0,100)
ax1.set_xticks([])

# Plot the second line graph on the second subplot
ax2.plot(range(0,6), sagi_think, color='pink', label='Sagittarius')
ax2.set_xlabel('Observations')
ax2.set_ylabel('Personality Score')
ax2.set_title('Sagittarius Extraversion')
ax2.legend()
ax2.set_ylim(0,100)
ax2.set_xticks([])

plt.tight_layout()

plt.show()


# In[ ]:


corr_matrix


# In[87]:


import seaborn as sns


# In[88]:


sns.heatmap(correlations)


# In[89]:


plt.scatter(demo_perso["Shoe size"], demo_perso["Height (in cm)"])
plt.title("Relation between Shoe Size and Height")
plt.xlabel("Shoe Size")
plt.ylabel("Height")
plt.show()


# In[100]:


# Split the data columnwise into the independt variables (x) and the dependent variable (y)
xDf = demo_perso[['Weight (in kg)']]
yDf = demo_perso['Height (in cm)']

# Split between train and test set
X_train, X_test, y_train, y_test = train_test_split(xDf, yDf, test_size=0.3, random_state=0)

#fitting the model
reg_lin_weight = LinearRegression().fit(X_train, y_train)
intercept_weight = reg_lin_weight.intercept_ 

#getting the accuracy
print(reg_lin_weight.score(X_test, y_test))
print(reg_lin_weight.coef_)
print(intercept_weight)

# Split the data columnwise into the independt variables (x) and the dependent variable (y)
xDf = demo_perso[['Shoe size']]
yDf = demo_perso['Height (in cm)']

# Split between train and test set
X_train, X_test, y_train, y_test = train_test_split(xDf, yDf, test_size=0.3, random_state=0)

#fitting the model
reg_lin_shoe = LinearRegression().fit(X_train, y_train)
intercept_shoe = reg_lin_shoe.intercept_ 

#getting the accuracy
print(reg_lin_shoe.score(X_test, y_test))
print(reg_lin_shoe.coef_)
print(intercept_shoe)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

# Plot the first line graph on the first subplot
ax1.scatter(demo_perso["Shoe size"], demo_perso["Height (in cm)"])
ax1.plot(demo_perso["Shoe size"], intercept_shoe + reg_lin_shoe.coef_*demo_perso["Shoe size"], color = "red")
ax1.set_xlabel('Shoe Size')
ax1.set_ylabel('Height (in cm)')
ax1.set_title('Regression Shoe Size & Height')
ax1.set_ylim(140,210)


# Plot the second line graph on the second subplot
ax2.scatter(demo_perso["Weight (in kg)"], demo_perso["Height (in cm)"])
ax2.plot(demo_perso["Weight (in kg)"], intercept_weight + reg_lin_weight.coef_*demo_perso["Weight (in kg)"], color = "red")
ax2.set_xlabel('Weight')
ax2.set_ylabel('Height (in cm)')
ax2.set_title('Regression Weight & Height')
ax2.set_ylim(140,210)


plt.tight_layout()

plt.show()


# In[ ]:




