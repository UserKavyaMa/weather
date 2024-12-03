#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
weather = pd.read_csv("dataset.csv", index_col="datetime")


# Here we have imported pandas to filter Ahmedabad's weather data set

# In[4]:


weather


# In[5]:


weather.index = pd.to_datetime(weather.index)
weather.index


# In[6]:


weather.index.year.value_counts().sort_index()


# We had plot temperature graph and will define a function to predict the future temperature

# In[8]:


weather["temp"].plot()


# In[9]:


weather["humidity"].plot()


# In[10]:


weather["Prediction-temperature"] =  weather.shift(-1)["temp"]


# In[11]:


weather


# In[12]:


weather = weather.ffill()
weather


# We had imported Ridge regression model from sklearn it is very similar to linear regression model  and initialized it.

# In[14]:


from sklearn.linear_model import Ridge
rr = Ridge(alpha=.1)


# Predictors columns are created to predict the temperature

# In[16]:


predictors = weather.columns[~weather.columns.isin(["name","Prediction-temperature"
                                                   ])]


# In[17]:


predictors


# We are defining function called backtest which is going 
# to take weather data frame,Ridge model,predictors
# 

# In[19]:


def backtest(weather,model,predictors,start=60,step=10):
    all_prediction = []

    for i in range(start,weather.shape[0],step):
        train = weather.iloc[:i,:]
        test = weather.iloc[i:(i+step),:]

        model.fit(train[predictors],train["Prediction-temperature"])

        preds = model.predict(test[predictors])

        preds = pd.Series(preds,index=test.index)
        combined = pd.concat([test["Prediction-temperature"],preds],axis=1)

        combined.columns = ["actual","prediction"]
        combined["diff"] = (combined["prediction"] - combined["actual"]).abs()

        all_prediction.append(combined)
    return pd.concat(all_prediction)


# In[20]:


predictions = backtest(weather, rr, predictors)


# In[21]:


predictions


# 
