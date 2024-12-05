#!/usr/bin/env python
# coding: utf-8

# ## PySpark ML

# In[21]:


from pyspark.sql import SparkSession


# In[2]:


spark=SparkSession.builder.appName('Missing').getOrCreate()


# In[22]:


training=spark.read.csv('test6.csv',header=True,inferSchema=True)
training.show()


# In[23]:


training.printSchema()


# In[24]:


training.columns


# In[ ]:


[Age, Experience] --> new feature--->independent feature


# In[28]:


from pyspark.ml.feature import VectorAssembler
featureassembler=VectorAssembler(inputCols=["Age","Experience"],outputCol="Independent Feature")


# In[29]:


output=featureassembler.transform(training)


# In[30]:


output.show()


# In[31]:


output.columns


# In[32]:


finalized_data = output.select("Independent Feature","Salary")


# In[33]:


finalized_data


# In[34]:


finalized_data.show()


# In[41]:


from pyspark.ml.regression import LinearRegression
train_data,test_data=finalized_data.randomSplit([0.75,0.25])
regressor=LinearRegression(featuresCol='Independent Feature', labelCol='Salary')
regressor =regressor.fit(train_data)


# In[42]:


regressor.coefficients


# In[43]:


regressor.intercept


# In[44]:


pred_results=regressor.evaluate(test_data)


# In[45]:


pred_results.predictions.show()


# In[ ]:




