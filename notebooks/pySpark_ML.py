# Databricks notebook source
# MAGIC %md
# MAGIC ### Introduction
# MAGIC 
# MAGIC Using Spark ML.
# MAGIC 
# MAGIC Try and predict count (cnt) value of number of bike rides a day in a bike ride sharing service.

# COMMAND ----------

# MAGIC %md #### Set up

# COMMAND ----------

from pyspark.sql.types import *
from pyspark.sql.functions import *

# Inspect spark context
print(sc.version)
print(sc.pythonVer)
print(sc.master) #master URl to connect to
print(str(sc.sparkHome))  # Path where Spark is installed on worker nodes
print(str(sc.sparkUser()))  # Retrieve name of the Spark User running SparkContext

print(sc.appName)  # Return application name
print(sc.applicationId)  # Retrieve application ID
print("Parrelism :", sc.defaultParallelism)  # Return default level of parallelism
print ("Minimum number of paritions :", sc.defaultMinPartitions)  # Default minimum number of partitions

# COMMAND ----------

# MAGIC %md #### Get Data

# COMMAND ----------

# For this example, the dataset we are going to be using is the bike sharing dataset which is available on `dbfs`.
df = sqlContext.read.format("csv")\
  .load("dbfs:/databricks-datasets/bikeSharing/data-001/day.csv", header=True)

display(df)

# COMMAND ----------

# MAGIC %md #### Convert Data Types

# COMMAND ----------

def convertColumn(df, names, newType):
  for name in names: 
     df = df.withColumn(name, df[name].cast(newType))
  return df

# Arrange data types
columns_float = [ 'temp', 'atemp', 'hum', 'windspeed']
columns_int = ['instant', 'yr', 'holiday', 'workingday', 'casual', 'registered', 'cnt']
# Categorical variables stored as string: Only the ones used later at the moment.
columns_string = ['season', 'weekday', 'mnth', 'weathersit']
columns_date = ['dteday']

# Convert to new data types
df = convertColumn(df, columns_float, FloatType())
df = convertColumn(df, columns_int, IntegerType())
df = convertColumn(df, columns_date, DateType())

df.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Exploratory Data Analysis
# MAGIC 
# MAGIC This is what is being predicted, the count of daily rides in a time series.

# COMMAND ----------

display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC Casual and registered are two components of count. The reflect if the riders are casual riders that day or registered riders. They should not be used in model training!

# COMMAND ----------

display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC The distribution of the count variable.

# COMMAND ----------

display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC QQ plots are a more descriptive way to view a distribution of a variable, they just take a bit of getting used to. In this case a categorical
# MAGIC variable is used to subset the plots to look for any systematic variations.
# MAGIC 
# MAGIC Comparing count to season or month shows variances.

# COMMAND ----------

display(df)

# COMMAND ----------

display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC Comparing weedays and working days.

# COMMAND ----------

display(df)

# COMMAND ----------

display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Pair plot to look for correlations within the dataset. Added weekday as a cetogorical scheme.
# MAGIC 
# MAGIC This shows that temperature has correlation that could be investigated further.

# COMMAND ----------

display(df)

# COMMAND ----------

# MAGIC %md #### Feature engineering

# COMMAND ----------

display(df)

# COMMAND ----------

# Just a demo, temp over humidty would not neccesairly be a good feature
df = df.withColumn("temp_hum", col("temp")/col("hum"))

display(df)

# COMMAND ----------

# MAGIC %md #### Data Processing
# MAGIC 
# MAGIC Firstly pySpark likes it´s data converted into a Dense Vector type for linear algerbra and machine learning

# COMMAND ----------

df.columns

# COMMAND ----------

# Reorder the dataset and only keep the required columns
df = df.select('cnt',
               'season',
               'yr',
               'mnth',
               'holiday',
               'weekday',
               'workingday',
               'weathersit',
               'temp',
               'atemp',
               'hum',
               'windspeed',
               'temp_hum'
               )

df.show()

# COMMAND ----------

# MAGIC %md
# MAGIC Change categorical variables into more useful pieces of data for machine learning.
# MAGIC 
# MAGIC Here we have some categorical variables like season, weather, month. To include these in our model, we'll need to make binary dummy variables.

# COMMAND ----------

import pyspark.sql.functions as F

season = df.select("season").distinct().rdd.flatMap(lambda x: x).collect()
season_expr = [F.when(F.col("season") == ty, 1).otherwise(0).alias("e_season_" + ty) for ty in season]

weathersit = df.select("weathersit").distinct().rdd.flatMap(lambda x: x).collect()
weathersit_expr = [F.when(F.col("weathersit") == ty, 1).otherwise(0).alias("e_weathersit_" + ty) for ty in weathersit]

mnth = df.select("mnth").distinct().rdd.flatMap(lambda x: x).collect()
mnth_expr = [F.when(F.col("mnth") == ty, 1).otherwise(0).alias("e_mnth_" + ty) for ty in mnth]

weekday = df.select("weekday").distinct().rdd.flatMap(lambda x: x).collect()
weekday_expr = [F.when(F.col("weekday") == ty, 1).otherwise(0).alias("e_weekday_" + ty) for ty in weekday]

# Remove columns that have been changed to binary dummy columns
#
# Dropping extra columns: workingday, atemp. No feature engineering included
df = df.select('cnt',
               'yr',
               'holiday',
               'temp',
               'hum',
               'windspeed', 
               *season_expr+weathersit_expr+mnth_expr+weekday_expr)
display(df)

# COMMAND ----------

from pyspark.ml.linalg import DenseVector

# Seperates the label and the features as label is now stored in the first column
input_data = df.rdd.map(lambda x: (x[0], DenseVector(x[1:])))

# New dataframe consists of a label and a Dense Vector called features
df = spark.createDataFrame(input_data, ["label", "features"])

# COMMAND ----------

# MAGIC %md
# MAGIC Scale variables to zero mean and standard deviation of one.

# COMMAND ----------

from pyspark.ml.feature import StandardScaler

# Remeber features is a now a dense vector containging all feature columns
scaler = StandardScaler(inputCol="features", outputCol="scaled_features",
                        withStd=True, withMean=True)

# Compute summary statistics by fitting the StandardScaler
scalerModel = scaler.fit(df)

# Normalize each feature to have unit standard deviation.
scaledData = scalerModel.transform(df)

scaledData.take(2)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Train test split

# COMMAND ----------

# Split the data into train and test sets
#
# Random split is not a good idea for time series data!
train_data_rs, test_data_rs = scaledData.randomSplit([.8,.2],seed=1234)

# COMMAND ----------

# MAGIC %md
# MAGIC So we should not use randomSplit, for time series it would be good to slice up the dataset based on days......
# MAGIC 
# MAGIC Turns out this is not so straight forward in pySpark as unlike Pandas there are not good ways to slice by index like doing this by array.
# MAGIC 
# MAGIC Turns out its even worse if you want to go from the bottom of the dataset up.
# MAGIC 
# MAGIC > Firstly, you must understand that DataFrames are distributed, that means you can't access them in a typical procedural way, you must do an analysis first.

# COMMAND ----------

# Save data for approximately the last 21 days 

# Array example
# test_data = data[-21:]

# Now remove the test data from the data set 

# Array example
# data = data[:-21]


# Hold out the last 60 days or so of the remaining data as a validation set

# Array examples
# train_features, train_targets = features[:-60], targets[:-60]
# val_features, val_targets = features[-60:], targets[-60:]

# COMMAND ----------

# MAGIC %md
# MAGIC #### Train a model

# COMMAND ----------

# Import `LinearRegression`
from pyspark.ml.regression import LinearRegression

# Initialize `lr`
lr = LinearRegression(labelCol="label", maxIter=10, regParam=0.3, elasticNetParam=0.8)

# Fit the data to the model
linearModel = lr.fit(train_data_rs)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Make predictions

# COMMAND ----------

# Generate predictions
predicted = linearModel.transform(test_data_rs)

# Extract the predictions and the "known" correct labels
predictions = predicted.select("prediction").rdd.map(lambda x: x[0])
labels = predicted.select("label").rdd.map(lambda x: x[0])

# Zip `predictions` and `labels` into a list
predictionAndLabel = predictions.zip(labels).collect()

# Print out first 5 instances of `predictionAndLabel` 
predictionAndLabel[:5]

# COMMAND ----------

# MAGIC %md
# MAGIC #### Evaluating the model

# COMMAND ----------

# Coefficients for the model
print(linearModel.coefficients)

# Intercept for the model
print(linearModel.intercept)

# COMMAND ----------

# Get the RMSE
print("RMSE score :", linearModel.summary.rootMeanSquaredError)

# Get the R2
print("R2 score :", linearModel.summary.r2)

# COMMAND ----------

display(predictionAndLabel)