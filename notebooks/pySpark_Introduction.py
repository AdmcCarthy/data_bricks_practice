# Databricks notebook source
# MAGIC %md
# MAGIC ### Pyspark introduction
# MAGIC 
# MAGIC Why pyspark instead of Scala. For me, I know some python and will ease of use over performance to start with.
# MAGIC 
# MAGIC > Spark Performance: Scala or Python?
# MAGIC > In general, most developers seem to agree that Scala wins in terms of performance and concurrency: it’s definitely faster than Python when
# MAGIC > you’re working with Spark, and when you’re talking about concurrency, it’s sure that Scala and the Play framework make it easy to write clean
# MAGIC > and performant async code that is easy to reason about. Play is fully asynchronous, which make it possible to have many concurrent
# MAGIC > connections without dealing with threads. It will be easier to make I/O calls in paralllel to improve performance and enables the use of 
# MAGIC > real-time, streaming, and server push technologies.
# MAGIC 
# MAGIC > Note that asynchronous code allows for non-blocking I/O when making calls to remote services. Let’s state it differently with an example:
# MAGIC > when you have two lines of code of which the first one queries a database and the next prints something to the console, synchronous
# MAGIC > programming will wait for the query to finish before printing something. Your program is (momentarily) blocked. If your programming language
# MAGIC > doesn’t support asynchronous programming, you’ll need to make threads to execute lines of code in parallel. Asynchronous programming, on the
# MAGIC > other hand, will already print to the console while the database is being queried. The query will be processed on the background.
# MAGIC 
# MAGIC > In short, the above explains why it’s still strongly recommended to use Scala over Python when you’re working with streaming data, even
# MAGIC > though structured streaming in Spark seems to reduce the gap already.
# MAGIC 
# MAGIC So for streaming Scala is the way forward!
# MAGIC 
# MAGIC > When you’re working with the DataFrame API, there isn’t really much of a difference between Python and Scala, but you do need to be wary of
# MAGIC > User Defined Functions (UDFs), which are less efficient than its Scala equivalents. That’s why you should favor built-in expressions if
# MAGIC > you’re working with Python. When you’re working with Python, also make sure not to pass your data between DataFrame and RDD unnecessarily, as
# MAGIC > the serialization and deserialization of the data transfer is particularly expensive.
# MAGIC 
# MAGIC But for dataframe based operations it should be ok, it just requires learning RDD (data table object in Spark) instead of using pandas functions.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Working with data, RDD, Datasets, Dataframe
# MAGIC 
# MAGIC ##### RDD
# MAGIC 
# MAGIC > RDDs are the building blocks of Spark. It’s the original API that Spark exposed and pretty much all the higher level APIs decompose to RDDs.
# MAGIC > From a developer’s perspective, an RDD is simply a set of Java or Scala objects representing data.
# MAGIC 
# MAGIC ##### Dataframes
# MAGIC 
# MAGIC > Because of the disadvantages that you can experience while working with RDDs, the DataFrame API was conceived: it provides you with a higher
# MAGIC > level abstraction that allows you to use a query language to manipulate the data. This higher level abstraction is a logical plan that
# MAGIC > represents data and a schema. This means that the frontend to interacting with your data is a lot easier! Because the logical plan will be
# MAGIC > converted to a physical plan for execution, you’re actually a lot closer to what you’re doing when you’re working with them rather than how
# MAGIC > you’re trying to do it, because you let Spark figure out the most efficient way to do what you want to do.
# MAGIC 
# MAGIC > Remember though that DataFrames are still built on top of RDDs!
# MAGIC 
# MAGIC > More specifically, the performance improvements are due to two things, which you’ll often come across when you’re reading up DataFrames:
# MAGIC > custom memory management (project Tungsten), which will make sure that your Spark jobs much faster given CPU constraints, and optimized
# MAGIC > execution plans (Catalyst optimizer), of which the logical plan of the DataFrame is a part.
# MAGIC 
# MAGIC ##### Datasets
# MAGIC 
# MAGIC > The only downside to using DataFrames is that you’ve lost compile-time type safety when you work with DataFrames, which makes your code more prone to errors. This is part of the reason why they have moved more to the notion of Datasets: getting back some type safety and the use of lambda functions, which means that you want to go a bit back to the advantage that RDDs has to offer, but you don’t want to lose all the optimalizations that the DataFrames offer.
# MAGIC 
# MAGIC > Note that the Spark DataSets, which are statically typed, don’t really have much of a place in Python.
# MAGIC 
# MAGIC > Note that, since Python has no compile-time type-safety, only the untyped DataFrame API is available. Or, in other words, Spark DataSets are statically typed, while Python is a dynamically typed programming language. That explains why the DataFrames or the untyped API is available when you want to work with Spark in Python. Also, remember that Datasets are built on top of RDDs, just like DataFrames.
# MAGIC 
# MAGIC ##### Pandas vs Spark
# MAGIC 
# MAGIC > DataFrames are often compared to tables in a relational database or a data frame in R or Python: they have a scheme, with column names and types and logic for rows and columns. This mimics the implementation of DataFrames in Pandas!
# MAGIC 
# MAGIC > Note that, even though the Spark, Python and R data frames can be very similar, there are also a lot of differences: as you have read above, Spark DataFrames carry the specific optimalization under the hood and can use distributed memory to handle big data, while Pandas DataFrames and R data frames can only run on one computer.
# MAGIC 
# MAGIC > However, these differences don’t mean that the two of them can’t work together: you can reuse your existing Pandas DataFrames to scale up to larger data sets. If you want to convert your Spark DataFrame to a Pandas DataFrame and you expect the resulting Pandas’s DataFrame to be small, you can use the following lines of code:

# COMMAND ----------

# Note that you do need to make sure that the DataFrame needs to be small enough 
# because all the data is loaded into the driver’s memory!
df.toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Persistence
# MAGIC 
# MAGIC Cahcing and persitence. Read more [here](http://spark.apache.org/docs/1.2.0/programming-guide.html#rdd-persistence).
# MAGIC 
# MAGIC > A couple of use cases for caching or persisting RDDs are the use of iterative algorithms and fast interactive RDD use.
# MAGIC 
# MAGIC Interesting to try on a large dataset and the connect to a BI application like PowerBI or perform quieries on the fly for EDA.
# MAGIC 
# MAGIC > RDDs are divided into partitions: each partition can be considered as an immutable subset of the entire RDD. When you execute your Spark program, each partition gets sent to a worker. This means that each worker operates on the subset of the data. Each worker can cache the data if the RDD needs to be re-iterated: the partitions that it elaborates are stored in memory and will be reused in other actions. As you read in the above paragraph, by persisting, Spark will have faster access to that data partition next time an operation makes use of it.
# MAGIC 
# MAGIC #### Broadcasting variable
# MAGIC 
# MAGIC > Instead of creating a copy of the variable for each machine, you use broadcast variables to send some immutable state once to each worker. Broadcast variables allow the programmer to keep a cached read-only variable in every machine. In short, you use these variables when you want a local copy of a variable.

# COMMAND ----------

SparkContext.broadcast(variable)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Good habbits when working in python
# MAGIC 
# MAGIC * Use Spark Datframes (They will optimize themselves)
# MAGIC * Don’t call collect() on large RDDs. (This can crash the driver as it will run out of memory)
# MAGIC * Reduce your RDD before joining (join is an expensive operation, filter or reduce your data before joining it)
# MAGIC * Avoid groupByKey() on large datasets (as this will do it for all data, consider reduceByKey() instead)
# MAGIC * Broadcase variables can be shared across all works
# MAGIC * Avoid flatmap(), join() and groupBy() Pattern (cogroup() instead)

# COMMAND ----------

# Context is already set in Databricks so no need to import
sc

# COMMAND ----------

# Inspect spark context
print(sc.version)
print(sc.pythonVer)
print(sc.master) #master URl to connect to
print(str(sc.sparkHome))  # Path where Spark is installed on worker nodes
print(str(sc.sparkUser()))  # Retrieve name of the Spark User running SparkContext

print(sc.appName)  # Return application name
print(sc.applicationId)  # Retrieve application ID
print(sc.defaultParallelism)  # Return default level of parallelism
print (sc.defaultMinPartitions)  # Default minimum number of partitions

# COMMAND ----------

# Make some dummy dataset
rdd = sc.parallelize([
  ('a',7),('a',2),('b',2)
])

rdd2 = sc.parallelize([
  ('a',2),('d',1),('b',1)
])

rdd3 = sc.parallelize(range(100))

rdd4 = sc.parallelize([
  ("a",["x","y","z"]),
  ("b",["p", "r"])
])

# COMMAND ----------

# Basic information about an RDD
print("Number of partitions :", rdd.getNumPartitions())  #List the number of partitions

print("Count: ", rdd.count())  # Count RDD instances 3

print("Count RDD instances :", rdd.countByKey())  # Count RDD instances by key

print("Count RDD instances by value :", rdd.countByValue())  # Count RDD instances by value

print("Return key value pairs :", rdd.collectAsMap())  # Return (key,value) pairs as a dictionary

print("Sumo of all elements :", rdd3.sum())  # Sum of RDD elements

print("Is RDD empty :", sc.parallelize([]).isEmpty())  # Check whetere RDD is empty

# COMMAND ----------

# Basic statistics
print("Max :", rdd3.max())
print("Min :", rdd3.min())
print("Mean :", rdd3.mean())
print("S.D. :", rdd3.stdev())
print("Variance :", rdd3.variance())

# COMMAND ----------

# Get the data for a histogram
rdd3.histogram(3)  # Argument is the number of buckets

# COMMAND ----------

# Summary statistics
rdd3.stats()

# COMMAND ----------

rdd.map(lambda x: x+(x[1],x[0])).collect()  # Apply a function to each RDD element

# COMMAND ----------

rdd5 = rdd.flatMap(lambda x: x+(x[1],x[0]))  # Apply a function to each RDD element and flatten the result
rdd5.collect()

# COMMAND ----------

rdd4.flatMapValues(lambda x: x).collect()  # Apply a flatMap function to each (key,value) pair of rdd4 without changing the keys