// Databricks notebook source
spark

// COMMAND ----------

sqlContext

// COMMAND ----------

sc

// COMMAND ----------

1+1

// COMMAND ----------

// MAGIC %md
// MAGIC ### Use different languages in different cells

// COMMAND ----------

// MAGIC %python
// MAGIC def function():
// MAGIC   return 1 + 1
// MAGIC 
// MAGIC function()

// COMMAND ----------

// MAGIC %r
// MAGIC a <- 1
// MAGIC b <- 2
// MAGIC c <- a + b
// MAGIC c

// COMMAND ----------

// MAGIC %sh
// MAGIC pwd
// MAGIC mkdir admc
// MAGIC cd admc
// MAGIC pwd

// COMMAND ----------

// MAGIC %sql SELECT 1

// COMMAND ----------

// MAGIC %md
// MAGIC ### Write to data bricks file structure

// COMMAND ----------

// MAGIC %python
// MAGIC # write a file to DBFS using python i/o apis
// MAGIC with open("/dbfs/tmp/test_dbfs.txt", 'w') as f:
// MAGIC   f.write("Apache Spark is awesome!\n")
// MAGIC   f.write("End of example!")
// MAGIC 
// MAGIC # read the file
// MAGIC with open("/dbfs/tmp/test_dbfs.txt", "r") as f_read:
// MAGIC   for line in f_read:
// MAGIC     print(line)

// COMMAND ----------

// scala
import scala.io.Source

val filename = "/dbfs/tmp/test_dbfs.txt"
for (line <- Source.fromFile(filename).getLines()) {
  println(line)
}

// COMMAND ----------

// MAGIC %md
// MAGIC #### Running functions or getting variables from other notebooks

// COMMAND ----------

// MAGIC %run ./extra

// COMMAND ----------

// MAGIC %python
// MAGIC # Taken from the extra notebook which was run above
// MAGIC print(extra_x)
// MAGIC 
// MAGIC # Function taken from the extra notebook which was run above
// MAGIC print(extra_function(5))

// COMMAND ----------

// MAGIC %md
// MAGIC #### Package cells to share variables across all notebooks on a cluster
// MAGIC 
// MAGIC Have not got this working yet, check out
// MAGIC [Package Cells](https://docs.databricks.com/user-guide/notebooks/package-cells.html)

// COMMAND ----------

package com.databricks.example

case class TestKey(id: Long, str: String)
