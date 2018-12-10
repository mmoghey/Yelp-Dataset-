import string
import re
#import pyspark
import findspark
findspark.init()
from pyspark.sql.functions import udf

from pyspark import SparkContext as sc
import pandas as pd
import os
import numpy as np
from pyspark.sql import SQLContext
from pyspark.sql import functions
import json
import pyspark
from pyspark.sql import SparkSession
from pyspark.ml.feature import *


def remove_punct(text):
    regex = re.compile('[' + re.escape(string.punctuation) + '0-9\\r\\t\\n]')
    nopunct = regex.sub(" ", text)  
    return nopunct
  
def convert_rating(rating):
    if rating >=4:
        return 1
    else:
        return 0
      

data_path = ['/content']
Business_filepath = os.sep.join(data_path + ['yelp_academic_dataset_review.json'])

#sc = sc(appName="Yelp")
#sqlContext = SQLContext(sc)
sqlContext = SparkSession.builder.master("local[*]").getOrCreate()


#Load Business data
Review_data = sqlContext.read.json(Business_filepath)
Review_data.show()

#Business = Business_data.select(pyspark.sql.functions.explode(Business_data.categories).alias("category"), Business_data.state,  Business_data.city, Business_data.stars, Business_data.review_count)
Review = Review_data.select(Review_data.review_id,  Review_data.text, Review_data.stars)
#Register as temp table
Review.registerTempTable("Reviews")

#Run the SQL Query
result = sqlContext.sql("SELECT review_id, \
text, \
stars FROM Reviews")

#saving the result in a csv file
result.coalesce(1).write.format('com.databricks.spark.csv').option("header", "true").save('review.csv')