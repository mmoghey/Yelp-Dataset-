import string
import re
import findspark
findspark.init()
from pyspark.sql.functions import udf
from pyspark.sql.types import IntegerType
from pyspark.sql.types import StringType
from pyspark.sql import SQLContext
from pyspark.sql import functions

from pyspark import SparkContext as sc
import pandas as pd
import os
import numpy as np


import json
import pyspark
from pyspark.sql import SparkSession
from pyspark.ml.feature import *
from pyspark.ml import Pipeline
from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import NGram, VectorAssembler
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# to remove the punctuations from the text
def remove_punct(text):
    regex = re.compile('[' + re.escape(string.punctuation) + '0-9\\r\\t\\n]')
    nopunct = regex.sub(" ", text)  
    return nopunct

  
# start the spark context. remove .master("local[*]) for the cluster mode"
sqlContext = SparkSession.builder.master("local[*]").getOrCreate()

# load the csv file
review  = sqlContext.read.format('com.databricks.spark.csv').options(header='true', inferschema='true').load('data_stop_word_removed.csv')
#review  = sqlContext.read.format('com.databricks.spark.csv').options(header='true', inferschema='true').load('review.csv')


type (review)
review.show(5)

# drop the reviews which are not available
review = review.dropna()
print (review.count())

# split the dataset into train and test 80% - train and 20% test
(train_df, val_df) = review.randomSplit([0.80, 0.20], seed = 2000)

# get the tokens from the text
tokenizer = Tokenizer(inputCol="text", outputCol="words")

# remove the stop words
remove_sw = StopWordsRemover(inputCol='words', outputCol='words_sr')

# apply the count vectorizer
count_v = CountVectorizer(vocabSize=2**14, inputCol="words_sr", outputCol='cv')

# define the inverse the document frequency
idf = IDF(inputCol='cv', outputCol="features", minDocFreq=5)

# apply string indexer
string_idx = StringIndexer(inputCol = "stars", outputCol = "label", handleInvalid="keep")

# define the model
lr = LogisticRegression(maxIter=50)

# now put everything in the pipeline and train the model
pipeline = Pipeline(stages=[tokenizer, remove_sw,  count_v, idf, string_idx, lr])
pipelineFit = pipeline.fit(train_df)

# get the predictions
predictions = pipelineFit.transform(val_df)

#find the accuracy
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction",
                                              metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Logistic Regression Accuracy Score (unigram)= " + str(accuracy))


# now try different model (Naive Bayes)
nb = NaiveBayes(smoothing=1.0, modelType="multinomial")

# set everything in a pipeline and train
pipeline =  Pipeline(stages=[tokenizer, remove_sw,  count_v, idf, string_idx,nb])
pipelineFit = pipeline.fit(train_df)

# find the predictions
predictions = pipelineFit.transform(val_df)

# compute accuracy on the test set
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction",
                                              metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Naive Bayes Accuracy Score (unigram)= " + str(accuracy))


########################################## BIGRAM ################################
# tokenize the review and remove stop words
tk = Tokenizer(inputCol="text", outputCol="words")
rm_sw = StopWordsRemover(inputCol='words', outputCol='words_sr')

# get the unigram, count vectorize it and calculate idf score
ng1 =  NGram(n=1, inputCol="words", outputCol="1_grams")
cv1 = CountVectorizer(vocabSize=5460,inputCol="1_grams", outputCol="1_tf") 
idf1 = IDF(inputCol="1_tf", outputCol="1_tfidf", minDocFreq=5)

# get the bigram and count vectorize it and calculate the idf score
ng2 =  NGram(n=2, inputCol="words", outputCol="2_grams")
cv2 = CountVectorizer(vocabSize=5460,inputCol="2_grams", outputCol="2_tf") 
idf2 = IDF(inputCol="2_tf", outputCol="2_tfidf", minDocFreq=5)

# assemble unigram and bigram
va = VectorAssembler( inputCols=["1_tfidf", "2_tfidf"], outputCol="features" )
string_idx = StringIndexer(inputCol = "stars", outputCol = "label", handleInvalid="keep")

# set the logistic regression model
lr2 = LogisticRegression(maxIter=100)

# define the pipeline
pipeline = Pipeline(stages=[tk, ng1, cv1, idf1, ng2, cv2, idf2, va, string_idx, lr2])
pipelineFit = pipeline.fit(train_df)

# get the predictions
predictions = pipelineFit.transform(val_df)

# apply multiclass evaluator
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction",
                                              metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print ("Logistic Regression Accuracy Score (bigram): ", accuracy)

# set the pipeline for naive bayes
pipeline = Pipeline(stages=[tk, ng1, cv1, idf1, ng2, cv2, idf2, ng3, cv3, idf3, va, string_idx, nb])
pipelineFit = pipeline.fit(train_df)

# get the predictions
predictions = pipelineFit.transform(val_df)

# compute accuracy on the test set
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction",
                                              metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Naive Bayes Accuracy Score (bigram)= " + str(accuracy))



########################################### TRIGRAM ######################################
# tokenize the words and remove stop words
tk = Tokenizer(inputCol="text", outputCol="words")
rm_sw = StopWordsRemover(inputCol='words', outputCol='words_sr')


# get the unigrams and calculate idf
ng1 =  NGram(n=1, inputCol="words", outputCol="1_grams")
cv1 = CountVectorizer(vocabSize=5460,inputCol="1_grams", outputCol="1_tf") 
idf1 = IDF(inputCol="1_tf", outputCol="1_tfidf", minDocFreq=5)

# get the bigrams and calculate idf
ng2 =  NGram(n=2, inputCol="words", outputCol="2_grams")
cv2 = CountVectorizer(vocabSize=5460,inputCol="2_grams", outputCol="2_tf") 
idf2 = IDF(inputCol="2_tf", outputCol="2_tfidf", minDocFreq=5)

# get the trigrams and calculate idf
ng3 =  NGram(n=3, inputCol="words", outputCol="3_grams")
cv3 = CountVectorizer(vocabSize=5460,inputCol="3_grams", outputCol="3_tf") 
idf3 = IDF(inputCol="3_tf", outputCol="3_tfidf", minDocFreq=5)

# assemble the unigram, bugram and trigram columns
va = VectorAssembler( inputCols=["1_tfidf", "2_tfidf", "3_tfidf"], outputCol="features" )
string_idx = StringIndexer(inputCol = "stars", outputCol = "label", handleInvalid="keep")

# set the logistic regression model
lr2 = LogisticRegression(maxIter=100)

# define the flow pipeline
pipeline = Pipeline(stages=[tk, ng1, cv1, idf1, ng2, cv2, idf2, ng3, cv3, idf3, va, string_idx, lr2])

pipelineFit = pipeline.fit(train_df)
predictions = pipelineFit.transform(val_df)

# set eveluator and calculate accuracy
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction",
                                              metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print ("Logistic Regression Accuracy Score (trigram): ", accuracy)


# define data flow pipeline for naive bayes
pipeline = Pipeline(stages=[tk, ng1, cv1, idf1, ng2, cv2, idf2, ng3, cv3, idf3, va, string_idx, nb])

pipelineFit = pipeline.fit(train_df)
predictions = pipelineFit.transform(val_df)

# compute accuracy on the test set
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction",
                                              metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Naive Bayes Accuracy Score (trigram) = " + str(accuracy))







  
