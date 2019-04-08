#! /usr/bin/env python3
from pyspark import SparkContext, SparkConf


def remove_test_set(line):
    if line in user_movie_pairs:
        return False
    else:
        return True


conf = SparkConf().setMaster("local[*]").setAppName("MovieSimilarities")
sc = SparkContext(conf=conf)
data = sc.textFile("/Users/arz/Desktop/bigdata-project/ml-1m/ratings_training_20__.dat")
user_movie_pairs = data.sample(False, .0006, 9030).take(10)
data = data.filter(remove_test_set)
count = data.count()
print(count)
data.repartition(1).saveAsTextFile("/Users/arz/Desktop/bigdata-project/ml-1m/ratings_10_test")
sc.stop()
random_pair_file= open("/Users/arz/Desktop/bigdata-project/ml-1m/random_pairs_10",'w')
for item in user_movie_pairs:
    random_pair_file.write(item + "\n")
random_pair_file.close()