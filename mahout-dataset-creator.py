#! /usr/bin/env python3
from pyspark import SparkContext, SparkConf


def main():
    conf = SparkConf().setMaster('local').setAppName('mahout dataset creator')
    sc = SparkContext(conf=conf)
    files = sc.textFile("/Users/arz/Desktop/bigdata-project/ml-1m/ratings.dat")
    data = files.map(lambda x: x.split("::")).map(lambda x: str(x[0]) + "," + str(x[1]) + "," + str(x[2]))
    data.repartition(1).saveAsTextFile("/Users/arz/Desktop/bigdata-project/ml-1m/mahout-data")
    sc.stop()


if __name__ == "__main__":
    main()
