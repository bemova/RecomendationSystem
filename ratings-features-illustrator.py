#! /usr/bin/env python3
import sys
from pyspark import SparkContext, SparkConf
import matplotlib.pyplot as plt
import numpy as np
import argparse


def stars_mapper(data):
    if data[0] == 1:
        return '1 Star', data[1]
    else:
        return str(data[0]) + " Stars", data[1]


def ratings_rates_plot_creator():
    conf = SparkConf().setMaster('local').setAppName('rates distribution')
    sc = SparkContext(conf=conf)
    user_data = sc.textFile("/Users/arz/Desktop/bigdata-project/ml-1m/ratings.dat")
    user_fields = user_data.map(lambda line: line.split("::"))
    ages = user_fields.map(lambda x: (int(x[2]), 1)).reduceByKey(lambda x, y: x + y).sortByKey().map(stars_mapper)
    x = ages.map(lambda x: x[0]).collect()
    y = ages.map(lambda x: x[1]).collect()
    sc.stop()
    x_axis = np.arange(len(y))
    plt.bar(x_axis, y, align='center', width=0.5, color="blue")
    plt.xticks(x_axis, x, rotation='vertical', fontsize=8)
    y_max = max(y) + 100
    plt.ylim(0, y_max)
    plt.title("Rates Distribution")
    plt.xlabel("Rate Categories")
    plt.ylabel("Value")
    plt.show()


def ratings_statistical_info():
    conf = SparkConf().setMaster('local').setAppName('rating statistical information')
    sc = SparkContext(conf=conf)
    ratings_data = sc.textFile("/Users/arz/Desktop/bigdata-project/ml-1m/ratings.dat")
    rates_data = ratings_data.map(lambda line: line.split("::"))
    rates = rates_data.map(lambda fields: int(fields[2]))
    rates_count = rates.count()
    print("rate count: {}".format(rates_count))
    mean_rating = rates.reduce(lambda x, y: x + y) / rates_count
    median_rating = np.median(rates.collect())
    sc.stop()
    print("mean: {:.4f}".format(mean_rating))
    print("median: {:.4f}".format(median_rating))


def main():
    parser = argparse.ArgumentParser(description='users features illustrator help:')
    # defines the query number arguments which should be answered, and the query number arguments help message
    parser.add_argument('-q', '--question', type=int,
                        help='you should enter the question or query number.', required=True)
    # parse script argument and find mistakes in calling script and return appropriate messages
    args = parser.parse_args()
    # if the range of query and question number is bigger that 2, application return ERROR messages and shows help.
    if args.question and args.question > 2:
        print("ERROR: question number is in rage of 1 to 2")
        parser.print_help()
        sys.exit(0)
    # answer query 1 if user enters query number 1 after '-q'
    if args.question == 1:
        ratings_rates_plot_creator()
    # answer query 2 if user enters query number 2 after '-q'
    elif args.question == 2:
        ratings_statistical_info()


if __name__ == "__main__":
    main()
