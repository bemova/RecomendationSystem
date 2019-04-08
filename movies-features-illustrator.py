#! /usr/bin/env python3
import sys
from pyspark import SparkContext, SparkConf
import matplotlib.pyplot as plt
import numpy as np
import argparse


def movies_genres_plot_creator():
    conf = SparkConf().setMaster('local').setAppName('inverted index')
    sc = SparkContext(conf=conf)
    user_data = sc.textFile("/Users/arz/Desktop/bigdata-project/ml-1m/movies.dat")
    user_fields = user_data.map(lambda line: line.split("::"))
    ages = user_fields.flatMap(lambda x: x[2].split("|")).map(lambda x: (x, 1)).reduceByKey(lambda x, y: x + y)\
        .map(lambda x: (x[1], x[0]))\
        .sortByKey(ascending=False)
    y = ages.map(lambda x: x[0]).collect()
    x = ages.map(lambda x: x[1]).collect()
    sc.stop()
    x_axis = np.arange(len(y))
    plt.bar(x_axis, y, align='center', width=0.5, color="blue")
    plt.xticks(x_axis, x, rotation='vertical', fontsize=8)
    y_max = max(y) + 100
    plt.ylim(0, y_max)
    plt.title("Genres Distribution")
    plt.xlabel("Genres Categories")
    plt.ylabel("Value")
    plt.show()


def load_movie_names():
    movie_names = {}
    with open("/Users/arz/Desktop/bigdata-project/ml-1m/movies.dat", encoding="ISO-8859-1") as f:
        for line in f:
            fields = line.split('::')
            try:
                movie_names[int(fields[0])] = str(fields[1])
            except:
                pass
    return movie_names


def most_viewed_movies():
    conf = SparkConf().setMaster("local").setAppName("Most Viewed Movies")
    sc = SparkContext(conf=conf)
    names_dictionary = sc.broadcast(load_movie_names())
    ratings_data = sc.textFile("/Users/arz/Desktop/bigdata-project/ml-1m/ratings.dat").map(lambda line: line.split("::"))
    movies = ratings_data.map(lambda x: (int(x[1]), 1))
    flipped_movie_counts = movies.reduceByKey(lambda x, y: x + y).map(lambda x: (x[1], x[0])).sortByKey(ascending=False)\
        .take(10)

    lst = []
    for movie in flipped_movie_counts:
        movie_rates_sum = ratings_data.map(lambda x: (int(x[1]), int(x[2]))).filter(lambda x: x[0] == movie[1]) \
            .reduceByKey(lambda x, y: x + y).first()
        movie_average_rate = float("%.2f" % (movie_rates_sum[1] / movie[0]))
        lst.append((names_dictionary.value[movie[1]], movie[0], movie_average_rate))
    sc.stop()
    return lst


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
        movies_genres_plot_creator()
    # answer query 2 if user enters query number 2 after '-q'
    elif args.question == 2:
        movies = most_viewed_movies()
        for movie in movies: print(movie)


if __name__ == "__main__":
    main()
