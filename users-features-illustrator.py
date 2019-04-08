#! /usr/bin/env python3
import sys
from pyspark import SparkContext, SparkConf
import matplotlib.pyplot as plt
import numpy as np
import argparse


def get_users_info():
    conf = SparkConf().setMaster('local').setAppName('user info')
    sc = SparkContext(conf=conf)
    user_data = sc.textFile("/Users/arz/Desktop/bigdata-project/ml-1m/users.dat")
    user_fields = user_data.map(lambda line: line.split("::"))
    num_users = user_fields.map(lambda fields: fields[0]).count()
    num_genders = user_fields.map(lambda fields: fields[1]).distinct().count()
    num_ages_category = user_fields.map(lambda fields: fields[2]).distinct().count()
    num_occupations = user_fields.map(lambda fields: fields[3]).distinct().count()
    num_zipcodes = user_fields.map(lambda fields: fields[4]).distinct().count()
    print("Users: %d, genders: %d, ages: %d, occupations: %d, ZIP codes: %d" % (
        num_users, num_genders, num_ages_category, num_occupations, num_zipcodes)
          )
    sc.stop()
    return num_users, num_genders, num_ages_category, num_occupations, num_zipcodes


def users_age_plot_creator():
    conf = SparkConf().setMaster('local').setAppName('user age distribution')
    sc = SparkContext(conf=conf)
    user_data = sc.textFile("/Users/arz/Desktop/bigdata-project/ml-1m/users.dat")
    user_fields = user_data.map(lambda line: line.split("::"))
    ages = user_fields.map(lambda x: (int(x[2]), 1)).reduceByKey(lambda x, y: x + y).sortByKey()
    x = ages.map(lambda x: x[0]).collect()
    y = ages.map(lambda x: x[1]).collect()
    sc.stop()
    x_axis = np.arange(len(x))
    plt.bar(x_axis, y, align='center', width=0.5, color="green")
    plt.xticks(x_axis, x)
    y_max = max(y) + 100
    plt.ylim(0, y_max)
    plt.title("Age Distribution")
    plt.xlabel("age categories")
    plt.ylabel("value")
    plt.show()


def users_gender_plot_creator():
    conf = SparkConf().setMaster('local').setAppName('user gender distribution')
    sc = SparkContext(conf=conf)
    user_data = sc.textFile("/Users/arz/Desktop/bigdata-project/ml-1m/users.dat")
    user_fields = user_data.map(lambda line: line.split("::"))
    ages = user_fields.map(lambda x: (x[1], 1)).reduceByKey(lambda x, y: x + y)
    x = ages.map(lambda x: x[0]).collect()
    y = ages.map(lambda x: x[1]).collect()
    sc.stop()
    x_axis = np.arange(len(y))
    plt.bar(x_axis, y, align='center', width=0.5, color="blue")
    plt.xticks(x_axis, x)
    y_max = max(y) + 100
    plt.ylim(0, y_max)
    plt.title("Gender Distribution")
    plt.xlabel("gender categories")
    plt.ylabel("value")
    plt.show()


def users_occupations_plot_creator():
    conf = SparkConf().setMaster('local').setAppName('user occupations distribution')
    sc = SparkContext(conf=conf)
    user_data = sc.textFile("/Users/arz/Desktop/bigdata-project/ml-1m/users.dat")
    user_fields = user_data.map(lambda line: line.split("::"))
    ages = user_fields.map(lambda x: (int(x[3]), 1)).reduceByKey(lambda x, y: x + y).map(lambda x: (x[1], x[0]))\
        .sortByKey(ascending=False)
    y = ages.map(lambda x: x[0]).collect()
    x = ages.map(lambda x: x[1]).collect()
    sc.stop()
    lst = ["other", "academic/educator", "artist", "clerical/admin", "college/grad student", "customer service",
           "doctor/health care", "executive/managerial", "farmer", "homemaker", "K-12 student", "lawyer", "programmer",
           "retired", "sales/marketing", "scientist", "self-employed", "technician/engineer", "tradesman/craftsman",
           "unemployed", "writer"]
    x_labels = []
    for element in x:
        x_labels.append(lst[element])
    x_axis = np.arange(len(y))
    plt.bar(x_axis, y, align='center', width=0.5, color="blue")
    plt.xticks(x_axis, x_labels, rotation='vertical', fontsize=8)
    y_max = max(y) + 100
    plt.ylim(0, y_max)
    plt.title("Occupation Distribution")
    plt.xlabel("Occupation categories")
    plt.ylabel("value")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='users features illustrator help:')
    # defines the query number arguments which should be answered, and the query number arguments help message
    parser.add_argument('-q', '--question', type=int,
                        help='you should enter the question or query number.', required=True)
    # parse script argument and find mistakes in calling script and return appropriate messages
    args = parser.parse_args()
    # if the range of query and question number is bigger that 4, application return ERROR messages and shows help.
    if args.question and args.question > 4:
        print("ERROR: question number is in rage of 1 to 4")
        parser.print_help()
        sys.exit(0)
    # answer query 1 if user enters query number 1 after '-q'
    if args.question == 1:
        get_users_info()
    # answer query 2 if user enters query number 2 after '-q'
    elif args.question == 2:
        users_age_plot_creator()
    # answer query 3 if user enters query number 3 after '-q'
    elif args.question == 3:
        users_gender_plot_creator()
    # answer query 4 if user enters query number 4 after '-q'
    elif args.question == 4:
        users_occupations_plot_creator()


if __name__ == "__main__":
    main()
