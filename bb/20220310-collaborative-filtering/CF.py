import json
import math
import os
import random
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


class UserCFRecTrad:
    def __init__(self,datafile,userSimPath):
        self.datafile = datafile
        self.userSimPath = userSimPath
        self.data = self.loadData()

        self.trainData,self.testData = self.splitData(3,47)  # 训练集与数据集
        self.users_sim = self.UserSimilarityBest()

    def loadData(self):
        print("加载数据...")
        rating = pd.read_csv(self.datafile)
        rating
        train_dict = rating.iloc[:,:3].to_dict('split')
        data = train_dict['data']
        data
        return data

    def splitData(self,k,seed,M=9):
        print("训练数据集与测试数据集切分...")
        train,test = {},{}
        random.seed(seed)
        for user,item,record in self.data:
            if random.randint(0,M) == k:
                test.setdefault(user,{})
                test[user][item] = record
            else:
                train.setdefault(user,{})
                train[user][item] = record
        return train,test

    def UserSimilarityBest(self):
        print("开始计算用户之间的相似度 ...")
        # 得到每个item被哪些user评价过
        item_users = dict()
        for u, items in self.trainData.items():
            for i in items.keys():
                item_users.setdefault(i,set())
                if self.trainData[u][i] > 0:
                    item_users[i].add(u)
        # 构建倒排表
        count = dict()
        user_item_count = dict()
        for i, users in item_users.items():
            for u in users:
                user_item_count.setdefault(u,0)
                user_item_count[u] += 1
                count.setdefault(u,{})
                for v in users:
                    count[u].setdefault(v, 0)
                    if u == v:
                        continue
                    count[u][v] += 1
                    # count[u][v] += 1 / math.log(1+len(users))
        # 构建相似度矩阵
        userSim = dict()
        for u, related_users in count.items():
            userSim.setdefault(u,{})
            for v, cuv in related_users.items():
                if u==v:
                    continue
                userSim[u].setdefault(v, 0.0)
                userSim[u][v] = cuv / math.sqrt(user_item_count[u] * user_item_count[v])
        joblib.dump(userSim, self.userSimPath)
        return userSim

    def recommend(self, user, k=8, nitems=40):
        result = dict()
        have_score_items = self.trainData.get(user, {})
        for v, wuv in sorted(self.users_sim[user].items(), key=lambda x: x[1], reverse=True)[0:k]:
            for i, rvi in self.trainData[v].items():
                if i in have_score_items:
                    continue
                result.setdefault(i, 0)
                result[i] += wuv * rvi
        return dict(sorted(result.items(), key=lambda x: x[1], reverse=True)[0:nitems])

    def metrics(self, k=8, nitems=10):
        print("开始计算准确率 ...")
        hit = 0
        precision = 0
        recall = 0
        for user in self.trainData.keys():
            tu = self.testData.get(user, {})
            rank = self.recommend(user, k=k, nitems=nitems)
            for item, rate in rank.items():
                if item in tu:
                    hit += 1
            precision += nitems
            recall += len(tu)
        return hit / (precision * 1.0), hit / (recall + 1.0)


import json
import math
import os
import random
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt


class UserCFRecKmeans:
    def __init__(self,datafile, userSimPath, n_cluster):
        self.datafile = datafile
        self.userSimPath = userSimPath
        self.n_cluster = n_cluster
        self.data = self.loadData()

        self.trainData,self.testData = self.splitData(3,47)  # 训练集与数据集
        self.user_group = self.cluster_user(self.n_cluster)
        self.users_sim = self.UserSimilarityBest(self.userSimPath, self.user_group)

    def loadData(self):
        print("加载数据...")
        rating = pd.read_csv(self.datafile)
        train_dict = rating.iloc[:,:3].to_dict('split')
        data = train_dict['data']
        return data

    def splitData(self,k,seed,M=9):
        print("训练数据集与测试数据集切分...")
        train,test = {},{}
        random.seed(seed)
        for user,item,record in self.data:
            if random.randint(0,M) == k:
                test.setdefault(user,{})
                test[user][item] = record
            else:
                train.setdefault(user,{})
                train[user][item] = record
        return train,test

    def get_genre_ratings(self, ratings, movies, genres, column_names):
        genre_ratings = pd.DataFrame()
        for genre in genres:
            genre_movies = movies[movies['genres'].str.contains(genre) ]
            avg_genre_votes_per_user = ratings[ratings['movieId'].isin(genre_movies['movieId'])].loc[:, ['userId', 'rating']].groupby(['userId'])['rating'].mean().round(2)

            genre_ratings = pd.concat([genre_ratings, avg_genre_votes_per_user], axis=1)

        genre_ratings.columns = column_names
        return genre_ratings

    def cluster_user(self, n_clusters):
        ratings = pd.read_csv('ml-latest-small/ratings.csv')
        movies = pd.read_csv('ml-latest-small/movies.csv')
        genre_list = [
            'Action',
            'Adventure',
            'Animation',
            "Children's",
            'Romance',
            'Sci-Fi',
            'Comedy',
            'Crime',
            'Documentary',
            'Drama',
            'Fantasy',
            'Film-Noir',
            'Horror',
            'Musical',
            'Mystery',
            'Thriller',
            'War',
            'Western'
        ]
        avg_name = ['avg_' + i + '_rating' for i in genre_list]
        genre_ratings = self.get_genre_ratings(ratings, movies, genre_list, avg_name)
        genre_ratings = genre_ratings.fillna(float(0))
        kmeans = KMeans(n_clusters=n_clusters)
        predictions = kmeans.fit_predict(genre_ratings.values)
        user_group = dict(enumerate(predictions.flatten(), 1))

        return user_group

    def UserSimilarityBest(self, userSimPath, user_group):
        print("开始计算用户之间的相似度 ...")
        # 得到每个item被哪些user评价过
        item_users = dict()
        for u, items in self.trainData.items():
            for i in items.keys():
                item_users.setdefault(i,set())
                if self.trainData[u][i] > 0:
                    item_users[i].add(u)
        # 构建倒排表
        count = dict()
        user_item_count = dict()
        for i, users in item_users.items():
            for u in users:
                user_item_count.setdefault(u,0)
                user_item_count[u] += 1
                count.setdefault(u,{})
                for v in users:
                    count[u].setdefault(v, 0)
                    if u == v :
                        continue
                    count[u][v] += 1
                    # count[u][v] += 1 / math.log(1+len(users))
        # 构建相似度矩阵
        userSim = dict()
        for u, related_users in count.items():
            userSim.setdefault(u,{})
            for v, cuv in related_users.items():
                if u==v or user_group[u] != user_group[v]:
                    continue
                userSim[u].setdefault(v, 0.0)
                userSim[u][v] = cuv / math.sqrt(user_item_count[u] * user_item_count[v])
        joblib.dump(userSim, userSimPath)
        return userSim

    def recommend(self, user, k=8, nitems=40):
        result = dict()
        have_score_items = self.trainData.get(user, {})
        for v, wuv in sorted(self.users_sim[user].items(), key=lambda x: x[1], reverse=True)[0:k]:
            for i, rvi in self.trainData[v].items():
                if i in have_score_items:
                    continue
                result.setdefault(i, 0)
                result[i] += wuv * rvi
        return dict(sorted(result.items(), key=lambda x: x[1], reverse=True)[0:nitems])

    def metrics(self, k=8, nitems=10):
        print("开始计算准确率 ...")
        hit = 0
        precision = 0
        recall = 0
        for user in self.trainData.keys():
            tu = self.testData.get(user, {})
            rank = self.recommend(user, k=k, nitems=nitems)
            for item, rate in rank.items():
                if item in tu and tu[item] > 3:
                    hit += 1
            precision += nitems
            recall += len(tu)
        return hit / (precision * 1.0), hit / (recall + 1.0)