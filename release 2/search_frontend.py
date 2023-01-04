from inspect import isdatadescriptor
import os
import pickle
import re
from collections import Counter
from pathlib import Path

from flask import Flask, request, jsonify
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict, Counter
import re
import nltk
import pickle
import numpy as np

nltk.download('stopwords')

from nltk.corpus import stopwords
from tqdm import tqdm
import operator
from itertools import islice, count
from contextlib import closing

import json
from io import StringIO
from pathlib import Path
from operator import itemgetter
import pickle
import matplotlib.pyplot as plt
from inverted_index_gcp import InvertedIndex


# os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"
# graphframes_jar = 'https://repos.spark-packages.org/graphframes/graphframes/0.8.2-spark3.2-s_2.12/graphframes-0.8.2-spark3.2-s_2.12.jar'
# spark_jars = '/usr/local/lib/python3.7/dist-packages/pyspark/jars'
# !wget -N -P $spark_jars $graphframes_jar
# import pyspark
# from pyspark.sql import *
# from pyspark import SparkConf

# # Initializing spark context
# # create a spark context and session
# conf = SparkConf().set("spark.ui.port", "4050")
# conf.set("spark.jars.packages", "graphframes:graphframes:0.8.2-spark3.2-s_2.12")
# sc = pyspark.SparkContext(conf=conf)
# sc.addPyFile(str(Path(spark_jars) / Path(graphframes_jar).name))
# spark = SparkSession.builder.getOrCreate()
# spark


class MyFlaskApp(Flask):
    def run(self, host=None, port=None, debug=None, **options):
        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)


app = MyFlaskApp(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False


@app.route("/search")
def search():
    ''' Returns up to a 100 of your best search results for the query. This is
        the place to put forward your best search engine, and you are free to
        implement the retrieval whoever you'd like within the bound of the
        project requirements (efficiency, quality, etc.). That means it is up to
        you to decide on whether to use stemming, remove stopwords, use
        PageRank, query expansion, etc.

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION

    # END SOLUTION
    return jsonify(res)


@app.route("/search_body")
def search_body():
    ''' Returns up to a 100 search results for the query using TFIDF AND COSINE
        SIMILARITY OF THE BODY OF ARTICLES ONLY. DO NOT use stemming. DO USE the
        staff-provided tokenizer from Assignment 3 (GCP part) to do the
        tokenization and remove stopwords.

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search_body?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION

    # TODO: Change later to take from bucket
    index = InvertedIndex.read_index("/content/body_index", "all_words")
    pkl_file = "/content/part15_preprocessed.pkl"
    with open(pkl_file, 'rb') as f:
        pages = pickle.load(f)
    ########################################
    # pages: list of tuples
    # Each tuple is a wiki article with id, title, body, and
    # [(target_article_id, anchor_text), ...].

    df_tfidfvect = []  # [(df_tfidfvect,tfidfvectorizer)]
    tfidfvectorizer = []
    i = 0
    for page in pages:
        try:
            df, vector = tf_idf_scores([page[2]])
            df_tfidfvect.append(df)
            tfidfvectorizer.append(vector)
        except ValueError:
            print(f"ERROR\t\t\tERROR\n\n{pages[i]}\n\nERROR\t\t\tERROR\n\n")
        #   return jsonify(res)
        i += 1

    query_vector = [x.transform(tokenizer(query)) for x in tfidfvectorizer]  # Should be tokenized?
    # cosine_sim_df = cosine_sim_using_sklearn(query,df_tfidfvect)
    # top_100_docs = top_N_documents(cosine_sim_df,1)
    # print(top_100_docs)
    # END SOLUTION
    print("\n\n\t\teyy lmao finito\n\t\talso have a good day <3\n\n")
    return jsonify(res)


@app.route("/search_title")
def search_title():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD
        IN THE TITLE of articles, ordered in descending order of the NUMBER OF
        DISTINCT QUERY WORDS that appear in the title. DO NOT use stemming. DO
        USE the staff-provided tokenizer from Assignment 3 (GCP part) to do the
        tokenization and remove stopwords. For example, a document
        with a title that matches two distinct query words will be ranked before a
        document with a title that matches only one distinct query word,
        regardless of the number of times the term appeared in the title (or
        query).

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_title?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION

    # TODO: Change later to take from bucket
    index = InvertedIndex.read_index("/content/title_index", "all_words")
    pkl_file = "/content/part15_preprocessed.pkl"
    with open(pkl_file, 'rb') as f:
        pages = pickle.load(f)
    ########################################

    newquery = tokenizer(query)
    ids = {}
    for word, pls in index.posting_lists_iter():
        for qword in newquery:
            if qword == word:
                for one_pls in pls:
                    ids[one_pls[0]] = ids.get(one_pls[0], 0) + 1  # ids{id: number_of_apperances}

    res = []
    for page in pages:
        if page[0] in ids.keys():
            res.append(((page[0], page[1]), ids[page[0]]))

    # END SOLUTION
    return jsonify([x[0] for x in sorted(res, key=lambda x: x[1], reverse=True)])


@app.route("/search_anchor")
def search_anchor():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD
        IN THE ANCHOR TEXT of articles, ordered in descending order of the
        NUMBER OF QUERY WORDS that appear in anchor text linking to the page.
        DO NOT use stemming. DO USE the staff-provided tokenizer from Assignment
        3 (GCP part) to do the tokenization and remove stopwords. For example,
        a document with a anchor text that matches two distinct query words will
        be ranked before a document with anchor text that matches only one
        distinct query word, regardless of the number of times the term appeared
        in the anchor text (or query).

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_anchor?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION

    # END SOLUTION
    return jsonify(res)


def get_pagerank():
    ''' Returns PageRank values for a list of provided wiki article IDs.

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pagerank
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pagerank', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of floats:
          list of PageRank scores that correrspond to the provided article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
        return jsonify(res)
    # BEGIN SOLUTION

    pr_rdd = pagerank_of_all()
    for id in wiki_ids:
        filtered_rdd = pr_rdd.filter(lambda x: id in x)
        res.append(filtered_rdd.values())

    # END SOLUTION
    return jsonify(res)

def pagerank_of_all():
    """ Returns the pagerank of all the ids
    :return:
    rdd of id,pagerank
    """
    # We will start by making an RDD similar to what was in assignment 3
    pkl_file = "/content/part15_preprocessed.pkl"
    with open(pkl_file, 'rb') as f:
        pgs = pickle.load(f)
    # Now my pgs are like so:
    # pgs: list of tuples
    # Each tuple is a wiki article with id, title, body, and
    # [(target_article_id, anchor_text), ...]

    # Now we will only take the important part, which is our id/anchor_text
    list_for_rdd = []
    for pg in pgs:
        list_for_rdd += pg[3]

    pages = sc.parallelize(list_for_rdd)
    edges = pages.flatMap(lambda x: ([(x[0], row[0]) for row in x[1]])).distinct()
    vertices = edges.flatMap(lambda x: (Row(x[0]), Row(x[1]))).distinct()
    edgesDF = edges.toDF(['src', 'dst']).repartition(4, 'src')
    verticesDF = vertices.toDF(['id']).repartition(4, 'id')
    g = GraphFrame(verticesDF, edgesDF)
    pr_results = g.pageRank(resetProbability=0.15, maxIter=10)
    pr = pr_results.vertices.select("id", "pagerank")
    pr.repartition(1).write.csv('pr', compression="gzip")
    return pr


@app.route("/get_pageview", methods=['POST'])
def get_pageview():
    ''' Returns the number of page views that each of the provide wiki articles
        had in August 2021.

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pageview
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pageview', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ints:
          list of page view numbers from August 2021 that correrspond to the
          provided list article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
        return jsonify(res)
    # BEGIN SOLUTION

    # END SOLUTION
    return jsonify(res)


if __name__ == '__main__':
    # run the Flask RESTful API, make the server publicly available (host='0.0.0.0') on port 8080
    app.run(host='0.0.0.0', port=8080, debug=True)


def tokenizer(text):
    # Staff-provided 3 tokenizer
    english_stopwords = frozenset(stopwords.words('english'))
    corpus_stopwords = ["category", "references", "also", "external", "links",
                        "may", "first", "see", "history", "people", "one", "two",
                        "part", "thumb", "including", "second", "following",
                        "many", "however", "would", "became"]

    all_stopwords = english_stopwords.union(corpus_stopwords)
    RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)
    tokens = [token.group() for token in RE_WORD.finditer(text.lower())]
    return tokens


def tf_idf_scores(data):  # From assignment 4.
    """
    This function calculates the tfidf for each word in a single document utilizing TfidfVectorizer via sklearn.

    Parameters:
    -----------
      data: list of strings.

    Returns:
    --------
      Two objects as follows:
                                a) DataFrame, documents as rows (i.e., 0,1,2,3, etc'), terms as columns ('bird','bright', etc').
                                b) TfidfVectorizer object.

    """
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(data)

    terms = vectorizer.get_feature_names_out()
    df = pd.DataFrame(data=X.toarray(), columns=terms)

    return df, vectorizer


def cosine_sim_using_sklearn(queries, tfidf):  # From assignment 4.
    """
    In this function you need to utilize the cosine_similarity function from sklearn.
    You need to compute the similarity between the queries and the given documents.
    This function will return a DataFrame in the following shape: (# of queries, # of documents).
    Each value in the DataFrame will represent the cosine_similarity between given query and document.

    Parameters:
    -----------
      queries: sparse matrix represent the queries after transformation of tfidfvectorizer.
      documents: sparse matrix represent the documents.

    Returns:
    --------
      DataFrame: This function will return a DataFrame in the following shape: (# of queries, # of documents).
      Each value in the DataFrame will represent the cosine_similarity between given query and document.
    """
    return cosine_similarity(queries, tfidf)


def top_N_documents(df, N):  # From assignment 4.
    """
    This function sort and filter the top N docuemnts (by score) for each query.

    Parameters
    ----------
    df: DataFrame (queries as rows, documents as columns)
    N: Integer (how many document to retrieve for each query)

    Returns:
    ----------
    top_N: dictionary is the following stracture:
          key - query id.
          value - sorted (according to score) list of pairs lengh of N. Eac pair within the list provide the following information (doc id, score)
    """
    lines = [sorted(list(enumerate(x)), key=lambda y: y[1], reverse=True)[:N] for x in df]
    return dict(enumerate(lines))
