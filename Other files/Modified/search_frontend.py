import math
import re
from collections import Counter

import nltk
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify

nltk.download('stopwords')

from nltk.corpus import stopwords

import pickle
from inverted_index_gcp import InvertedIndex
import glob



class MyFlaskApp(Flask):
    def run(self, host=None, port=None, debug=None, **options):
        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)


app = MyFlaskApp(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False

bucket_name = "training_index"
path = "/home/nirkor"


index_body = InvertedIndex.read_index(f"{path}/body_index", "index")
index_title = InvertedIndex.read_index(f"{path}/title_index", "index")
index_anchor = InvertedIndex.read_index(f"{path}/anchor_index", "index")


files = glob.glob(f"{path}/pr/*.gz")
pr_results = pd.read_csv(*files)

files = glob.glob(f"{path}/processed/processed.pickle")
with open(*files, 'rb') as f:
    pages = pickle.load(f)  # id: (title, len(text), count(most_frequent_term))

pages_len = len(pages)
d_avg = sum([page[1] for page in pages.values()]) / pages_len  #TODO: Fix
# test = dict(zip(*(index_body.posting_lists_iter())))

# TODO: Enable once we have the file.
# wiki_id_2_pageview = None
# with open(f"{path}/pr/pageviews-202108-user.pkl", 'rb') as f:
#     wiki_id_2_pageview = pickle.loads(f.read())




def tokenize_text(text):
    english_stopwords = frozenset(stopwords.words('english'))
    corpus_stopwords = ["category", "references", "also", "external", "links",
                        "may", "first", "see", "history", "people", "one", "two",
                        "part", "thumb", "including", "second", "following",
                        "many", "however", "would", "became"]

    all_stopwords = english_stopwords.union(corpus_stopwords)
    return [token.group() for token in RE_WORD.finditer(text.lower()) if token.group() not in all_stopwords]


def BM25(doc_id, query):

    tfij = 0
    dfj = 0



    b = 0
    k1 = 0
    k3 = 0

    djsize = pages[doc_id][1]
    N = pages_len
    BM25 = 0
    B = 1 - b + b * (djsize / d_avg)
    for term in query:
        tfij = 0
        tfiq = 0

        BM25 += ((k1 + 1) * tfij)/(B*k1+tfij) * math.log((N + 1)/dfj, 10) * ((k3 + 1) / tfiq / (k3 + tfiq))
    return BM25

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
    extended_stopwords = ['ok', 'their', 'before', 'are', 'now', 'until', 's', 'during', 'between', 'not', 'maybe', 'an', 'any', 'each', 'can', 'by', 'that', 'from', 'myself', 'than', 'also', 'off', 'these', 'they', 'am', 'no', 'will', 'yourself', 'do', 'against', 'out', 'him', 'your', 'whereas', 'once', 'have', 'were', 'down', 'its', 'been', 'after', 'could', 'was', 'what', 'doing', 'under', 'when', 'only', 'herself', 'always', 'be', 'mine', 'about', 'those', 'ourselves', 'our', 'itself', 'then', 'yours', 'in', 'most', 'having', 'we', 'her', 'whose', 'this', 'all', 'themselves', 'again', 'his', 'yet', 'further', 'become', 'whoever', 'of', 'neither', 'almost', 'else', 'them', 'whether', 't', 'although', 'the', 'why', 'to', 'he', 'yes', 'there', 'both', 'so', 'my', 'at', 'had', 'is', 'other', 'below', 'without', 'too', 'actually', 'hence', 'it', 'don', 'while', 'wherever', 'she', 'should', 'such', 'above', 'and', 'some', 'because', 'but', 'would', 'himself', 'with', 'own', 'became', 'on', 'might', 'how', 'few', 'as', 'does', 'may', 'through', 'which', 'very', 'into', 'just', 'a', 'over', 'theirs', 'ours', 'whenever', 'nor', 'here', 'did', 'if', 'up', 'must', 'within', 'for', 'me', 'has', 'where', 'whom', 'who', 'either', 'yourselves', 'you', 'more', 'or', 'being', 'same', 'oh', 'hers', 'i']
    query = request.args.get('query', '')
    query = tokenizer(query)
    query = [token for token in query if token.lower() not in extended_stopwords]
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
    query = tokenizer(query)
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION

    res = get_topN_score_for_queries(query, index_body)
    # END SOLUTION
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
    query = tokenizer(query)
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    ids = {}
    for word, pls in index_title.posting_lists_iter():
        for qword in query:
            if qword == word:
                for one_pls in pls:
                    ids[one_pls[0]] = ids.get(one_pls[0], 0) + 1  # ids{id: number_of_appearances}


    for id_inner in ids.keys():
        id_in_pages = pages.get(id_inner)
        if id_in_pages:
            res.append(((id_inner, pages[id_inner][0]), ids[id_inner]))
    res = [x[0] for x in sorted(res, key=lambda x: x[1], reverse=True)]
    # END SOLUTION
    return jsonify(res)


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
    query = tokenizer(query)
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION

    ids = {}
    for word, pls in index_anchor.posting_lists_iter():
        for qword in query:
            if qword == word:
                for one_pls in pls:
                    ids[one_pls[0]] = ids.get(one_pls[0], 0) + 1  # ids{id: number_of_apperances}
    for id_inner in ids.keys():
        id_in_pages = pages.get(id_inner)
        if id_in_pages:
            res.append(((id_inner, pages[id_inner][0]), ids[id_inner]))
    res = [x[0] for x in sorted(res, key=lambda x: x[1], reverse=True)]

    # END SOLUTION
    return jsonify(res)


@app.route("/get_pagerank", methods=['POST'])
def get_pagerank():  # TODO: Test
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

    pr = pr_results.vertices.select("id", "pagerank")
    for id in wiki_ids:
        filtered_rdd = pr.filter(lambda x: id in x)
        res.append(filtered_rdd.values().collect())

    # END SOLUTION
    return jsonify(res)


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


# Staff-provided 3 tokenizer
english_stopwords = frozenset(stopwords.words('english'))
corpus_stopwords = ["category", "references", "also", "external", "links",
                    "may", "first", "see", "history", "people", "one", "two",
                    "part", "thumb", "including", "second", "following",
                    "many", "however", "would", "became"]

all_stopwords = english_stopwords.union(corpus_stopwords)
RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)


def tokenizer(text):
    tokens = np.unique([token.group() for token in RE_WORD.finditer(text.lower())])
    return tokens




def get_candidate_documents_and_scores(index,words,pls):
    candidates = {}
    for term, pl in zip(words, pls):
        list_of_doc = pl

        normlized_tfidf = [(doc_id, (freq / pages[doc_id][1]) * math.log(pages_len / index.df[term], 10)) for doc_id, freq in list_of_doc]

        for doc_id, tfidf in normlized_tfidf:
            candidates[(doc_id, term)] = candidates.get((doc_id, term), 0) + tfidf

    return candidates


def myread(index, word):  # TODO: Change
    f_name, offset, n_bytes = index.posting_locs[word][0][0], index.posting_locs[word][0][1], index.df[word]*6
    with open(index.iname + "/" + f_name, 'rb') as f:
        mylist = []
        f.seek(offset)
        for i in range(int(n_bytes/6)):
            b = (f.read(6))
            doc_id = int.from_bytes(b[0:4], 'big')
            tf = int.from_bytes(b[4:], 'big')
            mylist.append((doc_id, tf))
    return mylist


def generate_document_tfidf_matrix(query_to_search, index):
    words = []
    pls = []
    for word in query_to_search:
        words.append(word)
        pls.append(myread(index, word))
    total_vocab_size = len(words)
    candidates_scores = get_candidate_documents_and_scores(index, words, pls)
    unique_candidates = np.unique([doc_id for doc_id, freq in candidates_scores.keys()])
    D = np.zeros((len(unique_candidates), total_vocab_size))

    D = pd.DataFrame(D)

    D.index = unique_candidates
    D.columns = words
    # dot product pandas df
    for key in candidates_scores:
        tfidf = candidates_scores[key]
        doc_id, term = key
        D.loc[[doc_id], [term]] = tfidf
    return D

def cosine_similarity(D, Q):

    D = D.dot(Q)  # This is type() = Series, me wanty wanty LINES from it
    #sorted...



    print(f"D:\n{D}\n")



    Dlines = sorted(D, key=lambda x: x[1])
    print(f"Dlines:\n{Dlines}\n")

    dic = {}
    for id, wij in Dlines:
      wijsq = np.sum(np.square(wij))
      cossim = np.dot(Q, wij)/np.sqrt(wijsq * wijsq)
      dic[id] = cossim
    return dic


def generate_query_tfidf_vector(query_to_search, index):
    C = Counter(query_to_search)
    Qvector = [C[word]/index.df[word]/len(query_to_search) for word in query_to_search]
    return Qvector
    '''
    epsilon = .0000001
    total_vocab_size = len(index.term_total)
    Q = np.zeros((total_vocab_size))
    term_vector = list(index.term_total.keys())
    counter = Counter(query_to_search)
    for token in np.unique(query_to_search):
        if token in index.term_total.keys():  # avoid terms that do not appear in the index.
            tf = counter[token] / len(query_to_search)  # term frequency divded by the length of the query
            df = index.df[token]
            idf = math.log(pages_len / (df + epsilon), 10)  # smoothing

            try:
                ind = term_vector.index(token)
                Q[ind] = tf * idf
            except:
                pass
    return Q
    '''


def get_top_n(sim_dict, N=100):
    return sorted([(doc_id,round(score,5)) for doc_id, score in sim_dict.items()], key = lambda x: x[1],reverse=True)[:N]


def get_posting_iter(index):
    words, pls = zip(*index.posting_lists_iter())
    return words,pls


def get_topN_score_for_queries(queries_to_search, index, N=100):
    D = generate_document_tfidf_matrix(queries_to_search, index)
    Q = generate_query_tfidf_vector(queries_to_search, index)



    sim_dict = cosine_similarity(D, Q)
    ranked = get_top_n(sim_dict, N)
    return ranked






if __name__ == '__main__':
    # run the Flask RESTful API, make the server publicly available (host='0.0.0.0') on port 8080
    app.run(host='0.0.0.0', port=8080, debug=True)