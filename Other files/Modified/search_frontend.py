import re

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


index_body = InvertedIndex.read_index(f"{path}/body_indices", "index")
index_title = InvertedIndex.read_index(f"{path}/title_index", "index")
index_anchor = InvertedIndex.read_index(f"{path}/anchor_index", "index")



files = glob.glob(f"{path}/pr/*.gz")
pr_results = pd.read_csv(*files)

files = glob.glob(f"{path}/processed/processed.pkl")
with open(*files, 'rb') as f:
    pages = pickle.load(f)

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
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    originalquery = tokenizer(query)

    ########################### TITLE ###########################
    query = originalquery
    ids = {}
    for word, pls in index_title.posting_lists_iter():
        for qword in query:
            if qword == word:
                for one_pls in pls:
                    ids[one_pls[0]] = ids.get(one_pls[0], 0) + 1  # ids{id: number_of_apperances}
    titles = ids
    ########################### BODY ###########################
    words, pls = zip(*(index_body.posting_lists_iter()))
    query = [token for token in originalquery if token.lower() not in extended_stopwords]
    query, candidates = get_candidates(words, pls, query)
    if len(query) == 0 or len(candidates) == 0:
        body = []
    else:
      df_tfidfvect, tfidfvectorizer = tf_idf_scores(
          [page[2] for page in pages if page[0] in candidates])
      query_vector = tfidfvectorizer.transform([' '.join(query)])

      cosine_sim_df = cosine_sim_using_sklearn(query_vector, df_tfidfvect)

      top_100_docs = top_N_documents(cosine_sim_df, 100)

      doc_ids = [x[0] for x in top_100_docs[0]];

      body = [([candidate for candidate in candidates][doc_id], weight) for doc_id, weight in top_100_docs[0]]
      body = [b for b in body if b[1] > 0]

    ########################### ANCHOR ###########################
    query = originalquery
    res = []
    if len(query) == 0:
        return jsonify(res)

    ids = {}
    for word, pls in index_anchor.posting_lists_iter():
        for qword in query:
            if qword == word:
                for one_pls in pls:
                    ids[one_pls[0]] = ids.get(one_pls[0], 0) + 1  # ids{id: number_of_apperances}

    anchors = ids
    ###########################***********###########################
    print(f"*************************\n\nTokenized query:\t{query}\n"
          f"Extended tokenized query:\t{[token for token in query if token.lower() not in extended_stopwords]}\n\n"
          f"titles:\n\t{titles}\n\n"
          f"body:\n\t{body}\n\n"
          f"anchors:\n\t{anchors}\n\n")

    w_title = 0.5
    w_body = 0.3
    w_anchor = 0.2
    weighted = {}
    for id, num in titles.items():
      weighted[id] = weighted.get(id, 0) + num * w_title

    for id, num in body:
      weighted[id] = weighted.get(id, 0) + num * w_body
    temp = [x[1] for x in body]
    norm = [(float(i)/max(temp)+1)/2 for i in temp]  # Refine later for usage.

    for id, num in anchors.items():
      weighted[id] = weighted.get(id, 0) + num * w_anchor

    weighted = dict(sorted(weighted.items(), key=lambda item: item[1], reverse=True))

    for page in pages:
      if page[0] in weighted.keys():
          res.append(((page[0], page[1]), weighted[page[0]]))
    print([x for x in sorted(res, key=lambda x: x[1], reverse=True)][:100])
    res = [x[0] for x in sorted(res, key=lambda x: x[1], reverse=True)][:100]

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

    words, pls = zip(*(index_body.posting_lists_iter()))
    tokenized = tokenizer(query)

    tokenized, candidates = get_candidates(words, pls, tokenized)

    if len(tokenized) == 0 or len(candidates) == 0:
        return jsonify(res)
    df_tfidfvect, tfidfvectorizer = tf_idf_scores(
        [page[2] for page in pages if page[0] in candidates])

    query_vector = tfidfvectorizer.transform([' '.join(tokenized)])

    cosine_sim_df = cosine_sim_using_sklearn(query_vector, df_tfidfvect)

    top_100_docs = top_N_documents(cosine_sim_df, 100)
    doc_ids = [x[0] for x in top_100_docs[0]]

    ordered_actual_ids = [[candidate for candidate in candidates][doc_id] for doc_id in doc_ids]

    temp = {}
    candidates = list(candidates)
    for page in pages:
        if page[0] in ordered_actual_ids:
            temp[candidates.index(page[0])] = (page[0], page[1])

    index_map = {v: i for i, v in enumerate(doc_ids)}
    res = [x[1] for x in sorted(temp.items(), key=lambda pair: index_map[pair[0]])]

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
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    newquery = tokenizer(query)
    ids = {}
    for word, pls in index_title.posting_lists_iter():
        for qword in newquery:
            if qword == word:
                for one_pls in pls:
                    ids[one_pls[0]] = ids.get(one_pls[0], 0) + 1  # ids{id: number_of_apperances}

    for id in ids.keys():
        id_in_pages = pages.get(id)
        if id_in_pages:
            res.append(((id, pages[id][0]), ids[id]))
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
    query = np.unique(tokenizer(query))
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION

    ids = {}
    for word, pls in index_anchor.posting_lists_iter():
        for qword in query:
            if qword == word:
                for one_pls in pls:
                    ids[one_pls[0]] = ids.get(one_pls[0], 0) + 1  # ids{id: number_of_apperances}
    for id in ids.keys():
        id_in_pages = pages.get(id)
        if id_in_pages:
            res.append(((id, pages[id][0]), ids[id]))
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
    vectorizer = TfidfVectorizer(stop_words=all_stopwords)
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
    lines = [sorted(list(enumerate(x)), key=lambda y: y[1], reverse=True)[:N] for x in
             df]  # TODO: Make more efficient, we only have on query
    return dict(enumerate(lines))


def get_candidates(words, pls, query_to_search, N=50):
    """
    This function goes over documents and checks for every query token if it appears in words, and counts the number of documents it appears in.
    This function is used to filter both documents and tokens; filters tokens that appeared in less than N documents.

    Parameters
    ----------
    query_to_search: list of tokens (str). This list will be preprocessed in advance.

    words,pls: iterator for working with posting.

    N: integer (default: 50)

    Returns:
    ----------
    filtered_tokens: Tokens that appeared in more than N documents.
    candidate_list: Set of doc_id that had a token in filtered_tokens appear in their body text.
    """

    candidates = {}
    filtered_tokens = []
    candidate_list = set()
    for term in np.unique(query_to_search):
        if term in words:
            pls_list = pls[words.index(term)]  # TODO: Can improve
            for v in pls_list:
                candidates[v] = candidates.get(v, 0) + 1
        if sum(candidates.values()) >= N:
            filtered_tokens.append(term)
            candidate_list.update([key[0] for key in candidates.keys()])
    return filtered_tokens, candidate_list

if __name__ == '__main__':
    # run the Flask RESTful API, make the server publicly available (host='0.0.0.0') on port 8080
    app.run(host='0.0.0.0', port=8080, debug=True)