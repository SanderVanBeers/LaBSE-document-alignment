from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd


def transform_docs_to_tfidf(n:int, all_docs:list) -> list:
    vectorizer = TfidfVectorizer(max_df=.65, min_df=1, stop_words=None, use_idf=True, norm=None)
    transformed_documents = vectorizer.fit_transform(all_docs)
    transformed_documents_as_array = transformed_documents.toarray()
    most_relevant_term_list = list()
    # loop each item in transformed_documents_as_array, using enumerate to keep track of the current position
    for counter, doc in enumerate(transformed_documents_as_array):
        # construct a dataframe
        tf_idf_tuples = list(zip(vectorizer.get_feature_names(), doc))
        one_doc_as_df = pd.DataFrame.from_records(tf_idf_tuples, columns=['term', 'score']).sort_values(by='score', ascending=False).reset_index(drop=True)
        most_relevant_terms = one_doc_as_df['term'].tolist()
        most_relevant_term_list.append(' '.join(most_relevant_terms[:n]))
    return most_relevant_term_list 