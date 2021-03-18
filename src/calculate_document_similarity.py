import torch
import tika
tika.initVM()
from tika import parser
from transformers import AutoTokenizer, AutoModel
import os
import time
import fasttext
from tfidf import transform_docs_to_tfidf
import jsonlines
import logging
import faiss

class LanguageDetector:
    def __init__(self, path_fasttext_model):
        self.fasttext_model = fasttext.load_model(path_fasttext_model)
    def fasttext_detect(self, text):
        text = " ".join(text.split())
        prediction = self.fasttext_model.predict([text])
        label = prediction[0][0][0]
        lang_code = label.replace('__label__', '')
        return lang_code

def create_embedding(sentence, max_length):
    encoded_input = tokenizer([sentence], padding=True, truncation=True, max_length=max_length, return_tensors='pt').to(device)
    with torch.no_grad():
        model_output = model(**encoded_input)

    embeddings = model_output.pooler_output
    embeddings = torch.nn.functional.normalize(embeddings)
    return(embeddings)
    

if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    logging.info(f"Device {device} will be used.")

    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/LaBSE")
    model = AutoModel.from_pretrained("sentence-transformers/LaBSE")
    model.to(device)
    LANGUAGE_DETECTION_MODEL = "/home/sander/Desktop/test_doc_alignment/src/langdetectmodel/lid.176.bin"
    INPUT_DIR_LANG1 = "/home/sander/Desktop/test_doc_alignment/docs/en"
    INPUT_DIR_LANG2 = "/home/sander/Desktop/test_doc_alignment/docs/de"
    LANG1 = "en"
    LANG2 = "de"
    JSONLINES_INTERMEDIATE = "/home/sander/Desktop/test_doc_alignment/docs/text_extracted_files.jsonl"
    USE_TFIDF = False
    NUMBER_OF_TOKENS = 512
    USE_FAISS = True
    RESULTS = f"/home/sander/Desktop/test_doc_alignment/results_test_faiss{NUMBER_OF_TOKENS}_langs_reversed.tsv"
    #languagedetector = LanguageDetector(LANGUAGE_DETECTION_MODEL)
    documents_lang1 = [os.path.join(INPUT_DIR_LANG1, filename) for filename in os.listdir(INPUT_DIR_LANG1)]
    documents_lang2 = [os.path.join(INPUT_DIR_LANG2, filename) for filename in os.listdir(INPUT_DIR_LANG2)]



    if not os.path.isfile(JSONLINES_INTERMEDIATE):
        logging.info("No jsonlines file found. Extracting text.")
        documents_lang1 = [(docpath, parser.from_file(docpath).get('content')) for docpath in documents_lang1]
        documents_lang2 = [(docpath, parser.from_file(docpath).get('content')) for docpath in documents_lang2]
        with jsonlines.open(JSONLINES_INTERMEDIATE, mode='w') as writer:
            for path, content in documents_lang1:
                obj = {
                    "path_document" : path,
                    "content": ' '.join(content.split()),
                    "language": LANG1
                }
                writer.write(obj)
            for path, content in documents_lang2:
                obj = {
                    "path_document" : path,
                    "content": ' '.join(content.split()),
                    "language": LANG2
                }
                writer.write(obj)
        logging.info("Text extraction finished.")
    else:
        logging.info("Existing jsonlines file found. Reading file.")
        with jsonlines.open(JSONLINES_INTERMEDIATE, 'r') as reader:
            documents_lang1 = list()
            documents_lang2 = list()
            for obj in reader:
                language = obj.get('language')
                if language == LANG1:
                    documents_lang1.append((obj.get('path_document'), obj.get('content')))
                elif language == LANG2:
                    documents_lang2.append((obj.get('path_document'), obj.get('content')))

    titles_lang1 = [doc[0] for doc in documents_lang1]
    titles_lang2 = [doc[0] for doc in documents_lang2]

    if USE_TFIDF:
        logging.info("Starting transformation of documents in language 1 to tfidf.")
        documents_lang1_tfidf = transform_docs_to_tfidf(NUMBER_OF_TOKENS, [doc[1] for doc in documents_lang1])
        logging.info("Starting transformation of documents in language 2 to tfidf.")
        documents_lang2_tfidf = transform_docs_to_tfidf(NUMBER_OF_TOKENS, [doc[1] for doc in documents_lang2])
        logging.info("Tfidf transformation complete.")
        logging.info("Creating embeddings for language 1")
        documents_lang1_embeddings = [create_embedding(doc, NUMBER_OF_TOKENS) for doc in documents_lang1_tfidf]
        logging.info("Creating embeddings for language 2")
        documents_lang2_embeddings = [create_embedding(doc, NUMBER_OF_TOKENS) for doc in documents_lang2_tfidf]

    else:
        logging.info("Creating embeddings for language 1")
        documents_lang1_embeddings = [create_embedding(doc[1], NUMBER_OF_TOKENS) for doc in documents_lang1]
        logging.info("Creating embeddings for language 2")
        documents_lang2_embeddings = [create_embedding(doc[1], NUMBER_OF_TOKENS) for doc in documents_lang2]

    if not USE_FAISS:
        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        logging.info("Starting cosine similarity search.")
        with open(RESULTS, 'w') as outputfile:
            for index1, embedding_lang1 in enumerate(documents_lang1_embeddings):
                max_similarity = -1
                index_max_similarity = -1
                for index2, embedding_lang2 in enumerate(documents_lang2_embeddings):
                    cosine_similarity = cos(embedding_lang1, embedding_lang2)
                    if cosine_similarity > max_similarity:
                        max_similarity = cosine_similarity
                        index_max_similarity = index2
                    else:
                        pass
                outputfile.write(f"{titles_lang1[index1]}\t{titles_lang2[index_max_similarity]}\t{max_similarity[0]}\n")
    
    else:
        logging.info("Starting KNN search using FAISS.")
        index = faiss.IndexFlatL2(768)
        #convert list of embeddings to numpy matrix
        documents_lang1_embeddings = torch.cat(documents_lang1_embeddings)
        documents_lang2_embeddings = torch.cat(documents_lang2_embeddings)

        index.add(documents_lang2_embeddings.cpu().numpy())
        #search for K-nearest neighbours
        K = 1
        D,I = index.search(documents_lang1_embeddings.cpu().numpy(), K)
        with open(RESULTS, 'w') as outputfile:
            indexes = I.tolist()
            for index_, value in enumerate(indexes):
                outputfile.write(f"{titles_lang1[index_]}\t{titles_lang2[value[0]]}\t{D.tolist()[index_][0]}\n")
    

