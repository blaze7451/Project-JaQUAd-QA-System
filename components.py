from sentence_transformers import SentenceTransformer, util
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from transformers import BertForQuestionAnswering
from transformers import AutoTokenizer
import argparse
from Fine_tuning.tuning_util import eval_util
from helpers.utils import Preprocessing
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description="Extractive QA system based on JaQUAd dataset")
    parser.add_argument("--dataset-path", type=str, default="C:\\Python\\Pytorch\\Transformer related\\Project JaQUAd QA System\\data\\JaQuAD_all.csv", help="JaQUAd Train + Test Dataset")
    parser.add_argument("--model-path", type=str, default="C:\\Python\\Pytorch\\Transformer related\\Project JaQUAd QA System\\weights\\weight13", help="input model path")
    parser.add_argument("--seq-length", type=int, default=512, help="input tokes length (default: 512)")
    parser.add_argument('--sen-tr-model-path', type=str, default="sentence-transformers/paraphrase-multilingual-mpnet-base-v2", help='sentence transformer model type')
    parser.add_argument('--stopword-path', type=str, default="C:\\Python\\Pytorch\\Transformer related\\Project JaQUAd QA System\\Stopwords\\stopwords-ja.txt", help='stopwords file path')
    return parser.parse_args()

args = parse_args()

preprocess = Preprocessing()

stopwords_list = preprocess.get_stopwords(args.stopword_path)

class doc_retriever():
    def __init__(self, docs, labels):  #docs and labels should be list or pd.Series of tokenized_context
        self.tokenized_docs = docs["tokenized_context"]
        self.original_docs = docs["context"]
        self.labels = labels
        tfidf_configs = {
            'token_pattern': r"(?u)\b\w\w+\b",
            'analyzer': 'word',
            'binary': True
        }
        retriever_configs = {
            'n_neighbors': 5,
            'metric': 'cosine'
        }

        self.embedding = TfidfVectorizer(**tfidf_configs)
        self.retriever = NearestNeighbors(**retriever_configs)
    
        self.embedding.fit_transform(self.tokenized_docs)
        self.X = self.embedding.fit_transform(self.tokenized_docs)
        self.retriever.fit(self.X, self.labels)

    def transform_text(self, text:str):
        print('Text:', text)
        vector = self.embedding.transform([text])
        vector = self.embedding.inverse_transform(vector)
        print('Vect:', vector)
    
    def get_docs(self, question:str):
        tokenized_question = preprocess.get_tokenized_sentence(question, stopwords_list) 
        question_embedding = self.embedding.transform([tokenized_question])
        doc_label = self.retriever.kneighbors(question_embedding, return_distance=False)[0]
        selected_context = self.original_docs[doc_label].reset_index()
        contexts = []
        for i in selected_context["context"]:
            contexts.append(i)
        return contexts

class paragraph_retriever():
    def __init__(self, docs:list):
        self.docs = docs
        self.encoder = SentenceTransformer(args.sen_tr_model_path)

    def split_doc(self):
        article = " ".join(self.docs)
        paragraphs = article.split( )

        return paragraphs

    def get_similar_paragraphs(self, question:str, paragraphs:list):    
        
        tokenized_q = preprocess.get_tokenized_sentence(question, stopwords_list)
        self.question = [tokenized_q]
        tokenized_p_list = []
        for i in paragraphs:
            tokenized_p = preprocess.get_tokenized_sentence(i, stopwords_list)
            tokenized_p_list.append(tokenized_p)
        q_embedding = self.encoder.encode([question], convert_to_tensor=True)
        p_embedding = self.encoder.encode(paragraphs, convert_to_tensor=True)
        cosine_scores = util.cos_sim(q_embedding, p_embedding)
        values, indices = torch.topk(cosine_scores, 5)
        indices = indices.tolist()[0]
        values = values.tolist()[0]
        similar_paragraphs = [paragraphs[i] for i in indices]
        return indices, values, similar_paragraphs

    def merge_similar_paragraphs(self, paragraphs:list):
        doc = "".join(paragraphs)
        doc = [doc]
        
        return doc

class Read_Comprehend():
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        self.model = BertForQuestionAnswering.from_pretrained(args.model_path)
    
    def get_answer(self, context:list, question:str):
        self.context = context
        self.question = [question]
        
        encoding = self.tokenizer(self.context, self.question, truncation="only_first", max_length=512, stride=50, padding = True, return_tensors='pt')

        input_ids = encoding['input_ids']

        attention_mask = encoding['attention_mask']

        outputs = self.model(input_ids, attention_mask=attention_mask)

        start_pred = torch.argmax(outputs['start_logits'], dim=1).item()
        end_pred = torch.argmax(outputs['end_logits'], dim=1).item() + 1

        ans_token = self.tokenizer.decode(encoding["input_ids"].tolist()[0]).split( )[start_pred:end_pred]

        ans_pred = "".join(ans_token)

        return ans_pred, ans_token

    def get_score(self, ans_pred, ans_token, answer:str):
        self.answer = [answer]
        true_ans_token = self.tokenizer.tokenize(self.answer)        
        eval_tool = eval_util()
        em_score = eval_tool.EM([ans_pred], self.answer)
        f1_score = eval_tool.compute_f1(ans_token, true_ans_token)
        return em_score, f1_score