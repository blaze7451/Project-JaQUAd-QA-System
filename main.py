import argparse
from components import doc_retriever, paragraph_retriever, Read_Comprehend
import pandas as pd



def parse_args():
    parser = argparse.ArgumentParser(description="Extractive QA system based on JaQUAd dataset")
    parser.add_argument("--dataset-path", type=str, default="C:\\Python\\Pytorch\\Transformer related\\Project JaQUAd QA System\\data\\JaQuAD_documents.csv", help="JaQUAd Dataset")
    parser.add_argument("--model-path", type=str, default="C:\\Python\\Pytorch\\Transformer related\\Project JaQUAd QA System\\weights\\weight13", help="input model path")
    parser.add_argument("--seq-length", type=int, default=512, help="input tokes length (default: 512)")
    parser.add_argument('--sen-tr-model-path', type=str, default="sentence-transformers/paraphrase-multilingual-mpnet-base-v2", help='sentence transformer model type')
    return parser.parse_args()

args = parse_args()

dataset = pd.read_csv(args.dataset_path)

docs = dataset

labels = dataset["doc_label"]

doc_rt = doc_retriever(docs=docs, labels=labels)

question = "ハリモグラの陰茎は何と類似していますか?"

retrieved_docs = doc_rt.get_docs(question)

pg_retriever = paragraph_retriever(retrieved_docs)

paragraphs = pg_retriever.split_doc()

_, _, similar_pgs = pg_retriever.get_similar_paragraphs(question=question, paragraphs=paragraphs)

merged_pg = pg_retriever.merge_similar_paragraphs(similar_pgs)

rc = Read_Comprehend()

ans_pred, _ = rc.get_answer(context=merged_pg, question=question)
