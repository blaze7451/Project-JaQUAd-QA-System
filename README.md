# Project JaQUAd QA System
Information retrieval-based question answering (IR QA) system using JaQuAD dataset is proposed.  About the JaQuAD (Japanese Question Answering Dataset), please check their [github](https://github.com/SkelterLabsInc/JaQuAD) and their [paper](https://arxiv.org/abs/2202.01764).

## Introducton
QA system is definitely one of the most important and popular NLP tasks in recent years. However, the existing QA system tutorial are mainly designed for english dataset like [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/) and [WebQA](https://webqna.github.io/), datasets for QA system in other language are rare and the corresponding QA model are accodingly rare as well. It is noted that the purpose of this project is not trying to reach the baseline of the state-of-the-art QA system.  Instead, the real purpose of this project is trying to explore the present language models and NLP tools to implement a japanese QA system.  As you could see in the main and componenets files, the whole QA system comprises three main parts: a document retriever, a paragraph retriever, and a document reader. The whole construction is basically shown as the following figure from the [article](https://qa.fastforwardlabs.com/methods/background/2020/04/28/Intro-to-QA.html). For further explanation about how did I really implement the QA system based on JaQuAd dataset, each of the componenets of QA system is described in the following sections.

![Image](https://qa.fastforwardlabs.com/images/post1/QAworkflow.png "Workflow of a generic IR-based QA system")

## Document Retriever
In the Document Retrieval (DR) part, the question and contexts (Japanese sentence) are initially tokenized by using spacy module. Furthermore, the stopword list was also used to filter the unnecessary stopwords in the sentences. After receiving the tokenized sentences, the TF-IDF and cosine similarity are used to search the top-5 most similar tokenized contexts relative to the tokenized question. The result is excellent, even though we used conventional TF-IDF instead of other advanced tricks such as bm25 or [Dense Passage Retrieval (DPR)](https://arxiv.org/abs/2004.04906), the accuracy (to the train dataset) is high as 99.95% as shown in the file train document retrieval.ipynb.  

## Paragraph Retriever

## Document Reader

## Future improvement
