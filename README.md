# Project JaQUAd QA System
Information retrieval-based question answering (IR QA) system using JaQuAD dataset is proposed.  About the JaQuAD (Japanese Question Answering Dataset), please check their [github](https://github.com/SkelterLabsInc/JaQuAD) and their [paper](https://arxiv.org/abs/2202.01764).

## Introducton
QA system is definitely one of the most important and popular NLP tasks in recent years. However, the existing QA system tutorial are mainly designed for english dataset like [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/) and [WebQA](https://webqna.github.io/), datasets for QA system in other language are rare and the corresponding QA model are accodingly rare as well. It is noted that the purpose of this project is not trying to reach the baseline of the state-of-the-art QA system.  Instead, the real purpose of this project is trying to explore the present language models and NLP tools to implement a japanese QA system.  As you could see in the main and componenets files, the whole QA system comprises three main parts: a document retriever, a paragraph retriever, and a document reader. The whole construction is basically shown as the following figure from the [article](https://qa.fastforwardlabs.com/methods/background/2020/04/28/Intro-to-QA.html). For further explanation about how did I really implement the QA system based on JaQuAd dataset, each of the componenets of QA system is described in the following sections.

![Image](https://qa.fastforwardlabs.com/images/post1/QAworkflow.png "Workflow of a generic IR-based QA system")
*Image by [article](https://qa.fastforwardlabs.com/methods/background/2020/04/28/Intro-to-QA.html)*

## Document Retriever
In the Document Retrieval (DR) part, the question and contexts (Japanese sentence) are initially tokenized by using spacy module. Furthermore, the stopword list was also used to filter the unnecessary stopwords in the sentences. After receiving the tokenized sentences, the TF-IDF and cosine similarity are used to search the top-5 most similar tokenized contexts relative to the tokenized question. The result is excellent, even though we used conventional TF-IDF instead of other advanced tricks such as bm25 or [Dense Passage Retrieval (DPR)](https://arxiv.org/abs/2004.04906), the accuracy (to the train dataset) is high as 99.95% as shown in the file train document retrieval.ipynb.  

## Paragraph Retriever
For the paragraph retriever, we split the contexts obtained from the document retriever into plural paragraphs, and used sentence transformer to embed each paragraph. In this process, the key is choosing appropriate sentence transformer to embed the question and paragraph, since the result of the similarity computataion hugely depends on the quality and performance of the sentence transformer model. In this part, the sentence transformer model "paraphrase-multilingual-mpnet-base-v2" is used. For detailed description of paraphrase-multilingual-mpnet-base-v2, please visit their [model card](https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2) in huggingface. 

After obtaining the embediing vector of each paragraph, cosine similarity is used again to compute the similarity between the question and paragraphs. Based on the observation, top-5 most sililar paragraphs are picked for next process. The concept idea could be seen from the [paper](https://arxiv.org/abs/1908.10084). 

## Document Reader

## Future improvement
