# Project JaQUAd QA System
Information retrieval-based question answering (IR QA) system using JaQuAD dataset is proposed.  About the JaQuAD (Japanese Question Answering Dataset), please check their [github](https://github.com/SkelterLabsInc/JaQuAD) and their [paper](https://arxiv.org/abs/2202.01764).

## Introducton
QA system is definitely one of the most important and popular NLP tasks in recent years. However, the QA system people  It is noted that the purpose of this project is not trying to reach the baseline of the state-of-the-art QA system.  Instead, the real purpose of this project is trying to explore the present language models and NLP tools to implement a japanese QA system.  As you could see in the main and componenets files, the whole QA system comprises three main parts: a document retriever, a paragraph retriever, and a document reader. The whole construction is basically shown as the following figure from the [article](https://qa.fastforwardlabs.com/methods/background/2020/04/28/Intro-to-QA.html). For further explanation about how did I really implement the QA system based on JaQuAd dataset, each of the componenets is described in the following sections.

![Image](https://qa.fastforwardlabs.com/images/post1/QAworkflow.png "Workflow of a generic IR-based QA system")

## Document Retriever
In the Document Retrieval (DR) part, the question (Japanese sentence) is initially tokenized by using spacy module and simplified by 

## Paragraph Retriever

## Document Reader

## Future 
