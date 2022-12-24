import spacy
nlp=spacy.load("ja_core_news_sm")


class Preprocessing():
    def get_stopwords(self, filepath):
        self.filepath = filepath
        stopwords_list = []
        with open(self.filepath, encoding="utf-8") as file:
            filedata = file.readlines()
            for word in filedata:
                stopwords_list.append(word.replace("\n", ""))
        stopwords_list = stopwords_list

        return stopwords_list

    def get_tokenized_sentence(self, context, stopwords_list):
        self.stopwords_list = stopwords_list
        self.context = context
        doc = nlp(context)
        token_list=[str(tokens) for tokens in doc if str(tokens) not in stopwords_list]
        tokenized_sentence = " ".join(token for token in token_list)
        return tokenized_sentence

class tools():
    def top_accuracy(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred
        right, count = 0, 0
        for y_t in self.y_true:
            count += 1
            if y_t in self.y_pred:
                right += 1
        return right / count if count > 0 else 0