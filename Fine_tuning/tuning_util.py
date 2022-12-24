from transformers import AutoTokenizer
import pandas as pd
import spacy_alignments as tokenizations
import torch
import string, re
from evaluate import load

class data_processing():

    def __init__(self, path:str):
        self.path = path
        self.word_tokenizer = AutoTokenizer.from_pretrained("cl-tohoku/bert-base-japanese")
        self.char_tokenizer = AutoTokenizer.from_pretrained("cl-tohoku/bert-base-japanese-char")

    def read_data(self):  
        dataset = pd.read_csv(self.path)
        contexts = []
        questions = []
        answers = []
        ans_starts = []
        
        for context in dataset["context"]:
            contexts.append(context)
        
        for question in dataset["question"]:
            questions.append(question)
        
        for i, ans_info in enumerate(dataset["answers"]):
            ans_idx_list = [198, 823, 825, 833, 835, 1876, 2375, 3066, 3125, 3131, 11094, 16510, 18047, 19122, 19125, 26387, 31113, 31206] #special indices need to be handled
            if i in ans_idx_list:
                answer = ans_info.split( )[1].replace("[\"", "").replace("\"],", "")
                answers.append(answer)
            else:
                answer = ans_info.split( )[1].replace("['", "").replace("'],", "")
                answers.append(answer)
            ans_start = ans_info.split( )[3].replace("[", "").replace("],", "") #This is char start position
            ans_start = int(ans_start)
            ans_starts.append(ans_start)
        
        return contexts, questions, answers, ans_starts

    def add_ans_end(self, answers:list, contexts:list, ans_starts:list): #char end position
        ans_ends = []

        for answer, context, ans_start in zip(answers, contexts, ans_starts):
            ans_end = ans_start + len(answer)-1

            # sometimes squad answers are off by a character or two
            if context[ans_end] == answer:
                ans_ends.append(ans_end)
            elif context[ans_start:ans_end+1] == answer:
                ans_ends.append(ans_end)
            elif context[ans_start-1:ans_end] == answer:
                answer['answer_start'] = ans_start -1
                ans_ends.append(ans_end-1)  # When the answer label is off by one character
            elif context[ans_start-2:ans_end-1] == answer:
                answer['answer_start'] = ans_start - 2
                ans_ends.append(ans_end-2)  # When the answer label is off by two characters
            else:
                ans_ends.append(0)
        return ans_ends

class search():
    def __init__(self, context_list:list):
        self.context_list = context_list
        self.word_tokenizer = AutoTokenizer.from_pretrained("cl-tohoku/bert-base-japanese")
        self.char_tokenizer = AutoTokenizer.from_pretrained("cl-tohoku/bert-base-japanese-char")
        self.char2token_list = []

    def char_to_token(self):
        w_list_collection = []
        c_list_collection = []
        for i, context in enumerate(self.context_list):
            word_encoding = self.word_tokenizer(self.context_list[i])["input_ids"]
            char_encoding = self.char_tokenizer(self.context_list[i])["input_ids"]
            w_decoding = self.word_tokenizer.decode(word_encoding).split( )[:-1]
            c_decoding = self.char_tokenizer.decode(char_encoding).split( )[:-1]
            w_list = [item for item in w_decoding]
            c_list = list(context)
            c_list.insert(0, '[CLS]')
            a2b, b2a = tokenizations.get_alignments(w_list, c_list)
            for i in b2a:
                if i == []:
                    i.append(None)
            char2token = sum(b2a, [])
            char2token.pop(0)
            w_list.append("[SEP]")
            c_list.append("[SEP]")
            self.char2token_list.append(char2token)
            w_list_collection.append(w_list)
            c_list_collection.append(c_list)
        return self.char2token_list, w_list_collection, c_list_collection

    def ans_start_end_token_index(self, ans_start_char_index:list, ans_end_char_index:list):
        self.ans_start_char_index = ans_start_char_index
        self.ans_end_char_index = ans_end_char_index
        ans_start_token_index_collection = []
        ans_end_token_index_collection = []
        for i, char2token in enumerate(self.char2token_list):
            ans_start_token_index = char2token[self.ans_start_char_index[i]]
            ans_end_token_index = char2token[self.ans_end_char_index[i]]
            if (ans_start_token_index == None) or (ans_end_token_index == None):
                ans_start_token_index_collection.append(0)
                ans_end_token_index_collection.append(0)
            else:
                ans_start_token_index_collection.append(ans_start_token_index)
                ans_end_token_index_collection.append(ans_end_token_index)
        return ans_start_token_index_collection, ans_end_token_index_collection
        

class cq_pair_search():
    def __init__(self, encoding):
        self.encoding = encoding
        self.token_type_ids = self.encoding["token_type_ids"]
    def find_context_start_end_index(self):
        """
        returns the token index in whih context starts and ends
        """
        context_token_start = [0]*len(self.token_type_ids)
        context_token_end = []
        for i, portion in enumerate(self.token_type_ids):
            token_idx = 0
            while portion[token_idx] != 1:  #means its special tokens or tokens of context
                token_idx += 1                   # loop only break when question starts in tokens
            context_end_idx = token_idx-1
            context_token_end.append(context_end_idx)
        self.context_token_start = context_token_start
        self.context_token_end = context_token_end

        return context_token_start, context_token_end

    def find_ans_start_end_token_index(self, ans_start_token_index_collection:list, ans_end_token_index_collection:list):
        self.ans_start_token_index_collection = ans_start_token_index_collection
        self.ans_end_token_index_collection = ans_end_token_index_collection
        context_start_token_index = self.context_token_start
        context_end_token_index = self.context_token_end
        cqans_start_tok_index_collection = []
        cqans_end_tok_index_collection = []
        for i in range(len(self.ans_start_token_index_collection)):
            if (self.ans_start_token_index_collection[i] > context_end_token_index[i]-1) or (self.ans_end_token_index_collection[i] > context_end_token_index[i]-1):
                cqans_start_tok_index_collection.append(0)
                cqans_end_tok_index_collection.append(0)                
            else:
                cqans_start_tok_index_collection.append(self.ans_start_token_index_collection[i])
                cqans_end_tok_index_collection.append(self.ans_end_token_index_collection[i])
        
        return cqans_start_tok_index_collection, cqans_end_tok_index_collection
        
class get_answer():
    def __init__(self, model, tokenizer):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model = model
        self.tokenizer = tokenizer

    def get_prediction(self, context, question):
        inputs = self.tokenizer.encode_plus(question, context, return_tensors='pt').to(self.device)
        outputs = self.model(**inputs)
  
        answer_start = torch.argmax(outputs[0])  
        answer_end = torch.argmax(outputs[1]) + 1 
  
        answer = self.tokenizer.decode(outputs["input_ids"][0][answer_start:answer_end])
  
        return answer

class eval_util():
    def __init__(self):
        pass
    def EM(self, pred:list, ref:list):
        exact_match_metric = load("exact_match")
        results = exact_match_metric.compute(predictions=pred, references=ref)
        return results
    
    def compute_f1(self, pred_tokens, true_tokens):
        self.pred_tokens = pred_tokens
        self.true_tokens = true_tokens
  
        # if either the prediction or the truth is no-answer then f1 = 1 if they agree, 0 otherwise
        if len(pred_tokens) == 0 or len(true_tokens) == 0:
            return int(pred_tokens == true_tokens)
  
        common_tokens = set(pred_tokens) & set(true_tokens)
  
        # if there are no common tokens then f1 = 0
        if len(common_tokens) == 0:
            return 0
  
        prec = len(common_tokens) / len(pred_tokens)
        rec = len(common_tokens) / len(true_tokens)

        f1 = round(2 * (prec * rec) / (prec + rec), 2)
  
        return f1
    
    
  






































#tokenizer2 = AutoTokenizer.from_pretrained("cl-tohoku/bert-base-japanese")

#context = ["サッカーのワールドカップ・カタール大会では、日本代表の活躍だけでなく、試合後にごみを拾い集める日本のサポーターの姿が話題になりました。", "結婚前提で付き合っている彼氏に、こう言われた。"]
#question = ["国内で最も小柄なチャンピオンが誕生した。山本菫（すみれ）、20歳。", "1年前に出会い、付き合いだした。"]
#train_encoding = tokenizer2(context, question, truncation="only_first", max_length = 30, return_overflowing_tokens=True)

#a = search(context)

#b = cq_pair_search(train_encoding)

#c, d = b.find_context_start_end_index()

#d = b.find_ans_start_end_token_index([3,5])
#b1, b2, b3 =a.char_to_token()


#ans_start = [5, 12]

#ans_end = [16, 13]

#collection1, collection2 = a.ans_start_end_token_index(ans_start, ans_end)

#cq_ans_start, cq_ans_end = b.find_ans_start_end_token_index(ans_start_token_index_collection=collection1, ans_end_token_index_collection=collection2)

#print(cq_ans_start)
#print(cq_ans_end)