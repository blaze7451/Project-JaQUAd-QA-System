import torch
from transformers import AutoTokenizer
from tuning_util import data_processing, search, cq_pair_search, eval_util
from torch.utils.data import DataLoader, Dataset
from transformers import BertForQuestionAnswering
import argparse
from tqdm import tqdm
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def parse_args():
    parser = argparse.ArgumentParser(description="Extractive QA system based on JaQUAd dataset")
    parser.add_argument("--dataset-path", type=str, default="C:\\Python\\Pytorch\\Transformer related\\Project JaQUAd QA System\\data\\JaQuAD_test.csv", help="JaQUAd Test Dataset")
    parser.add_argument("--model-path", type=str, default="C:\\Python\\Pytorch\\Transformer related\\Project JaQUAd QA System\\weights\\weight13", help="input model path")
    parser.add_argument("--seq-length", type=int, default=512, help="input tokes length (default: 512)")
    parser.add_argument('--batch-size', type=int, default=8, help='training batch size (default: 16)')
    parser.add_argument('--learning-rate', type=float, default=5e-5, help='learning rate (default: 5e-5)')
    return parser.parse_args()

args = parse_args()

test_data = data_processing(args.dataset_path)

test_contexts, test_questions, test_answers, test_ans_char_starts = test_data.read_data()

test_ans_char_ends = test_data.add_ans_end(answers=test_answers, contexts=test_contexts, ans_starts=test_ans_char_starts)

tokenizer = AutoTokenizer.from_pretrained(args.model_path)
char_tokenizer = AutoTokenizer.from_pretrained("cl-tohoku/bert-base-japanese-char")

test_encoding = tokenizer(test_contexts, test_questions, truncation="only_first", max_length=args.seq_length, stride=50, padding = True)

test_ans_search =search(test_contexts)

test_char2token_lists, test_w_lists, test_c_lists = test_ans_search.char_to_token()

test_ans_token_start, test_ans_token_end = test_ans_search.ans_start_end_token_index(ans_start_char_index=test_ans_char_starts, ans_end_char_index=test_ans_char_ends)

test_cq = cq_pair_search(test_encoding)

test_context_start, test_context_end = test_cq.find_context_start_end_index()

real_ans_start, real_ans_end = test_cq.find_ans_start_end_token_index(ans_start_token_index_collection=test_ans_token_start, ans_end_token_index_collection=test_ans_token_end)

test_encoding.update({'start_positions': real_ans_start, 'end_positions': real_ans_end})

class JaQuAD_Dataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
    def __len__(self):
        return len(self.encodings.input_ids)

test_dataset = JaQuAD_Dataset(test_encoding)

test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

model = BertForQuestionAnswering.from_pretrained(args.model_path)

model = model.to(device)

model.eval()

acc=[]

start_pred_list = []
end_pred_list = []
start_true_list = []
end_true_list = []
for batch in tqdm(test_loader):
    with torch.no_grad():
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        start_true = batch['start_positions'].to(device)
        end_true = batch['end_positions'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask)

        start_pred = torch.argmax(outputs['start_logits'], dim=1)
        end_pred = torch.argmax(outputs['end_logits'], dim=1)

        acc.append(((start_pred == start_true).sum()/len(start_pred)).item())
        acc.append(((end_pred == end_true).sum()/len(end_pred)).item())
        
        start_pred = start_pred.cpu().tolist()
        start_pred_list += start_pred

        end_pred = end_pred.cpu().tolist()
        end_pred_list += end_pred

        start_true = start_true.cpu().tolist()
        start_true_list += start_true

        end_true = end_true.cpu().tolist()
        end_true_list += end_true

acc = sum(acc)/len(acc)
print(acc)  #Best result is weight13 with acc 0.41227180530405916. #0.4064401623020307 for weight11

#Compute EM score  #0.3500888550393501  #0.3351104341203351 for weight 11
ans_pred_list = []
ans_true_list = []
ans_token_list = []
true_ans_token_list = []
for i, start_pred in enumerate(start_pred_list):
    end_pred = end_pred_list[i] +1
    ans_token = tokenizer.decode(test_encoding["input_ids"][i]).split( )[start_pred:end_pred]
    ans_token_list.append(ans_token)
    ans_pred = "".join(ans_token)
    ans_pred_list.append(ans_pred)
    
    start_true = start_true_list[i]
    end_true = end_true_list[i]+1
    true_ans_token = tokenizer.decode(test_encoding["input_ids"][i]).split( )[start_true:end_true]
    true_ans_token_list.append(true_ans_token)
    ans_true = "".join(true_ans_token)
    ans_true_list.append(ans_true)

eval_tool = eval_util()

em_score = eval_tool.EM(ans_pred_list, ans_true_list)

print("The EM score is {}.".format(em_score['exact_match']))

#Compute F1 score  #0.5441914191419139 #0.5450418888042655 for weight11
total_f1_score = 0

for i in range(len(ans_token_list)):
    pred_tokens = ans_token_list[i]
    true_tokens = true_ans_token_list[i]
    
    f1_score = eval_tool.compute_f1(pred_tokens, true_tokens)
    total_f1_score += f1_score

avg_f1_score = total_f1_score/len(ans_token_list)

print("The average f1 score is {}.".format(avg_f1_score))