import argparse
import torch
from torch.utils.data import Dataset, DataLoader
from tuning_util import data_processing, search, cq_pair_search
from transformers import AutoTokenizer
from transformers import BertForQuestionAnswering
from transformers import AdamW
from tqdm import tqdm
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def parse_args():
    parser = argparse.ArgumentParser(description="Extractive QA system based on JaQUAd dataset")
    parser.add_argument("--dataset-path", type=str, default="C:\\Python\\Pytorch\\Transformer related\\Project JaQUAd QA System\\data\\JaQuAD_train.csv", help="JaQUAd Dataset")
    parser.add_argument("--model-path", type=str, default="C:\\Python\\Pytorch\\Transformer related\\Project JaQUAd QA System\\weights\\weight2", help="output model path")
    parser.add_argument("--seq-length", type=int, default=512, help="input tokes length (default: 512)")
    parser.add_argument('--batch-size', type=int, default=8, help='training batch size (default: 16)')
    parser.add_argument('--learning-rate', type=float, default=5e-5, help='learning rate (default: 0.0001)')
    parser.add_argument('--epochs', type=int, default=15, help='number of epochs to train (default: 5)')
    return parser.parse_args()

args = parse_args()
train_data = data_processing(args.dataset_path)

train_contexts, train_questions, train_answers, train_ans_char_starts = train_data.read_data()

train_ans_char_ends = train_data.add_ans_end(answers=train_answers, contexts=train_contexts, ans_starts=train_ans_char_starts)

tokenizer = AutoTokenizer.from_pretrained("cl-tohoku/bert-base-japanese")
char_tokenizer = AutoTokenizer.from_pretrained("cl-tohoku/bert-base-japanese-char")

train_encoding = tokenizer(train_contexts, train_questions, truncation="only_first", max_length=args.seq_length, stride=50, padding = True)

train_ans_search = search(train_contexts)

train_char2token_lists, train_w_lists, train_c_lists = train_ans_search.char_to_token()

train_ans_token_start, train_ans_token_end = train_ans_search.ans_start_end_token_index(ans_start_char_index=train_ans_char_starts, ans_end_char_index=train_ans_char_ends)

train_cq = cq_pair_search(train_encoding)

train_context_start, train_context_end = train_cq.find_context_start_end_index()

real_ans_start, real_ans_end = train_cq.find_ans_start_end_token_index(ans_start_token_index_collection=train_ans_token_start, ans_end_token_index_collection=train_ans_token_end)

train_encoding.update({'start_positions': real_ans_start, 'end_positions': real_ans_end})

class JaQuAD_Dataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
    def __len__(self):
        return len(self.encodings.input_ids)

train_dataset = JaQuAD_Dataset(train_encoding)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

model = BertForQuestionAnswering.from_pretrained("cl-tohoku/bert-base-japanese")

optim = AdamW(model.parameters(), lr=args.learning_rate)

model.to(device)
model.train()

for epoch in range(args.epochs):
    loop = tqdm(train_loader, leave=True)
    for batch in loop:
        optim.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch["attention_mask"].to(device)
        start_positions = batch['start_positions'].to(device)
        end_positions = batch["end_positions"].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, start_positions=start_positions, end_positions=end_positions)
        loss = outputs[0]
        loss.backward()
        optim.step()

        loop.set_description(f'Epoch {epoch+1}')
        loop.set_postfix(loss=loss.item())
    
    if epoch+1 == 10:
        model.save_pretrained("C:\\Python\\Pytorch\\Transformer related\\Project JaQUAd QA System\\weights\\weight10")
        tokenizer.save_pretrained("C:\\Python\\Pytorch\\Transformer related\\Project JaQUAd QA System\\weights\\weight10")
    elif epoch+1 == 11:
        model.save_pretrained("C:\\Python\\Pytorch\\Transformer related\\Project JaQUAd QA System\\weights\\weight11")
        tokenizer.save_pretrained("C:\\Python\\Pytorch\\Transformer related\\Project JaQUAd QA System\\weights\\weight11")
    elif epoch+1 == 12:
        model.save_pretrained("C:\\Python\\Pytorch\\Transformer related\\Project JaQUAd QA System\\weights\\weight12")
        tokenizer.save_pretrained("C:\\Python\\Pytorch\\Transformer related\\Project JaQUAd QA System\\weights\\weight12")
    elif epoch+1 == 13:
        model.save_pretrained("C:\\Python\\Pytorch\\Transformer related\\Project JaQUAd QA System\\weights\\weight13")
        tokenizer.save_pretrained("C:\\Python\\Pytorch\\Transformer related\\Project JaQUAd QA System\\weights\\weight13")
    elif epoch+1 == 14:
        model.save_pretrained("C:\\Python\\Pytorch\\Transformer related\\Project JaQUAd QA System\\weights\\weight14")
        tokenizer.save_pretrained("C:\\Python\\Pytorch\\Transformer related\\Project JaQUAd QA System\\weights\\weight14")
    elif epoch+1 == 15:
        model.save_pretrained("C:\\Python\\Pytorch\\Transformer related\\Project JaQUAd QA System\\weights\\weight15")
        tokenizer.save_pretrained("C:\\Python\\Pytorch\\Transformer related\\Project JaQUAd QA System\\weights\\weight15")


