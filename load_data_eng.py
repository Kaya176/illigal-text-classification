#For preprocessing
import pandas as pd
import string
#For Embedding
import numpy as np
import re 
from tqdm import tqdm
#torch
from torch.utils.data import Dataset
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

class DataEng(Dataset):
    def __init__(self,file_name,tokenizer,max_len,padding_idx):
        self.file_name = file_name
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.punc = string.punctuation
        self.table = dict((ord(char), u' ') for char in self.punc)
        self.analysis_type = 'noun'
        self.padding_idx = padding_idx
        self.final = self.load_data()
        self.len = len(self.final)
        
    def load_data(self):
        '''
        보통 csv파일 아니면 엑셀파일이므로 이 두 #형식의 데이터만 처리함.
        '''
        if self.file_name.split(sep = '.')[-1] == 'csv':
            final = pd.read_csv(self.file_name)
        elif self.file_name.split(sep = '.')[-1] == 'txt':
            final = pd.read_table(self.file_name)
        else:
            final = pd.read_excel(self.file_name,engine = 'openpyxl')
        return final
    

    def _add_padding2data(self,sentence):
        if len(sentence) < self.max_len:
            pad_seq = np.array([self.padding_idx] * (self.max_len - len(sentence)))
            sentence = np.concatenate([sentence,pad_seq])
        else:
            sentence = sentence[:self.max_len]
        return sentence

    def __getitem__(self,idx):
        result_sent = []
        data= self.final
        line = data.iloc[idx]

        text = line['text'] #'text' -> 'document' (text attr name)
        #for label to onehot
        label = line['label'] #'lable' -> 'label' (label attr name)
        onehot = np.zeros(2) #number of labels
        onehot[label] = 1

        rnn_input = self.tokenizer.encode_plus(str(text),add_special_tokens = True,max_length = 100,truncation = True,
                                            pad_to_max_length = True)
        input_ids = rnn_input.input_ids
        input_attention_mask = rnn_input.attention_mask
        input_type_ids = rnn_input.token_type_ids
        
        return {"input_ids" : np.array(input_ids,dtype = np.int_),
                'attention_mask' : np.array(input_attention_mask,dtype = np.int_),
                "type_ids" : np.array(input_type_ids,dtype = np.int_),
                'label' : np.array(onehot),
                'not_onehot' : np.array(label)}

    def __len__(self):
        return self.len
        
if __name__ == '__main__':
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
    rnninp = DataEng(file_name= './ECAI/data/toxic/11_train.csv',tokenizer=tokenizer,max_len = 100,padding_idx= 0)
    print(rnninp[-2])