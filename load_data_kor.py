#For preprocessing
from konlpy.tag import Mecab
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

class Data(Dataset):
    def __init__(self,file_name,tokenizer,max_len,padding_idx):
        self.file_name = file_name
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.mecab = Mecab(dicpath="C:\mecab\mecab-ko-dic")
        self.punc = string.punctuation
        self.table = dict((ord(char), u' ') for char in self.punc)
        self.analysis_type = 'noun'
        self.padding_idx = padding_idx
        self.final = self.load_data()
        self.len = len(self.final)
        
    def load_data(self):

        if self.file_name.split(sep = '.')[-1] == 'csv':
            final = pd.read_csv(self.file_name)
        elif self.file_name.split(sep = '.')[-1] == 'txt':
            final = pd.read_table(self.file_name)
        else:
            final = pd.read_excel(self.file_name,engine = 'openpyxl')

        return final
    
    
    def _remove_link_img(self,string):
        string = str(string)
        string = string.split()
        string = " ".join([s for s in string if "http" not in s and 'pic' not in s])
        return string
    
    def _cleaning(self,string):
        string = self._remove_link_img(string)
        string = re.sub(r"\'s ", " ", string)
        string = re.sub(r"\'m ", " ", string)
        string = re.sub(r"\'ve ", " ", string)
        string = re.sub(r"n\'t ", " not ", string)
        string = re.sub(r"\'re ", " ", string)
        string = re.sub(r"\'d ", " ", string)
        string = re.sub(r"\'ll ", " ", string)
        string = re.sub("-", " ", string)
        string = re.sub(r"@", " ", string)
        string = re.sub('\'', '', string)
        string = string.translate(self.table)
        string = string.replace("..", "").strip()
        string = string.lower()
        return ' '.join([t for t in string.split() if t != "" and t.find(" ") == -1])

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

        text = self._cleaning(line['text']) 
        #for label to onehot
        label = line['label']
        onehot = np.zeros(2) #number of labels
        onehot[label] = 1
        if self.analysis_type == 'noun':
            text = self.mecab.nouns(text)
        else:
            text = self.mecab.morphs(text)
        text = " ".join(text)

        rnn_input = self.tokenizer.encode_plus(text,add_special_tokens = True,max_length = 100,truncation = True,
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
    rnninp = Data(file_name= './data/toxic/11_valid.csv',tokenizer=tokenizer,max_len = 100,padding_idx= 0)
    print(rnninp[3])