import torch
from load_data_kor import Data
from load_data_eng import DataEng
from torch.utils.data import DataLoader
import torch.nn as nn
from torch import optim
from torch.nn import functional as F
from transformers import BertModel,BertTokenizer
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score,precision_score,recall_score

import warnings
warnings.filterwarnings('ignore')

BATCH_SIZE = 16
tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")

#train_data = Data(file_name= './ECAI/data/ratings_train.txt',tokenizer=tokenizer,max_len = 100,padding_idx= 0)
#valid_data = Data(file_name= './ECAI/data/ratings_test.txt',tokenizer=tokenizer,max_len = 100,padding_idx= 0)
#test_data = Data(file_name= './ECAI/data/ratings_test.txt',tokenizer=tokenizer,max_len = 100,padding_idx= 0)

#for toxic data
train_data = DataEng(file_name= './ECAI/data/toxic/11_train.csv',tokenizer=tokenizer,max_len = 100,padding_idx= 0)
valid_data = DataEng(file_name= './ECAI/data/toxic/11_valid.csv',tokenizer=tokenizer,max_len = 100,padding_idx= 0)
test_data = DataEng(file_name= './ECAI/data/toxic/11_test.csv',tokenizer=tokenizer,max_len = 100,padding_idx= 0)

train_total = DataLoader(train_data,
                    batch_size = BATCH_SIZE,
                    num_workers=5)

valid_total = DataLoader(valid_data,
                    batch_size = BATCH_SIZE,
                    num_workers=5)

test_total = DataLoader(test_data,
                    batch_size = BATCH_SIZE,
                    num_workers=5)

################################################

class LSTM_CLS(nn.Module):
    def __init__(self,batch_size,num_cls):
        super(LSTM_CLS,self).__init__()
        self.batch_size = batch_size
        self.bert = BertModel.from_pretrained("bert-base-multilingual-cased")#skt/kobert-base-v1
        self.lstm = nn.LSTM(self.bert.config.hidden_size,256,num_layers = 3,dropout = 0.2,batch_first = True)
        self.drop1 = nn.Dropout(0.1)
        self.fn1 = nn.Linear(256,64)
        self.fn2 = nn.Linear(64,num_cls)
    
    def forward(self,input_ids,mask):
        
        x = self.bert(input_ids,attention_mask = mask)
        out,_ = self.lstm(x.last_hidden_state)
        out = out[:,-1,:]
        out = self.drop1(out) #CLS token
        out = self.fn1(out)
        out = F.relu(out)
        out = self.fn2(out)
        return out

def predict(model,data):
    model.eval()
    predictions = []
    labels = []

    with torch.no_grad():
        for batch in data:
            input_ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            
            label = batch['label']
            y_pred = model(input_ids = input_ids,mask = mask)
            y_pred = y_pred.cpu()
            predictions += torch.argmax(y_pred,axis=1).numpy().tolist()
            labels += np.argmax(label,axis=1).tolist()

    print("Prediction samples : ",predictions[:10])
    print("Label samples : ",labels[:10])
    print(f"Accuracy : {accuracy_score(labels,predictions):.4f} | Precision : {precision_score(labels,predictions):.4f}| Recall : {recall_score(labels,predictions):.4f}")

if __name__ == "__main__":

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = LSTM_CLS(batch_size = BATCH_SIZE,num_cls= 2)
    model.to(device)
    optimizer = optim.AdamW(model.parameters(),lr = 1e-6)
    loss_fn = nn.BCEWithLogitsLoss()
    epochs = 4

    for epoch in range(epochs):
        total_loss = 0

        model.train()
        for batch_idx,data in tqdm(enumerate(train_total)):

            input_ids = data['input_ids'].to(device)
            mask = data['attention_mask'].to(device)
            
            label = torch.tensor(data['label'],dtype = torch.float,device = device)

            out = model(input_ids,mask = mask)
            optimizer.zero_grad()

            loss = loss_fn(out,label)

            loss.backward()
            optimizer.step()

            total_loss += loss.detach().item()

            if batch_idx % 100 == 0:
                print(f"Epoch : {epoch+1} | train loss : {loss:.4f}")

        with torch.no_grad():
            predictions = []
            labels = []
            for batch in valid_total:
                input_ids = batch['input_ids'].to(device)
                mask = batch['attention_mask'].to(device)
                
                labels += batch['not_onehot'].cpu()
                y_pred = model(input_ids = input_ids,mask = mask)
                y_pred = y_pred.cpu().numpy()
                predictions += list(np.argmax(y_pred,axis=1))
                
            print("Prediction samples : ",predictions[:10])
            print("Label samples : ",labels[:10])
            print(f"Accuracy : {accuracy_score(labels,predictions):.4f} | Precision : {precision_score(labels,predictions):.4f}| Recall : {recall_score(labels,predictions):.4f}")
        
    predict(model,test_total)