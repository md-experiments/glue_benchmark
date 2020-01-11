import torch
import torchtext
import random
import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef

def get_device():
  # If there's a GPU available...
  if torch.cuda.is_available():    

      # Tell PyTorch to use the GPU.    
      device = torch.device("cuda")   
      print('There are %d GPU(s) available.' % torch.cuda.device_count())

      print('We will use the GPU:', torch.cuda.get_device_name(0))

  # If not...
  else:
      print('No GPU available, using the CPU instead.')
      device = torch.device("cpu")
  return device

def train(model, iterator, optimizer, criterion, metric, device):
    
    epoch_loss = 0
    epoch_acc = 0
    #epoch_label , epoch_preds = [], []
    model.train()
    
    for batch in iterator:
        
        optimizer.zero_grad()
        try:
          text, label = batch.text, batch.label
        #if type(batch)== torch.utils.data.dataloader.DataLoader:
        except:
          batch = tuple(t.to(device) for t in batch)
          #if task_type=='sentiment':
          text, label = batch
        #elif type(batch)== torchtext.data.iterator.Iterator:
    
        predictions = model(text).squeeze(1)
        #predictions = model(batch.text).squeeze(1)
        '''
        elif task_type=='similarity':
          text, textlabel = batch
          predictions = model(text, text1).squeeze(1)
          #predictions = model(batch.text,batch.text2).squeeze(1)
        '''
        loss = criterion(predictions, label)
        
        acc = metric(predictions, label)
        #epoch_label.append(label.to('cpu'))
        #epoch_preds.append(predictions.detach().cpu())

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    #epoch_label=torch.cat(epoch_label,0)
    #epoch_preds=torch.cat(epoch_preds,0)

    #epoch_loss = criterion(epoch_preds, epoch_label)
    #epoch_acc = metric(epoch_preds, epoch_label)
    return epoch_loss / len(iterator), epoch_acc / len(iterator)
    #return epoch_loss, epoch_acc

def evaluate(model, iterator, criterion, metric, device):
    
    #epoch_loss = 0
    #epoch_acc = 0
    
    model.eval()

    epoch_label , epoch_preds = [], []
    with torch.no_grad():

        for batch in iterator:
            try:
              text, label = batch.text, batch.label
            #if type(batch)== torch.utils.data.dataloader.DataLoader:
            except:
              batch = tuple(t.to(device) for t in batch)
              #if task_type=='sentiment':
              text, label = batch
            predictions = model(text).squeeze(1)
            
            #loss = criterion(predictions, label)
            
            #acc = metric(predictions, label)
            epoch_label.append(label.to('cpu'))
            epoch_preds.append(predictions.detach().cpu())

            #epoch_loss += loss.item()
            #epoch_acc += acc.item()

    epoch_label=torch.cat(epoch_label,0)
    epoch_preds=torch.cat(epoch_preds,0)

    epoch_loss = criterion(epoch_preds, epoch_label)
    epoch_acc = metric(epoch_preds, epoch_label)
    #return epoch_loss / len(iterator), epoch_acc / len(iterator)
    return epoch_loss, epoch_acc

import time

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """

    #round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float() #convert into float for division 
    acc = correct.sum() / len(correct)
    return acc


def matthews_corr(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """

    #round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    rounded_preds = rounded_preds.detach().cpu().numpy().flatten()
    y = y.to('cpu').numpy().flatten()
    #correct = (rounded_preds == y).float() #convert into float for division 
    #acc = correct.sum() / len(correct)
  
    return matthews_corrcoef(y,rounded_preds)

def train_model(trn_iter, val_iter, model, optimizer, criterion, metric, exp_name, task_type, device, N_EPOCHS = 12):
  

  print('=====================')
  print(f'Running experiment {exp_name}')
  best_valid_loss = float('inf')
  model_res=[]
  log_col=['Exp Name', 'Epoch', 'Epoch Mins', 'Epoch Secs', 'Train Loss', 'Train Accuracy', 'Valid Loss', 'Valid Accuracy']
 
  for epoch in range(N_EPOCHS):
      
      start_time = time.time()
      
      train_loss, train_acc = train(model, trn_iter, optimizer, criterion, metric, device)

      valid_loss, valid_acc = evaluate(model, val_iter, criterion, metric, device)
          
      end_time = time.time()
          
      epoch_mins, epoch_secs = epoch_time(start_time, end_time)
          
      if valid_loss < best_valid_loss:
          best_valid_loss = valid_loss
          torch.save(model.state_dict(), exp_name+'-model.pt')
      
      print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
      print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
      print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')
      new_val=[exp_name, epoch, epoch_mins, epoch_secs, train_loss, train_acc, valid_loss, valid_acc, ]
      model_res=model_res+[new_val]
    
  df=pd.DataFrame(model_res, columns=log_col)
  return df

