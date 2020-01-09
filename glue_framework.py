import torch
import torchtext

def train(model, iterator, optimizer, criterion, metric, device):
    
    epoch_loss = 0
    epoch_acc = 0
    
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
        
        loss.backward()
        
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, criterion, metric, device):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()
    
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
            
            loss = criterion(predictions, label)
            
            acc = metric(predictions, label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

import time

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs
