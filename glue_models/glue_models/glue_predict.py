from transformers import BertTokenizer, BertModel
from glue_models import models as gl

import torch



def glue_init_model(tokenizer, bert, model_placeholder, model_chkp):
    HIDDEN_DIM = 256
    OUTPUT_DIM = 1
    N_LAYERS = 2
    BIDIRECTIONAL = True
    DROPOUT = 0.25

    model = model_placeholder(bert,
                          HIDDEN_DIM,
                          OUTPUT_DIM,
                          N_LAYERS,
                          BIDIRECTIONAL,
                          DROPOUT)

    model.load_state_dict(torch.load(model_chkp, map_location=torch.device('cpu')))
   
    return model

class FrameworkPredictor():
    # NB: all predicts use QNLI for now
    def __init__(self, path):

        self.mdl_blank=gl.BERTGRUSentiment
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.max_input_length = self.tokenizer.max_model_input_sizes['bert-base-uncased']
        self.model_QNLI = glue_init_model(self.tokenizer, self.bert, self.mdl_blank, path+'QNLI-model.pt')
        self.model_QQP = glue_init_model(self.tokenizer, self.bert, self.mdl_blank, path+'QQP-model.pt')
    
    def question_similarity(self, sentence, sentence2):
        self.model_QQP.eval()
        tokens = self.tokenizer.tokenize('[CSL] '+sentence+' [SEP] '+sentence2+' [SEP]')
        indexed = self.tokenizer.convert_tokens_to_ids(tokens)
        tensor = torch.LongTensor(indexed)#.to(device)
        tensor = tensor.unsqueeze(0)
        prediction = torch.sigmoid(self.model_QQP(tensor))
        return prediction.item()
    
    def question_answering(self, sentence, sentence2):
        self.model_QNLI.eval()
        tokens = self.tokenizer.tokenize('[CSL] '+sentence+' [SEP] '+sentence2+' [SEP]')
        indexed = self.tokenizer.convert_tokens_to_ids(tokens)
        tensor = torch.LongTensor(indexed)#.to(device)
        tensor = tensor.unsqueeze(0)
        prediction = torch.sigmoid(self.model_QNLI(tensor))
        return prediction.item()