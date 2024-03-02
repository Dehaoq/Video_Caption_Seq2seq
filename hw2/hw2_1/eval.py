import torch
import sys

from utils import dataset
from utils import model
from utils import helper
from bleu_eval import *




data_loc = sys.argv[1]
output_filename = sys.argv[2]

# /home/dehaoq/HW2/dataset2/MLDS_hw2_1_data
test_data_path = f"{data_loc}/MLDS_hw2_1_data/testing_data/"
test_labels = str(f"{data_loc}/MLDS_hw2_1_data/testing_label.json")

train_dataset = dataset.train_VideoCaptionDataset(f'{data_loc}/MLDS_hw2_1_data/training_data',
                                            f'{data_loc}/MLDS_hw2_1_data/training_label.json')



batch_size = 10
device = 'cpu'

# Need to follow the setting in train.py
hidden_dim = 256
vocab_size = len(train_dataset.tokens_idx)
detokenize_dict = train_dataset.tokens_idx
feat_size = 4096
seq_length = 80
caption_length = train_dataset.clength
drop = 0.3




model = model.S2VT_model(vocab_size, 
                         batch_size, 
                         feat_size, 
                         hidden_dim, 
                         drop, 
                         80, 
                         device, 
                         caption_length)

test_ds = dataset.test_VideoCaptionDataset(test_data_path, 
                                           test_labels, 
                                           train_dataset.vocab, 
                                           train_dataset.tokens_word, 
                                           train_dataset.tokens_idx, 
                                           train_dataset.clength)

test_dataset = torch.utils.data.DataLoader(test_ds, 
                                           batch_size=batch_size, 
                                           shuffle=True)


model.load_state_dict(torch.load("/home/dehaoq/hw2/model.pth"))
helper.evaluator(model, test_dataset, device, caption_length, vocab_size, detokenize_dict, output_filename)
helper.bleu_score(output_filename=output_filename, correct_label_path=test_labels)