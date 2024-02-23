import torch
import json
from itertools import takewhile
from bleu_eval import *

def eval_during_training(mod, test_dataset, device, caption_length, vocab_size, detokenize_dict):
    print("Testing Model:")
    mod.eval()
    criterion = torch.nn.NLLLoss()
    store_labels = []
    store_predicted_labels = []
    video_labels = []
    with torch.no_grad():
        for idx, data in enumerate(test_dataset):
            mod.zero_grad()
            feat = data[0].requires_grad_().to(device)
            labels = data[1].max(2)[1].to(device)
            predicted_labels = mod(feat.float(), labels)

            predicted_labels = predicted_labels.reshape(-1, caption_length-1, vocab_size)
            store_labels, store_predicted_labels = detoken(predicted_labels, data[2], store_labels, store_predicted_labels, detokenize_dict)
            
            loss = 0
            for b in range(data[0].shape[0]):          
                loss += criterion(predicted_labels[b,:], labels[b,1:])
                video_labels.append(data[3][b])
            
            print('Prediction:')
            for i in store_predicted_labels:
                print(i)
            print('Label:')
            for j in store_labels:
                print(j)
            
            if idx == 0:
                break
            
            
            
def detoken(predicted_labels, labels, store_labels, store_predicted_labels, detokenize_dict):
    endsyntax = ["<EOS>", "<PAD>"]
    predicted_labels_index = predicted_labels.max(2)[1]
    for i in range(predicted_labels.shape[0]):
        predicted_label = [detokenize_dict[int(w_idx.cpu().numpy())] for w_idx in predicted_labels_index[i,:]]
        predicted_label = list(takewhile(lambda x: x not in endsyntax, predicted_label))
        
        store_labels.append(str(labels[i]))
        store_predicted_labels.append(" ".join(predicted_label))
    return store_labels, store_predicted_labels


def evaluator(mod, test_dataset, device, caption_length, vocab_size, detokenize_dict, output_filename):
    print("Testing Model")
    mod.eval()
    criterion = torch.nn.NLLLoss()
    store_labels = []
    store_predicted_labels = []
    video_labels = []
    with torch.no_grad():
        for idx, data in enumerate(test_dataset):
            mod.zero_grad()
            feat = data[0].requires_grad_().to(device)
            labels = data[1].max(2)[1].to(device)
            # print(labels.device)
            # print(feat.device)
            predicted_labels = mod(feat.float(), labels)

            predicted_labels = predicted_labels.reshape(-1, caption_length-1, vocab_size)
            store_labels, store_predicted_labels = detoken(predicted_labels, data[2], store_labels, store_predicted_labels, detokenize_dict)
            
            loss = 0
            for b in range(data[0].shape[0]):          
                loss += criterion(predicted_labels[b,:], labels[b,1:])
                video_labels.append(data[3][b])
    w2f(video_labels, store_predicted_labels, store_labels, output_filename=output_filename)
    
    
    
def w2f(test_fname, predicted_labels, store_labels, output_filename="result.txt"):
    with open(output_filename, "w") as f:
        for i in range(len(store_labels)):
            f.write(f"{test_fname[i]}, {predicted_labels[i]}\n")

def bleu_score(output_filename="result.txt", correct_label_path="./MLDS_hw2_1_data/testing_label.json"):
    test = json.load(open(correct_label_path,'r'))
    result = {}
    with open("./"+output_filename,'r') as f:
        for line in f:
            line = line.rstrip()
            comma = line.index(',')
            test_id = line[:comma]
            caption = line[comma+1:]
            result[test_id] = caption
    bleu=[]
    for item in test:
        score_per_video = []
        captions = [x.rstrip('.') for x in item['caption']]
        score_per_video.append(BLEU(result[item['id']],captions,True))
        bleu.append(score_per_video[0])
    average = sum(bleu) / len(bleu)
    print("Average bleu score is " + str(average))