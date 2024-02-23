import torch

from argparse import ArgumentParser
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils import dataset
from utils import model
from utils import helper



parser = ArgumentParser(description="Training script parameters")

parser.add_argument('--train_label_loc', type=str,
                    default='/home/dehaoq/HW2/dataset2/MLDS_hw2_1_data/training_label.json', 
                    help='training label location')
parser.add_argument('--test_label_loc', type=str, default='/home/dehaoq/HW2/dataset2/MLDS_hw2_1_data/testing_label.json', 
                    help='testing label location')
parser.add_argument('--train_input_loc', type=str, default='/home/dehaoq/HW2/dataset2/MLDS_hw2_1_data/training_data', 
                    help='training label location')
parser.add_argument('--test_input_loc', type=str, default='/home/dehaoq/HW2/dataset2/MLDS_hw2_1_data/testing_data', 
                    help='testing label location')

parser.add_argument('--batch_size', type=int, default=64, help='batch size for training and testing')
parser.add_argument('--epoch', type=int, default=150, help='Number of training epoch')
parser.add_argument('--hidden_size', type=int, default=256, help='model hidden size')
parser.add_argument('--dropout_rate', type=float, default=0.3, help='drop rate for encoder')
parser.add_argument('--lr', type=float, default=0.0005, help='model learning rate')
parser.add_argument('--eta_min', type=float, default=0.0001, help='model learning rate')

args = parser.parse_args()



if __name__ == "__main__":
    
    print('Working on training dataset ...')
    train_dataset = dataset.train_VideoCaptionDataset(args.train_input_loc, args.train_label_loc)
   
    print('Working on tesitng dataset ...')
    test_dataset = dataset.test_VideoCaptionDataset(args.test_input_loc, 
                                                    args.test_label_loc, 
                                                    train_dataset.vocab, 
                                                    train_dataset.tokens_word, 
                                                    train_dataset.tokens_idx, 
                                                    train_dataset.clength)
   
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = args.batch_size, drop_last=True, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = 5, drop_last=True, shuffle=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = model.S2VT_model(vocab_size = len(train_dataset.tokens_idx), 
                             batch_size = args.batch_size, 
                             frame_dim = train_dataset[0][0].shape[1],
                             hidden_dim = args.hidden_size, 
                             dropout = args.dropout_rate, 
                             f_len = 80, 
                             device = device, 
                             caption_length = train_dataset.clength)
    
    
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    optimizer = torch.optim.Adam([{'params': model.parameters(), 'initial_lr': args.lr}], lr=args.lr)
    scheduler =  torch.optim.lr_scheduler.CosineAnnealingLR(optimizer = optimizer, T_max = args.epoch, eta_min = args.eta_min) 

    print('Model:')
    print(model)
    print('Training ...')
    start_all = torch.cuda.Event(enable_timing=True)
    end_all = torch.cuda.Event(enable_timing=True)
    start_all.record()
    
    model.to(device)
    criterion = torch.nn.NLLLoss()
    losses = []
    for i in range(args.epoch):
        
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        
        model.train()
        store_labels = []
        for idx, data in enumerate(train_loader):
            model.zero_grad()
            feat = data[0].requires_grad_().to(device)
            labels = data[1].max(2)[1].to(device)
            predicted_labels = model(feat.float(), labels)
            predicted_labels = predicted_labels.reshape(-1, train_dataset.clength-1, len(train_dataset.tokens_idx))
                                                        
            loss = 0
            for b in range(data[0].shape[0]):          
                loss += criterion(predicted_labels[b,:], labels[b,1:])
            loss.backward()
            optimizer.step()
            
        end.record()
        torch.cuda.synchronize()
        print(f'******* Epochs: {i}, Loss: {loss.item()/len(train_loader):.3f}, Time:{start.elapsed_time(end)/1000:.3f}s, LR:{scheduler.get_last_lr()[0]:.8f} *******')
        losses.append(loss.item()/len(train_loader))
        
        helper.eval_during_training(model, 
                                    test_loader, 
                                    device, 
                                    train_dataset.clength, 
                                    len(train_dataset.tokens_idx), 
                                    train_dataset.tokens_idx)
        scheduler.step()
    
    
    end_all.record()
    torch.cuda.synchronize()
        
    print(f'****** Done in {start_all.elapsed_time(end_all)/3600000:.3f}h! Saving model and plotting loss. ******')
    torch.save(model.state_dict(), f'model.pth')
    
    figure = plt.figure(figsize=(5, 2.5))
    plt.plot([i for i in range(len(losses))], losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show() 
    figure.savefig('epoch_vs_loss.png')
    
    
       

    

    

