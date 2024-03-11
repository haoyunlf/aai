
import torch
import time
import random
from models import Modern_DBLSTM_1, DBLSTM_LayerNorm, Traditional_BLSTM_2, Modern_BLSTM_2
import numpy as np
from data_utils import pad_collate
import configargparse
from configs import configs
from utils import EarlyStopping
import os
from torch.utils.data import Dataset, DataLoader, random_split
import wandb
from sklearn.model_selection import KFold
import scipy.signal
import pickle
import librosa


class MyDataset(Dataset):
    def __init__(self, processed_root, spks, feature, exclude_tr=False):
        self.root_path = processed_root
        self.x = []
        self.y = []
        self.n_mfcc = 40
        self.hop_length = int(10 / 1000 * 16000)
        self.frame_length = int(25 / 1000 * 16000)
        self.haskins = ["F01", "F02", "F03", "F04", "M01", "M02", "M03", "M04"]
        for spk in spks:
            if feature == 'mfcc':
                with open(os.path.join(processed_root, spk, 'ema.pkl'), 'rb') as f2:
                    target = pickle.load(f2)
                id_list = []
                mfcc_list = []
                input = []
                for wav_file in os.listdir(os.path.join(processed_root, spk, 'wav')):
                    wav_name = os.path.splitext(wav_file)[0]
                    wav, sr = librosa.load(os.path.join(processed_root, spk, 'wav', wav_file), sr=16000)
                    mfcc = librosa.feature.mfcc(y=wav, sr=sr, n_mfcc=self.n_mfcc,
                                       n_fft=self.frame_length, hop_length=self.hop_length, n_mels=self.n_mfcc).T
                    id_list.append(wav_name)
                    mfcc_list.append(mfcc)

                mfccs = np.concatenate(mfcc_list, axis=0)
                std_mfcc = np.std(mfccs, axis=0, keepdims=True)
                mean_mfcc = np.mean(mfccs, axis=0, keepdims=True)

                for i in range(len(id_list)):
                    input.append({'id': id_list[i], 'data': (mfcc_list[i]-mean_mfcc)/std_mfcc})
                    
            else:
                with open(os.path.join(processed_root, spk, 'feature', feature+'.pkl'), 'rb') as f1:
                    input = pickle.load(f1)
                with open(os.path.join(processed_root, spk, 'ema.pkl'), 'rb') as f2:
                    target = pickle.load(f2)

            input = sorted(input, key=lambda x: x['id'])
            target = sorted(target, key=lambda x: x['id'])
            for i in range(len(input)):
                assert input[i]['id'] == target[i]['id']
                
                x = input[i]['data']
                y = target[i]['data']
                
                if spk in self.haskins and exclude_tr == True:
                    y = y[:, 0,1,4,5,6,7,8,9,10,11]
                    
                y = scipy.signal.resample(y, num=len(x))
                if np.any(np.isnan(x)):
                    print('Found nan in {} feature'.format(spk))
                if np.any(np.isnan(y)):
                    print('Found nan in {} ema'.format(spk))
                
                if np.any(np.isnan(x)) or np.any(np.isnan(y)):
                    pass
                else:
                    self.x.append(x)
                    self.y.append(y)


    def __getitem__(self, index):
        x = torch.Tensor(self.x[index])
        y = torch.Tensor(self.y[index])

        return x, y

    def __len__(self):
        return len(self.x)
        

def train(args):

    print(args.__dict__)
    configs = args.__dict__

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    output_path = '/data/data2/yun/aai_expr/exp'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    input_path = '/data/data2/yun/aai_expr/Preprocessed_data'

    if args.feature in ["facebook/wav2vec2-base", "facebook/wav2vec2-base-960h"]:
        layer_number = 13
    elif args.feature in ["facebook/wav2vec2-large", "facebook/wav2vec2-large-960h", "facebook/mms-300m"]:
        layer_number = 25
    elif args.feature in ["facebook/mms-1b", "facebook/mms-1b-all"]:
        layer_number = 49
    elif args.feature in ["mfcc"]:
        layer_number = 1

    if args.layer_from is None and args.layer_to is None:
        layer_range = range(layer_number)
    elif args.layer_from is None:
        layer_range = range(args.layer_to+1)
    elif args.layer_to is None:
        layer_range = range(args.layer_from, layer_number)
    else:
        layer_range = range(args.layer_from, args.layer_to+1)

    fold_number = args.fold_number
    spks = ['F01', 'F02', 'F03', 'M01', 'M02', 'M03']
    test_spks = ['F04', 'M04']

    kf = KFold(n_splits=fold_number, shuffle=False)

    for i in layer_range:
        if args.feature in ["mfcc", "mms-1b-all_eng_layer_28"]:
            feature = args.feature
        else:
            feature = args.feature.split('/')[-1] + '_layer_' + str(i)
        print('feature:', feature)      
        fold_log = np.zeros((fold_number, 3, 2))
        
        for fold, (train_index, test_index) in enumerate(kf.split(spks)):
    
            print('FOLD {}'.format(fold))
            print('-----------------------------')
            start = time.time()
            train_spks = [spks[i] for i in train_index]
            val_spks = [spks[i] for i in test_index]
            print('train spks', train_spks)
            print('val spks', val_spks)
        
            train_dataset = MyDataset(input_path, train_spks, feature, exclude_tr=args.exclude_tr)
            print('train data:', len(train_dataset))
            val_dataset = MyDataset(input_path, val_spks, feature, exclude_tr=args.exclude_tr)
            test_dataset = MyDataset(input_path, test_spks, feature, exclude_tr=args.exclude_tr)
    
            configs['train_spks'] = str(train_spks)
            configs['val_spks'] = str(val_spks)
            configs['test_spks'] = str(test_spks)
    
    
            train_loader = torch.utils.data.DataLoader(train_dataset,
                                                         batch_size=args.batch_size, shuffle=True,
                                                         num_workers=4, collate_fn=pad_collate)
        
            val_loader = torch.utils.data.DataLoader(val_dataset,
                                                      batch_size=args.batch_size, shuffle=True,
                                                      num_workers=4, collate_fn=pad_collate)
        
            test_loader = torch.utils.data.DataLoader(test_dataset,
                                                         batch_size=args.batch_size, shuffle=False,
                                                         num_workers=4,collate_fn=pad_collate)
    
            if args.Traditional_BLSTM_2:
                model = Traditional_BLSTM_2(args).to(device)
            if args.Modern_BLSTM_2:
                model = Modern_BLSTM_2(args).to(device)
            if args.Modern_BLSTM_1:
                model = Modern_DBLSTM_1(args).to(device)
        
            if args.SGD:
                optimizer = torch.optim.SGD(model.parameters(), lr=1e-6, momentum=0.9)
            if args.Adam:
                optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
            print(model, optimizer)
            es = EarlyStopping()
    
        # Train the model
    
            epoch_log = np.zeros((args.num_epochs, 3, 2))
        
            for epoch in range(args.num_epochs):
                total_loss = 0
                print("Epoch ", epoch + 1)
                all_pearson = []
                metrics = {}
                
                for i, sample in enumerate(train_loader):
                    xx_pad,  yy_pad, _, _, mask = sample
    
                    inputs = xx_pad.to(device)
                    targets = yy_pad.to(device)
    
                    if args.art_norm:
                        targets = (targets + ema_mean)*(4 * ema_std)
    
                    mask = mask.to(device)
    
                    outputs = model(inputs)
                    if args.art_norm:
                        outputs = (outputs + ema_mean)*(4 * ema_std)
    
                    loss = torch.sum(((outputs - targets) * mask) ** 2.0) / torch.sum(mask).item()
    
                    if targets.shape[0] == args.batch_size:
                        total_loss += loss.item()
                        #print(loss.item())
                    else:
                        total_loss += loss.item() * (targets.shape[0] / args.batch_size)
    
                    y = (outputs * mask).detach().cpu().numpy()
                    y_pred = (targets * mask).detach().cpu().numpy()
    
                    for b in range(y.shape[0]):
                        vy = y[b].T
                        vy_pred = y_pred[b].T
                        pearson = np.diagonal(np.corrcoef(vy, vy_pred), offset=vy.shape[0])
                        all_pearson.append(pearson)
    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
    
                total_loss = np.sqrt(total_loss / len(train_loader))
                epoch_log[epoch,0,0] = total_loss
                print('all_pearson', len(all_pearson), all_pearson[0].shape)
                epoch_log[epoch,0,1] = np.mean(np.concatenate(all_pearson))
    
                #metrics['train_rmse_fold'+str(fold)], metrics['train_pcc_fold'+str(fold)] = epoch_log[epoch,0,0], epoch_log[epoch,0,1]
                print('Epoch [{}/{}], Train RMSE: {:.4f}, PCC: {:.4f}'.format(epoch + 1, args.num_epochs, total_loss, epoch_log[epoch,0,1]))
    
                torch.cuda.empty_cache()
    
                with torch.no_grad():
                    total_loss = 0
                    all_pearson = []
                    for i, sample in enumerate(val_loader):
                        xx_pad, yy_pad, _, _, mask = sample
                        inputs = xx_pad.to(device)
                        targets = yy_pad.to(device)
                        if args.art_norm:
                            targets = (targets + ema_mean) * (4 * ema_std)
                        mask = mask.to(device)
    
                        outputs = model(inputs)
    
                        if args.art_norm:
                            outputs = (outputs + ema_mean) * (4 * ema_std)
    
                        loss = torch.sum(((outputs-targets)*mask)**2.0) / torch.sum(mask).item()
    
                        # Weigh differently the last smaller batch
                        if targets.shape[0] == args.batch_size:
                            total_loss += loss.item()
                        else:
                            total_loss += loss.item() * (targets.shape[0]/args.batch_size)
    
                        y = (outputs * mask).detach().cpu().numpy()
                        y_pred = (targets * mask).detach().cpu().numpy()
                        for b in range(y.shape[0]):
                            vy = y[b].T
                            vy_pred = y_pred[b].T
                            pearson = np.diagonal(np.corrcoef(vy, vy_pred), offset=vy.shape[0])                    
                            all_pearson.append(pearson)
    
                    total_loss = np.sqrt(total_loss / len(val_loader))
                    epoch_log[epoch, 1, 0] = total_loss
                    epoch_log[epoch, 1, 1] = np.mean(np.concatenate(all_pearson))
    
                    # metrics['validation_rmse_fold'+str(fold)], metrics['validation_pcc_fold'+str(fold)] = epoch_log[epoch, 1, 0], epoch_log[epoch, 1, 1]
                    print('Epoch [{}/{}], Validation RMSE: {:.4f}, PCC: {:.4f}'.format(epoch + 1, args.num_epochs, total_loss,
                                                                                                   epoch_log[epoch, 1, 1]))
    
                torch.cuda.empty_cache()
    
                with torch.no_grad():
                    total_loss = 0
                    all_pearson = []
                    for i, sample in enumerate(test_loader):
                        xx_pad, yy_pad, _, _, mask = sample
                        inputs = xx_pad.to(device)
                        targets = yy_pad.to(device)
                        if args.art_norm:
                            targets = (targets + ema_mean) * (4 * ema_std)
                        mask = mask.to(device)
    
                        outputs = model(inputs)
                        if args.art_norm:
                            outputs = (outputs + ema_mean) * (4 * ema_std)
                        loss = torch.sum(((outputs-targets)*mask)**2.0) / torch.sum(mask).item()
                        if targets.shape[0] == args.batch_size:
                            total_loss += loss.item()
                        else:
                            total_loss += loss.item() * (targets.shape[0] / args.batch_size)
                        
                        y = (outputs * mask).detach().cpu().numpy()
                        y_pred = (targets * mask).detach().cpu().numpy()
                        for b in range(y.shape[0]):
                            vy = y[b].T
                            vy_pred = y_pred[b].T
                            pearson = np.diagonal(np.corrcoef(vy, vy_pred), offset=vy.shape[0])
                            all_pearson.append(pearson)
                    
                    total_loss = np.sqrt(total_loss / len(test_loader))
                    epoch_log[epoch, 2, 0] = total_loss
                    epoch_log[epoch, 2, 1] = np.mean(np.concatenate(all_pearson))
    
                    #metrics['test_rmse_fold'+str(fold)], metrics['test_pcc_fold'+str(fold)] = epoch_log[epoch, 2, 0], epoch_log[epoch, 2, 1]
                    print('Epoch [{}/{}], Test RMSE: {:.4f}, PCC: {:.4f}'.format(epoch + 1, args.num_epochs, total_loss, epoch_log[epoch, 2, 1]))
    
                torch.cuda.empty_cache()
                
                # Early stopping:

                if es.call(epoch_log[epoch, 1, 0]):
                    best_id = np.argmin(epoch_log[:i, 1])
                    best_id = epoch
                    print("Early stopping! Best test loss: ", epoch_log[epoch,2])
                    break

            print("Runtime:", time.time()-start)
            fold_log[fold] = epoch_log[epoch, :].copy()
            print(fold_log)
            torch.save(model.state_dict(), os.path.join(output_path, feature + '_' + str(fold) + '.pt'))
            

        with open(os.path.join(output_path, feature + '_fold' + str(args.fold_number) + '.txt'), 'w') as f:
          f.write('fold\ttrain_rmse\ttrain_pcc\tval_rmse\tval_pcc\ttest_rmse\ttest_pcc\n')
          for i, fold_i in enumerate(fold_log):
            f.write(str(i)+'\t')
            for dataset in fold_i:
              f.write(str(dataset[0])+'\t'+str(dataset[1])+'\t')
            f.write('\n')
              
        print('K-Fold Validation Results')
        print('--------------------------------------')
        result = np.mean(fold_log, axis=0)
        print('Average train RMSE: {:.4f}'.format(result[0, 0]))
        print('Average train PCC:  {:.4f}'.format(result[0, 1]))
        print('Average validation RMSE: {:.4f}'.format(result[1, 0]))
        print('Average validation RMSE: {:.4f}'.format(result[1, 1]))
        print('Average test RMSE: {:.4f}'.format(result[2, 0]))
        print('Average test RMSE: {:.4f}'.format(result[2, 1]))
        run.finish()


if __name__ == '__main__':

    p = configargparse.ArgParser()
    p.add('-c', '--my-config', required=True, is_config_file=True, help='config file path')
    p.add_argument('--spk_group', type=int, help='Specify train/dev/test speakers group')
    p.add_argument('--layer_from', type=int, help='Specify start layers of ssl feature')
    p.add_argument('--layer_to', type=int, help='Specify end layers of ssl feature')
    p.add_argument('--fold_number', type=int, default=6, help='K fold validation number')
    p.add_argument('--exclude_tr', action="store_true", help='if true, the tr sensors in English train set will be ignored')

    args = configs.parse(p)

    manual_seed = 2
    random.seed(manual_seed)
    np.random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed(manual_seed)
    torch.cuda.manual_seed_all(manual_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



    train(args)
