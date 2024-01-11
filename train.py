
import torch
import random
from models import Modern_DBLSTM_1, DBLSTM_LayerNorm, Traditional_BLSTM_2, Modern_BLSTM_2
import matplotlib.pyplot as plt
import numpy as np
#from nnmnkwii.datasets import FileSourceDataset
from data_utils import pad_collate
import configargparse
from configs import configs
from torch.utils.tensorboard import SummaryWriter
from utils import EarlyStopping
import os
from torch.utils.data import Dataset, DataLoader, random_split
import wandb
from sklearn.model_selection import KFold
import scipy.signal

wandb.login()

class MyDataset(Dataset): #HY
    def __init__(self, processed_root, spks, feature):
        self.root_path = processed_root
        self.x = []
        self.y = []
        #self.device = device
        for spk in spks:
            for file in os.listdir(os.path.join(self.root_path, spk, feature)):
                x = np.load(os.path.join(self.root_path, spk, feature, file))
                y = np.load(os.path.join(self.root_path, spk, "ema_final", file))
                y = scipy.signal.resample(y, num=len(x))
                # self.x.append(np.load(os.path.join(self.root_path, spk, "mfcc_final", file)))
                # self.y.append(np.load(os.path.join(self.root_path, spk, "ema_final", file)))
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
        
def worker_init_fn(worker_id):
    # After creating the workers, each worker has an independent seed that is initialized to the curent random seed + the id of the worker
    np.random.seed(manual_seed + worker_id)

def train(args):

    print(args.__dict__)
    writer = SummaryWriter()
    configs = args.__dict__

    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    print('device:', device)
    output_path = '/data2/yun/aai_expr/exp'

    if args.feature == 'wav2vec2':
        feature = 'wav2vec2_layer_12'

    # if args.MFCC:
    #     # Load MNGU0 features
    #     train_x = FileSourceDataset(MFCCSourceNPY("trainfiles.txt"))
    #     val_x = FileSourceDataset(MFCCSourceNPY("validationfiles.txt"))
    #     test_x = FileSourceDataset(MFCCSourceNPY("testfiles.txt"))
    # elif args.LSF:
    #     # Load LSF features
    #     train_x = FileSourceDataset(LSFSource("trainfiles.txt"))
    #     val_x = FileSourceDataset(LSFSource("validationfiles.txt"))
    #     test_x = FileSourceDataset(LSFSource("testfiles.txt"))
    # else:
    #     raise NameError("No frontend loaded!")

    # if args.art_norm:

    #     train_y = FileSourceDataset(NormalisedArticulatorySource("trainfiles.txt"))
    #     val_y = FileSourceDataset(NormalisedArticulatorySource("validationfiles.txt"))
    #     test_y = FileSourceDataset(NormalisedArticulatorySource("testfiles.txt"))

    #     # Loads normalisaion means and standard deviations for metric handling - 4 times because z-score norm
    #     mngu0_mean = np.genfromtxt("mngu0_ema/all_normalised/norm_parms/ema_means.txt")
    #     ema_mean = torch.FloatTensor(mngu0_mean[:12]).to(device)
    #     mngu0_std = np.genfromtxt("mngu0_ema/all_normalised/norm_parms/ema_stds.txt")
    #     ema_std = torch.FloatTensor(mngu0_std[:12]).to(device)

    # else:
    #     train_y = FileSourceDataset(ArticulatorySource("trainfiles.txt"))
    #     val_y = FileSourceDataset(ArticulatorySource("validationfiles.txt"))
    #     test_y = FileSourceDataset(ArticulatorySource("testfiles.txt"))

    # dataset = NanamiDataset(train_x, train_y)
    # dataset_val = NanamiDataset(val_x, val_y)
    # dataset_test = NanamiDataset(test_x, test_y)

    spks = ['F01', 'F02', 'F03', 'M01', 'M02', 'M03']
    test_spks = ['F04', 'M04']
    kf = KFold(n_splits=6, shuffle=False)
    
    run = wandb.init(
        # Set the project where this run will be logged
        project="haskins_inversion",
        name="kf_" + str(args.feature),
        # Track hyperparameters and run metadata
        config=args.__dict__,)

    fold_log = np.zeros((6, 3, 2))
    
    for fold, (train_index, test_index) in enumerate(kf.split(spks)):
        #feature = 'wav2vec2_layer_12'

        print('FOLD {}'.format(fold))
        print('-----------------------------')
        
        train_spks = [spks[i] for i in train_index]
        val_spks = [spks[i] for i in test_index]
    
        train_dataset = MyDataset("/data2/yun/aai_expr/Preprocessed_data_HY", train_spks, feature)
        val_dataset = MyDataset("/data2/yun/aai_expr/Preprocessed_data_HY", val_spks, feature)
        test_dataset = MyDataset("/data2/yun/aai_expr/Preprocessed_data_HY", test_spks, feature)
    # val_set, test_set = random_split(test_dataset, [int(0.5*len(test_dataset)),int(0.5*len(test_dataset))])

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

        if args.train:
            writer.add_hparams(args.__dict__,{'started':True})
    
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
    
                    # if loss.isnan():
                    #     print('Inputs: ', torch.isnan(inputs).any())
                    #     print(outputs.size(), targets.size())
                    #     print('Outputs: {}\n Targets: {}'.format(torch.isnan(outputs).any(), torch.isnan(targets).any()))
                    #     # print((((outputs - targets) * mask) ** 2.0))
                    #     # print(torch.sum(((outputs - targets) * mask) ** 2.0))
                    #     # print(torch.sum(mask))
    
                    if targets.shape[0] == args.batch_size:
                        total_loss += loss.item()
                        #print(loss.item())
                    else:
                        total_loss += loss.item() * (targets.shape[0] / args.batch_size)
    
                    y = (outputs * mask).detach().cpu().numpy()
                    y_pred = (targets * mask).detach().cpu().numpy()
                    # print(y.shape, y_pred.shape)
                    # vx = y - torch.mean(y, dim=1, keepdim=True)
                    # vy = y_pred - torch.mean(y_pred, dim=1, keepdim=True)
    
                    #pearson = np.zeros((y.size()[0], args.num_classes))
                    for b in range(y.shape[0]):
                        vy = y[b].T
                        vy_pred = y_pred[b].T
                        # print(vy.shape, vy_pred.shape)
                        pearson = np.diagonal(np.corrcoef(vy, vy_pred), offset=vy.shape[0])
                        # print(pearson.shape)
                    all_pearson.append(pearson)
                        
                    #     for k in range(args.num_classes):
                    #         #p = torch.sum(vx[b,:,k] * vy[b,:,k]) / (torch.sqrt(torch.sum(vx[b,:,k] ** 2)) * torch.sqrt(torch.sum(vy[b,:,k] ** 2)))
                    #         #print(p)
                    #         pearson[b][k] = p.item()
                    # #print(pearson.shape)
                    # all_pearson.append(pearson)
                            # should be a scalar
                        #pearson = np.array(pearson).reshape((1, args.num_classes))
    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
    
                total_loss = np.sqrt(total_loss / len(train_loader))
                writer.add_scalar('Loss/Train', total_loss, epoch + 1)
                epoch_log[epoch,0,0] = total_loss
                epoch_log[epoch,0,1] = np.mean(np.concatenate(all_pearson))
    
                #all_pearson = np.mean(np.concatenate(all_pearson))
                # metrics['train_rmse_fold'+str(fold)], metrics['train_pcc_fold'+str(fold)] = total_loss, np.mean(np.concatenate(all_pearson))
                metrics['train_rmse_fold'+str(fold)], metrics['train_pcc_fold'+str(fold)] = epoch_log[epoch,0,0], epoch_log[epoch,0,1]
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
                    writer.add_scalar('Loss/Validation', total_loss, epoch + 1)
                    epoch_log[epoch, 1, 0] = total_loss
                    epoch_log[epoch, 1, 1] = np.mean(np.concatenate(all_pearson))
    
                    metrics['validation_rmse_fold'+str(fold)], metrics['validation_pcc_fold'+str(fold)] = epoch_log[epoch, 1, 0], epoch_log[epoch, 1, 1]
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
                    writer.add_scalar('Loss/Test', total_loss, epoch + 1)
                    epoch_log[epoch, 2, 0] = total_loss
                    epoch_log[epoch, 2, 1] = np.mean(np.concatenate(all_pearson))
    
                    metrics['test_rmse_fold'+str(fold)], metrics['test_pcc_fold'+str(fold)] = epoch_log[epoch, 2, 0], epoch_log[epoch, 2, 1]
                    print('Epoch [{}/{}], Test RMSE: {:.4f}, PCC: {:.4f}'.format(epoch + 1, args.num_epochs, total_loss, epoch_log[epoch, 2, 1]))
    
                torch.cuda.empty_cache()
    
                wandb.log(metrics)
                
                #print(epoch_log[epoch, :])
                if es.call(epoch_log[epoch, 1, 0]):
                    #best_id = np.argmin(epoch_log[:i, 1])
                    best_id = epoch
                    #print("Early stopping! Best test loss: ", epoch_log[epoch,2])
                    break

            fold_log[0] = epoch_log[epoch, :].copy()
            print(fold_log)
            torch.save(model.state_dict(), os.path.join(output_path, feature + '_fold' + str(fold) + '.pt'))
            
            writer.add_hparams(args.__dict__,
                          {'hparam/train': epoch_log[best_id,0], 'hparam/val': epoch_log[best_id,1], 'hparam/test': epoch_log[best_id,2]})
            writer.close()
        else:
            model.load_state_dict(torch.load("model_dblstm_48.ckpt"))
    
            for i, sample in enumerate(test_loader):
                xx_pad, yy_pad, _, _, _ = sample
                inputs = xx_pad.to(device)
                targets = yy_pad.to(device)
    
                predicted = model(inputs).detach().cpu().numpy()
                targets = targets.detach().cpu().numpy()
                plt.plot(targets[0,:,0], label='Original data')
                plt.plot(predicted[0,:,0], label='Fitted line')
                plt.legend()
                plt.show()

    print('K-Fold Validation Results')
    print('--------------------------------------')
    result = np.mean(fold_log, axis=0)
    print('Average train RMSE: {:.4f}'.format(result[0, 0]))
    print('Average train PCC:  {:.4f}'.format(result[0, 1]))
    print('Average validation RMSE: {:.4f}'.format(result[1, 0]))
    print('Average validation RMSE: {:.4f}'.format(result[1, 1]))
    print('Average test RMSE: {:.4f}'.format(result[2, 0]))
    print('Average test RMSE: {:.4f}'.format(result[2, 1]))


if __name__ == '__main__':

    p = configargparse.ArgParser()
    p.add('-c', '--my-config', required=True, is_config_file=True, help='config file path')
    #p.add('--savepath', required=True, type=str, help='Model save path')
    p.add_argument('--spk_group', type=int, help='Specify train/dev/test speakers group')

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