'''
Classwise Correction 
'''

# Generic Libraries
import os
import json
import sys
import math
import cv2
import argparse
import imageio
import copy
import wandb
import warnings
from tqdm import tqdm
import numpy as np
import sklearn.metrics as sm
from datetime import datetime
from collections import defaultdict

# Torch Imports 
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

# Library Parameters
np.set_printoptions(threshold=sys.maxsize)
warnings.filterwarnings("ignore")
cv2.setNumThreads(0)

# WandB Login
wid = 
WANDB_API_KEY="4169d431880a3dd9883605d90d1fe23a248ea0c5"
WANDB_ENTITY="amber1121"
os.environ["WANDB_RESUME"] = "allow"
os.environ["WANDB_RUN_ID"] = wid
wandb.init(project="GCN Experiments",id=wid,resume='allow')

classes = ['Hole(Physical)','Hole(Virtual)','Character Line Segment', 'Character Component','Picture','Decorator','Library Marker', 'Boundary Line', 'Physical Degradation']

# File Imports
from utilities import utils
from utilities import *
from datasets.edge_imageprovider import *
import datasets.edge_imageprovider as image_provider
from models.combined_model import Model
from losses.Hausdorff_loss import AveragedHausdorffLoss
from losses.fm_maps import compute_edts_forPenalizedLoss

# CUDA Device Settings 
if torch.cuda.is_available():
    print('------------------- Device Information -------------------------------')
    print('Allocated GPUs : {} , CPUs : {}'.format(torch.cuda.device_count(),os.cpu_count()))
    print('__CUDNN VERSION  : {} '.format(torch.backends.cudnn.version()))
    print('__Number CUDA Devices  : {}'.format(torch.cuda.device_count()))
    if(torch.cuda.device_count()<2):
        print('Only {} GPUs  allocated , minimum required ! '.format(torch.cuda.device_count()))
    else: 
        device0 = torch.device("cuda:0")
        device1 = torch.device("cuda:1")
else:
    print(" NO GPU ALLOCATED , Exiting . ")
    sys.exit()


def create_folder(args):
    try : 
        path=args.expdir
        pathCheckpoints=args.expdir+'checkpoints/'
        # pathVis=args.expdir+'vis/'
        os.system('mkdir -p %s' % path)
        os.system('mkdir -p %s' % pathCheckpoints)
        print('Experiment folder created ')

        # wid =wandb.util.generate_id() 
        WANDB_API_KEY="4169d431880a3dd9883605d90d1fe23a248ea0c5"
        WANDB_ENTITY="amber1121"
        os.environ["WANDB_RESUME"] = "allow"
        os.environ["WANDB_RUN_ID"] = args.wid
        wandb.init(project="GCN Experiments",id=args.wid,resume='allow')
        print('WandB login completed ..')

    except Exception as ex : 
        print('Folder Creation Error , Exiting : {}'.format(ex))
        sys.exit()


'''
We need to add exp dir part to the json , apart from regular 
resume flag , if weighted_trainloader flag too , weighted_epoch number
'''

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--title',type=str,default=None)
    parser.add_argument('--exp', type=str)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--expdir',type=str,default=None)
    parser.add_argument('--hd_weight_file',type=str,default='hd_enc.json')
    parser.add_argument('--vis',type=bool,default=False)
    parser.add_argument('--wid',type=str,default=str(wandb.util.generate_id()))
    # HD weighing related part.
    parser.add_argument('--gamma',type=float,default=0.5)
    parser.add_argument('--increment',type=float,default=0.5)
    args = parser.parse_args()
    parser.add_argument('--vis_file',type=str,default='val_opts_{}.json'.format(args.title))
    args = parser.parse_args()
    return args
'''
Remember to add all classes into the experiment json , as well as 
val , test , train and train-val details . 
'''

# Normal Loaders wrt class weights 
def get_data_loaders(opts, DataProvider):
    print('Building Train Set dataloaders..')
    train_split = 'train'
    train_val_split ='val'
    dataset_train = DataProvider(split=train_split, opts=opts[train_split], mode=train_split)
    dataset_val = DataProvider(split=train_val_split, opts=opts[train_val_split], mode=train_val_split)
    weights, label_to_count = dataset_train.getweight()
    weights = torch.DoubleTensor(weights)
    # Pytorch Official Sampler - replacement is set to False (by default) , he isn't changing it .
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
    train_loader = DataLoader(dataset_train, batch_size=opts[train_split]['batch_size'],
                              sampler = sampler, shuffle=False, num_workers=opts[train_split]['num_workers'],
                              collate_fn=image_provider.collate_fn)

    val_loader = DataLoader(dataset_val, batch_size=opts[train_val_split]['batch_size'],
                            shuffle=False, num_workers=opts[train_val_split]['num_workers'],
                            collate_fn=image_provider.collate_fn)
    
    return train_loader, val_loader, label_to_count

class Trainer(object):
    def __init__(self, args, opts):

        # Create the directories 
        create_folder(args)

        self.opts = opts
        self.args=args
        
        # Edit in the json as well , defaulted to 0.0
        self.classwise_hd=self.instance_dict={}
        self.alpha=self.opts['alpha']
        self.beta=self.opts['beta']
        self.gamma=self.args.gamma
        self.increment=self.args.increment
        self.number=1
        self.hd_classwise_global_thresh={}
        self.hd_classwise={}
        self.trackInstances={}
        self.trackLoss={}
        self.track_classwise_instances={}

        
        self.global_step = 0
        self.epoch = 0
        self.opts = opts
        self.args=args
        # self.weights=None

        
        self.fp_loss_fn = nn.MSELoss()
        self.gcn_loss_sum_train = 0
        self.gcn_loss_sum_val = 0
        self.x1 = []
        self.y1 = []
        self.y2 = []

        self.hausdorff_loss = AveragedHausdorffLoss()
        self.model_path = self.opts['encoder_weight_path']
        self.model = Model(self.opts)

        # TrainLoader instantiation 
        self.train_loader, self.val_loader, self.label_to_count = get_data_loaders(self.opts['dataset'], image_provider.DataProvider)
  
        
        # OPTIMIZER
        no_wd = []
        wd = []
        print('Weight Decay applied to: ')
        ct = 0
        for name, p in self.model.named_parameters():
            ct += 1
            if not p.requires_grad:
                # No optimization for frozen params
                continue
            # print(ct, p)
            if 'bn' in name or 'bias' in name:
                no_wd.append(p)
            else:
                wd.append(p)

        # Allow individual options
        self.optimizer = optim.Adam(
            [
                {'params': no_wd, 'weight_decay': 0.0},
                {'params': wd}
            ],
            lr=self.opts['lr'],
            weight_decay=self.opts['weight_decay'],
            amsgrad=False)

        self.lr_decay = optim.lr_scheduler.StepLR(self.optimizer, step_size=self.opts['lr_decay'],
                                                  gamma=0.1)
        if args.resume is not None:
            self.resume(args.resume)
        
        # HD Computation 
        self.load_hd_encoder()

    #classes = ['Hole(Physical)','Hole(Virtual)','Character Line Segment', 'Character Component','Picture','Decorator','Library Marker', 'Boundary Line', 'Physical Degradation']

    def save_checkpoint(self, epoch):
        save_state = {
            'epoch': epoch,
            'global_step': self.global_step,
            'gcn_state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'lr_decay': self.lr_decay.state_dict(),
            'epoch_loss_prev':self.trackLoss[self.epoch-1],
            'num_instances':len(self.trackInstances[self.epoch-1]),
            'threshold_info_HP':self.hd_classwise_global_thresh['Hole(Physical)'],
            'threshold_info_HV':self.hd_classwise_global_thresh['Hole(Virtual)'],
            'threshold_info_CLS':self.hd_classwise_global_thresh['Character Line Segment'],
            'threshold_info_CC':self.hd_classwise_global_thresh['Character Component'],
            'threshold_info_P':self.hd_classwise_global_thresh['Picture'],
            'threshold_info_D':self.hd_classwise_global_thresh['Decorator'],
            'threshold_info_LM':self.hd_classwise_global_thresh['Library Marker'],
            'threshold_info_BL':self.hd_classwise_global_thresh['Boundary Line'],
            'threshold_info_PD':self.hd_classwise_global_thresh['Physical Degradation']
        }
        
        save_name = os.path.join(self.args.expdir, 'checkpoints', 'epoch_encoder_%d_step%d.pth' \
                                 % (epoch, self.global_step))
        torch.save(save_state, save_name)
        print('Saved model @ Epoch : {}'.format(self.epoch))

    def resume(self, path):
        print(' GCN Training Resumed .. !')
        self.model_path=path
        save_state = torch.load(self.model_path)
        try :
            self.model.load_state_dict(save_state['gcn_state_dict'])
            print('Model loaded ..... GCN  ')
        except Exception as ex :
            print('Model loading error ___ {}'.format(ex)) 
        self.global_step = save_state['global_step']
        self.epoch = save_state['epoch']
        self.optimizer.load_state_dict(save_state['optimizer'])
        self.trackLoss[self.epoch-1]=save_state['epoch_loss_prev']
       
        self.hd_classwise_global_thresh={} 
        self.hd_classwise_global_thresh['Hole(Physical)']=save_state['threshold_info_HP']
        self.hd_classwise_global_thresh['Hole(Virtual)']=save_state['threshold_info_HV']
        self.hd_classwise_global_thresh['Character Line Segment']= save_state['threshold_info_CLS']
        self.hd_classwise_global_thresh['Character Component']=save_state['threshold_info_CC']
        self.hd_classwise_global_thresh['Picture']=save_state['threshold_info_P']
        self.hd_classwise_global_thresh['Decorator']=save_state['threshold_info_D']
        self.hd_classwise_global_thresh['Library Marker']=save_state['threshold_info_LM']
        self.hd_classwise_global_thresh['Boundary Line']=save_state['threshold_info_BL']
        self.hd_classwise_global_thresh['Physical Degradation']=save_state['threshold_info_PD']

        
       
        print('Model reloaded to resume from Epoch %d, Global Step %d from model at %s' % (self.epoch, self.global_step, path))
    
    def load_hd_encoder(self):
        # Read JSON file 
        with open(self.args.hd_weight_file,'r') as file:
            hd_data=json.load(file)
        
        self.classwise_hd_initial={}
        self.classwise_hd_storage={}
        self.instance_dict={}
        countLen=0
        # Create a dictionary with all classes 
     
        for cl in classes:
            self.classwise_hd_initial[cl]={'len':0,'hd':0.0,'hd95':0.0,'std':0.0,'max':0.0,'min':0.0}
            self.classwise_hd_storage[cl]=[]
            self.track_classwise_instances[cl]=0

        # Run through all the instances in the json file , match the class and sum the classwise threshold 
        # embedd self.number into it . 
       
        for i in hd_data: 
            key=list(i.keys())[0]
            val=list(i.values())[0]
            hd=val['hd']
            label=val['label']

            # Storing the value 
            
            self.instance_dict[key]={'hd':hd,'label':label}

            if(label in classes):
                self.classwise_hd_initial[label]['len']+=1
                self.classwise_hd_initial[label]['hd']+=hd
                self.classwise_hd_storage[label].append(hd)
                
    
       
        for key in classes:
            # self.classwise_hd_initial[key]['hd']/=int(self.classwise_hd_initial[key]['len'])
            hd_arr=np.asarray(self.classwise_hd_storage[key])
            self.classwise_hd_initial[key]['hd']=np.mean(hd_arr)
            print('Class : {} , HD Threshold : {}'.format( key,self.classwise_hd_initial[key]['hd']))
            self.classwise_hd_initial[key]['std']=np.std(hd_arr)
            self.classwise_hd_initial[key]['max']=np.max(hd_arr)
            self.classwise_hd_initial[key]['min']=np.min(hd_arr)

        self.hd_classwise_global_thresh=copy.deepcopy(self.classwise_hd_initial)
        # Intial threshold 
        for key in self.hd_classwise_global_thresh.keys():
            self.hd_classwise_global_thresh[key]['hd']=int(self.gamma*(self.hd_classwise_global_thresh[key]['hd']))            



    # Non-Weighted Traning .
    def loop(self):
        for epoch in range(self.epoch, self.opts['max_epochs']):
            self.epoch = epoch
            self.lr_decay.step()
            wandb.log({'current_lr':self.optimizer.param_groups[0]['lr'],'epoch':self.epoch})
            self.train(epoch)
            print('Epoch : {} ---- Instances Length : {} '.format(self.epoch,len(self.train_loader)))

    def train(self,epoch):
        print('Starting training...')
        self.model.train()
        losses = []
        gcn_losses = []
        poly_losses = []
        # class_losses = []
        accum = defaultdict(float)
        countTrue=countFalse=0

        self.trackInstances[self.epoch]=[]
        self.trackLoss[self.epoch]=0.0

        if(self.epoch>0):
            self.validate()
            self.save_checkpoint(epoch)

        for step, data in enumerate(self.train_loader):
            print('GCN Train Epoch --- {} Step --- {} '.format(self.epoch,self.global_step))

            id=data['id'] 
            class_label=data["cm_label"][0]
            # self.hd_classwise_global_thresh[key]['hd']=int(self.gamma*(self.hd_classwise_global_thresh[key]['hd']))            

            hd_thresh=int(self.hd_classwise_global_thresh[str(class_label)]['hd'])
            hd_enc=int(self.instance_dict[str(id[0])]['hd'])
            print('ID : {} Label : {}  HD_Threshold :{} HD_Enc: {}'.format(str(id[0]),class_label,hd_thresh,hd_enc))
    
            if(hd_enc<=hd_thresh):
                try :
                    img = data['img']
                    img = torch.cat(img)
                    img = img.view(-1, img.shape[0], img.shape[1], 3)
                    img = torch.transpose(img, 1, 3)
                    img = torch.transpose(img, 2, 3)
                    img = img.float()
                    # ----------- GT formation -----------
                    bbox = torch.from_numpy(np.asarray(data['bbox'])).to(device0)
                    palm_leaf_pred1=copy.deepcopy(data['img_orig'][0])
                    
                    w_img = torch.tensor(data["w"]).to(device0).float()
                    h_img = torch.tensor(data["h"]).to(device0).float()
                    dp_poly = data['actual_gt_poly']
                    
                    dp = data['actual_gt_poly11']

                    # MCNN + GCN Output  - All are in device 1 so , its fine 
                    # Inputs -> Image , BBox -> Device 0 | It doesn't manner the source of img , we care changing it later .
                    
                    output_dict, poly_logits = self.model(img, bbox, 'train')

                    poly_mask = np.zeros((poly_logits[:,0,:,:].shape[1], poly_logits[:,0,:,:].shape[2]),np.float32)
                    poly_mask  = utils.get_poly_mask(dp_poly.cpu().numpy()[0],poly_mask)

                    edge_mask77 = np.zeros((poly_logits[:,0,:,:].shape[1], poly_logits[:,0,:,:].shape[2]),np.float32)
                    edge_mask77  = utils.get_edge_mask(dp_poly.cpu().numpy()[0],edge_mask77)

                    n_poly = (np.sum(poly_mask)).astype(np.float32)
                    
                    back_mask = 1.0-poly_mask
                    n_back = (np.sum(back_mask)).astype(np.float32)

                    n_edge = (np.sum(edge_mask77)).astype(np.float32)
                    w1,h1 = poly_logits[:,0,:,:].shape[1],poly_logits[:,0,:,:].shape[2]


                    # ----------- Distance Maps Initialization -----------
                    DT_mask = compute_edts_forPenalizedLoss(edge_mask77)
                    DT_mask = torch.from_numpy(np.asarray(DT_mask)).to(device0)
                    DT_mask = DT_mask.float()

                    
                    # ----------- BCE Loss -----------
                    self.p_n2 = torch.ones([w1,h1], dtype= torch.float32)
                    self.p_n2 = (self.p_n2*(n_back/n_poly)).to(device0)
                    
                    self.poly_loss_fn = nn.BCEWithLogitsLoss(pos_weight = self.p_n2, reduction = 'none')
                    

                    poly_mask = torch.from_numpy(np.asarray(poly_mask)).to(device0)
                    poly_mask = poly_mask.view(1,poly_mask.shape[0],poly_mask.shape[1]).to(device0)
                    poly_loss1 = self.poly_loss_fn(poly_logits[:,0,:,:], poly_mask.to(device0))

                    # ----------- Focal Loss -----------
                    pt3 = torch.exp(-poly_loss1)
                    poly_loss1 = ((1-pt3))**2 * poly_loss1

                    poly_loss1 = poly_loss1*DT_mask
                    poly_loss1 = torch.mean(poly_loss1)

                    gt_label = torch.tensor(data["gt_label"]).to(device0)
                    

                    # ----------- classifier loss ----- # Not in use now . 
                    # self.edge_loss_fn1 = nn.CrossEntropyLoss(weight = self.class_weight)
                    # class_loss1 = 0


                    pred_cps = (output_dict['pred_polys'][-1]).float()
                    pred_cps88 = pred_cps
                    n_points = output_dict['n_points']
                    hull_binary = output_dict['hull_binary'].float().to(device0)
                    hull_original = output_dict['hull_original'].float().to(device0)

                    pred_cps5 = pred_cps88.detach().cpu().numpy()

                    # ----------- Uniform Sampling of target points on GT contour -----------
                    dp = utils.uniformsample_batch(dp, n_points)
                    dp = dp[0]
                    dp = dp.cpu().numpy()
                    dp = np.asarray(dp)
                    

                    x = dp[:, 1]/float(h_img[0])
                    y = dp[:, 0]/float(w_img[0])

                    dp_x = x
                    dp_y = y

                    dp_x = (torch.from_numpy(dp_x)).view(-1, n_points, 1)
                    dp_y = (torch.from_numpy(dp_y)).view(-1, n_points, 1)
                    dp = torch.cat((dp_x, dp_y), dim=2).to(device0)


                    dpf = dp.cpu().numpy()
                    dpfer = dp.cpu().numpy()

                    
                    # ----------- Hausdorff Loss -----------
                    han_loss = self.hausdorff_loss(pred_cps88[0,:,:].float(), dp[0,:,:].float())

                    # Its frozen . MCNN weights .
                    if self.opts["enc_freeze"]:
                        loss_v = han_loss
                    else:
                        loss_v = han_loss + 200*poly_loss1

                    
                    loss_sum = han_loss
                    self.gcn_loss_sum_train = han_loss
                    poly_loss_sum = poly_loss1
                    # class_loss_sum = class_loss1

                    self.optimizer.zero_grad()

                    # Backpropagation of loss 
                    loss_v.backward()
                    # Store in a dictionary
                    self.trackInstances[self.epoch].append(id)
                    self.trackLoss[self.epoch]+=loss_v
                    self.track_classwise_instances[class_label]+=1

                    if 'grad_clip' in self.opts.keys():
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.opts['grad_clip'])

                    self.optimizer.step()
                    
                    w1_img = torch.tensor(data["w"]).to(device0).float()
                    h1_img = torch.tensor(data["h"]).to(device0).float()

                    pred_cps = (output_dict['pred_polys'][-1]).float()
                    pred_cps5 = pred_cps[0]
                    pred_x = ((pred_cps5[:, 0] * h1_img[0]).view(n_points,1)).int()
                    pred_y = ((pred_cps5[:, 1] * w1_img[0]).view(n_points,1)).int()

                    pred = torch.cat((pred_y, pred_x), dim=1)
                    pred = pred.cpu().numpy()
                    pred67 = pred
                    pred = np.asarray(pred)

                    
                    mask_h = int(h1_img[0].cpu().numpy())
                    mask_w = int(w1_img[0].cpu().numpy())
                    
                    mask = np.zeros((mask_h, mask_w))
                    cv2.fillPoly(mask, np.int32([pred]), [1])

                    original_mask = np.asarray(data["original_mask"][0])
                    original_mask = original_mask.astype(np.uint8)
                    original_mask = (original_mask*255).astype(np.uint8)

                    pred_mask = mask.astype(np.uint8)
                    pred_mask = (pred_mask*255).astype(np.uint8) 

                
                    if(self.gcn_loss_sum_train>1000):
                        print('Heavy Loss : Faulty Instances --- > {} {} {} {} '.format(self.epoch,step,self.gcn_loss_sum_train,data['id']))
                        continue
                    
                    
                    loss = loss_sum
                    losses.append(loss_sum)
                    gcn_losses.append(self.gcn_loss_sum_train)
                    poly_losses.append(poly_loss_sum)
                    # class_losses.append(class_loss_sum)

                    accum['loss'] += float(loss)
                    accum['gcn_loss'] += float(self.gcn_loss_sum_train)
                    accum['edge_loss'] += float(poly_loss_sum)
                    # accum['vertex_loss'] += float(class_loss_sum)
                    accum['length'] += 1

                    print('GCN_loss : {} at {} , {} '.format(float(self.gcn_loss_sum_train),step,self.epoch))


                    if (step % self.opts['print_freq'] == 0):
                        # Mean of accumulated values
                        for k in accum.keys():
                            if k == 'length':
                                continue
                            accum[k] /= accum['length']

                        print("[%s] Epoch: %d, Step: %d, Loss: %f, GCN Loss: %f, Edge Loss: %f " % (
                        str(datetime.now()), epoch, self.global_step, accum['loss'], accum['gcn_loss'], accum['edge_loss']))
                        accum = defaultdict(float)
                    
                    countTrue+=1
        
                except Exception as exp : 
                    print('Dataloading Error : {}'.format(exp))
                    countFalse+=1
                    continue

            else: 
                countFalse+=1
                continue

            self.global_step += 1
        
        # Loss Summation  & Computation . 
        avg_epoch_loss = 0.0    
        for i in range(len(losses)):
            avg_epoch_loss += losses[i]

        avg_epoch_loss = avg_epoch_loss / len(losses)
        self.trackLoss[self.epoch]=self.trackLoss[self.epoch]/len(losses)

        if(self.epoch>=2):
            # Compute the HD threshold for next loop ; if loss platues.
            diff=(self.trackLoss[self.epoch-1]-self.trackLoss[self.epoch])
            diff=np.asscalar(diff)
            wandb.log({'loss_diff':int(diff),'epoch':self.epoch})
            if(abs(diff)<5.0):
                for key in self.hd_classwise_global_thresh.keys():
                    self.hd_classwise_global_thresh[key]['hd']+=(self.increment)*int(self.classwise_hd_initial[key]['std'])

        self.gcn_loss_sum_train = avg_epoch_loss   
        # WandB Logging
        wandb.log({'train_gcn_avg_epoch_loss':self.gcn_loss_sum_train, 'epoch': self.epoch})
        wandb.log({'track_instances ':len(self.trackInstances[self.epoch]),'epoch':self.epoch})

        for key in classes:
            wandb.log({'{}_hd_threshold'.format(str(key)):self.hd_classwise_global_thresh[key]['hd'],'epoch':self.epoch})
            
        print("Average Epoch %d loss is : %f" % (epoch, avg_epoch_loss))
        print('Statistics :   Epoch --- {} , Invalid Samples --- {} , Valid Samples --- {}'.format(self.epoch,countFalse,countTrue))
        torch.cuda.empty_cache()
    
    def validate(self):
        
        print('GCN Validation @ Epoch :  {} '.format(self.epoch))

        self.args.vis=True
        self.model.eval()
        losses = []
        gcn_losses = []
        poly_losses = []
        urls = []
        class_losses = []
        pred_cm = []
        gt_cm = []
        avg_acc = 0.0
        avg_iou = 0.0

        final_ious = {} 
        final_acc = {}
        final_hd = {} 
        final_hd95 = {}

        classes = ['Hole(Physical)','Hole(Virtual)','Character Line Segment', 'Character Component','Picture','Decorator','Library Marker', 'Boundary Line', 'Physical Degradation']
        
        tool_dict = []
        testcount = {}
        testarr=[]

        for clss in classes: 
            final_ious[clss] = 0.0
            final_acc[clss] = 0.0
            final_hd[clss] = 0.0
            final_hd95[clss] = 0.0

        with torch.no_grad():

            for step, data in enumerate(self.val_loader):

                try : 
                    print('Validation Step : {} ,Current Epoch : {} '.format(step,self.epoch))
      
                    id=str(data['id'][0])      
                    img = data['img']
                    img = torch.cat(img)
                    img = img.view(-1, img.shape[0], img.shape[1], 3)
                    img = torch.transpose(img, 1, 3)
                    img = torch.transpose(img, 2, 3)
                    img = img.float()

                    self.optimizer.zero_grad()

                    bbox = torch.from_numpy(np.asarray(data['bbox'])).to(device0)

                    w1_img = torch.tensor(data["w"]).to(device0).float()
                    h1_img = torch.tensor(data["h"]).to(device0).float()
                    
                    dp101 = data['actual_gt_poly11'].detach().clone()
                    dp = data['actual_gt_poly11']
                    dp_poly = data['actual_gt_poly']
                    
                    # Output from the model.
                    output_dict, poly_logits = self.model(img, bbox, 'val')

                    # ----------- Mask extraction for metrics -----------
                    poly_logits88 = torch.sigmoid(poly_logits[0,0,:,:]).cpu().numpy()
                    yy = poly_logits88 > 0.5
                    yy = yy+0
                    poly_logits88 = yy.astype(np.float32)

                    poly_mask = np.zeros((poly_logits[:,0,:,:].shape[1], poly_logits[:,0,:,:].shape[2]),np.float32)
                    poly_mask  = utils.get_poly_mask(dp_poly.cpu().numpy()[0],poly_mask)

                    edge_mask77 = np.zeros((poly_logits[:,0,:,:].shape[1], poly_logits[:,0,:,:].shape[2]),np.float32)
                    edge_mask77  = utils.get_edge_mask(dp_poly.cpu().numpy()[0],edge_mask77)

                    n_poly = (np.sum(poly_mask)).astype(np.float32)

                    back_mask = 1.0-poly_mask
                    n_back = (np.sum(back_mask)).astype(np.float32)

                    w1,h1 = poly_logits[:,0,:,:].shape[1],poly_logits[:,0,:,:].shape[2]

                    # ----------- Distance maps computation -----------
                    DT_mask = compute_edts_forPenalizedLoss(edge_mask77)
                    DT_mask = torch.from_numpy(np.asarray(DT_mask)).to(device0)
                    DT_mask = DT_mask.float()

                    # ----------- BCE Loss -----------
                    self.p_n2 = torch.ones([w1,h1], dtype= torch.float32)
                    self.p_n2 = (self.p_n2*(n_back/n_poly)).to(device0)            
                    self.poly_loss_fn = nn.BCEWithLogitsLoss(pos_weight = self.p_n2,reduction = 'none')

                    poly_mask = torch.from_numpy(np.asarray(poly_mask)).to(device0)
                    poly_mask = poly_mask.view(1,poly_mask.shape[0],poly_mask.shape[1]).to(device0)

                    poly_loss1 = self.poly_loss_fn(poly_logits[:,0,:,:], poly_mask.to(device0))
                    
                    # ----------- Focal Loss -----------
                    pt3 = torch.exp(-poly_loss1)
                    poly_loss1 = ((1-pt3))**2 * poly_loss1
                    
                    poly_loss1 = poly_loss1*DT_mask
                    

                    poly_loss1 = torch.mean(poly_loss1)

                    # self.edge_loss_fn1 = nn.CrossEntropyLoss(weight = self.class_weight)


                    gt_label = torch.tensor(data["gt_label"]).to(device0)

                    # try:
                    #     class_loss1 = self.edge_loss_fn1(class_prob.to(device0), gt_label.to(device0))
                    # except Exception as e:
                    #     # print(gt_label)
                    #     print(e)
                    #     continue

                    pred_cps = output_dict['pred_polys'][-1]

                    encoder_output = output_dict['hull_original']

                    pred_cps7 = pred_cps.detach().cpu().numpy()

                    n_points = output_dict['n_points']

                    dp = utils.uniformsample_batch(dp, n_points)
                    dp7 = dp[0].cpu().numpy()
                    dp = (torch.stack(dp)).to(device0)
                    dp_x = dp[:, :, 1].view(-1, n_points, 1)
                    dp_x = dp_x/float(h1_img[0])
                    dp_y = dp[:, :, 0].view(-1, n_points, 1)
                    dp_y = dp_y/float(w1_img[0])
                    dp = torch.cat((dp_x, dp_y), dim=2)
                    dp = torch.tensor(dp).to(device0)

                    dp_vis = dp[0]

                    dpf = dp.cpu().numpy()

                    pred_cps5 = pred_cps[0]

                    encoder_output = encoder_output[0]
                    pred_x = ((encoder_output[:, 0]).view(n_points,1)).int()
                    pred_y = ((encoder_output[:, 1]).view(n_points,1)).int()
                    encoder_output = torch.cat((pred_y, pred_x), dim=1)
                    encoder_output = encoder_output.cpu().numpy()
                    encoder_output = np.asarray(encoder_output)

                    pred_x = ((pred_cps5[:, 0] * h1_img[0]).view(n_points,1)).int()
                    pred_y = ((pred_cps5[:, 1] * w1_img[0]).view(n_points,1)).int()

                    pred = torch.cat((pred_y, pred_x), dim=1)
                    pred = pred.cpu().numpy()
                    pred67 = pred
                    pred = np.asarray(pred)


                    mask_h = int(h1_img[0].cpu().numpy())
                    mask_w = int(w1_img[0].cpu().numpy())
                    
                    mask = np.zeros((mask_h, mask_w))
                    cv2.fillPoly(mask, np.int32([pred]), [1])
                    
                    palm_leaf_pred = copy.deepcopy(data['img_orig'][0])
                    palm_leaf_pred1 = copy.deepcopy(data['img_orig'][0])
                    original_mask = np.asarray(data["original_mask"][0])


                    original_mask = original_mask.astype(np.uint8)
                    original_mask = (original_mask*255).astype(np.uint8)

                    pred_mask = mask.astype(np.uint8)
                    pred_mask = (pred_mask*255).astype(np.uint8)

                    
                    # ----------- Metrics -----------
                    iou1, accuracy1 = utils.compute_iou_and_accuracy(pred_mask, original_mask)
                    hd1 = utils.hd(pred_mask, original_mask)
                    hd951 = utils.hd95(pred_mask, original_mask)
                    
                    # self.args.vis=True

                
                    # encoder_output = output_dict['hull_original']
                    # encoder_output = encoder_output[0]
                    # pred_x = ((encoder_output[:, 0]).view(n_points,1)).int()
                    # pred_y = ((encoder_output[:, 1]).view(n_points,1)).int()
                    # encoder_output = torch.cat((pred_y, pred_x), dim=1)
                    # encoder_output = encoder_output.cpu().numpy()
                    # encoder_output = np.asarray(encoder_output)

                    # # ----------- Saving Images -----------
                    # # palm_leaf_pred=cv2.fillPoly(np.int32(palm_leaf_pred), np.int32([pred]), (210,0,0))
                    # # palm_leaf_pred=cv2.addWeighted(np.int32(palm_leaf_pred), 0.2, np.int32(palm_leaf_pred1), 1 - 0.3, 0, np.int32(palm_leaf_pred1))
                    # # palm_leaf_pred1=cv2.polylines(np.int32(palm_leaf_pred1), np.int32([dp7]), True, [255,255,255], thickness=1)
                    # palm_leaf_pred1=cv2.polylines(np.int32(palm_leaf_pred1), np.int32([pred]), True, (250,0,0), thickness=1)
                    # palm_leaf_pred1=cv2.polylines(np.int32(palm_leaf_pred1), np.int32([encoder_output]), True, (0,0,255), thickness=1)
                    # # for point in pred:
                    # #     palm_leaf_pred1=cv2.circle(np.int32(palm_leaf_pred1), (int(point[0]), int(point[1])), 1, (0, 0, 210), -1)
                    
                    # try : 
                    #     cv2.imwrite(self.args.expdir+'vis/'+'_'+str(step)+".jpg", np.int32(palm_leaf_pred1))
                    #     print('Image Saved ..... {} {} '.format(step,self.args.expdir+'vis/'+'_'+str(step)+".jpg"))
                    # except Exception as exp : 
                    #     print('Image Not Saving : {}'.format(exp))
                    #     sys.exit()
                        

                    # Uncomment the class part .
                    # class_prob = F.softmax(class_prob)
                    # class_prob = torch.squeeze(class_prob)
                    # class_label, index = torch.topk(class_prob, 1)
                    
                    # label46 = data["gt_label"][0]

                    avg_iou += iou1
                    avg_acc += hd1

                    class_lab = data['label'][0]

                    # gt_cmr = data['cm_label'][0]
                    # ----------- Confusion matrix parameters -----------
                    # pred_cm.append(classes[index[0]])
                    # gt_cm.append(gt_cmr)
                    

                    final_acc[class_lab] += accuracy1
                    final_ious[class_lab]+=iou1
                    final_hd[class_lab] += hd1
                    final_hd95[class_lab] += hd951

                    testarr.append(class_lab)

                    han_loss = self.hausdorff_loss(pred_cps[0,:,:].float(), dp[0,:,:].float())
                    loss_sum = han_loss
                    self.gcn_loss_sum_val = han_loss
                    poly_loss_sum = poly_loss1

                    # class_loss_sum = class_loss1
                    if(self.epoch%10==0 and self.epoch>0):
                        tool_dict.append({"id":str(id) ,"poly": (dp101[0].cpu().numpy()).tolist(), "encoder_output":encoder_output.tolist(), "gcn_output":pred.tolist() ,"image_url":data["image_url"][0], "bbox":bbox[0].tolist(), "label":data["cm_label"][0], "iou":iou1, "hd":hd1,"hd95":hd951})

                    loss = loss_sum
                    losses.append(loss)
                    gcn_losses.append(self.gcn_loss_sum_val)
                    poly_losses.append(poly_loss_sum)

                except Exception as ex :
                    print('Validation Exception , ignore : {} --------------{}'.format(ex,str(data['id'])))
                    continue

                # class_losses.append(class_loss_sum)

        if(self.epoch%10==0 and self.epoch>0):
            with open(self.args.expdir+'/'+args.vis_file,"w",encoding='utf-8') as outfile:
                json.dump(tool_dict,outfile,ensure_ascii=False)

        # cm = sm.confusion_matrix(gt_cm, pred_cm, labels = ['Hole(Physical)','Character Line Segment', 'Character Component','Picture','Decorator','Library Marker','Boundary Line','Physical Degradation'])
        # print(cm)

        avg_epoch_loss = 0.0
        avg_gcn_loss = 0.0
        avg_poly_loss = 0.0
        # avg_class_loss = 0.0


        for i in range(len(losses)):
            avg_epoch_loss += losses[i]
            avg_gcn_loss += gcn_losses[i]
            avg_poly_loss += poly_losses[i]
            # avg_class_loss += class_losses[i]

        for ij in testarr:
            testcount[ij] = testcount.get(ij, 0) + 1

        for key in final_ious:
            if key in list(set(testcount)):
                if int(testcount[key])==0:
                    final_ious[key] = 0.0
                else:    
                    final_ious[key] /=  testcount[key]
            else:
                continue

        for key in final_acc:
            if key in list(set(testcount)):
                if int(testcount[key])==0:
                    final_acc[key] = 0.0
                else:    
                    final_acc[key] /= testcount[key]
            else:
                continue

        for key in final_hd:
            if key in list(set(testcount)):
                if int(testcount[key])==0:
                    final_hd[key] = 0.0
                else:    
                    final_hd[key] /=  testcount[key]
            else:
                continue

        for key in final_hd95:
            if key in list(set(testcount)):
                if int(testcount[key])==0:
                    final_hd95[key] = 0.0
                else:    
                    final_hd95[key] /=  testcount[key]
            else:
                continue    

        print("Class-wise IOUs: ",final_ious)
        print("Class-wise IOUs average: ",np.mean(np.array(list(final_ious.values())).astype(np.float)))
        print('--------------------------------------')
        print("Class-wise Accs: ",final_acc)
        print("Class-wise Accs average: ",np.mean(np.array(list(final_acc.values())).astype(np.float)))
        print('--------------------------------------')
        print("Class-wise HD: ",final_hd)
        print("Class-wise HD average: ",np.mean(np.array(list(final_hd.values())).astype(np.float)))
        print('--------------------------------------')
        print("Class-wise HD95: ",final_hd95)
        print("Class-wise HD95 average: ",np.mean(np.array(list(final_hd95.values())).astype(np.float)))
        print('--------------------------------------')

        avg_epoch_loss = avg_epoch_loss / len(losses)
        avg_gcn_loss = avg_gcn_loss / len(losses)
        avg_poly_loss = avg_poly_loss / len(losses)
        # avg_class_loss = avg_class_loss / len(losses)

        self.gcn_loss_sum_val = avg_gcn_loss
        avg_iou = avg_iou / len(losses)
        avg_acc = avg_acc / len(losses)

        print("Avg. IOU", avg_iou)
        print("Avg. Accuracy", avg_acc)
        print('Validation Loss : {} '.format(avg_gcn_loss))
        print("Average VAL error is : %f, Average VAL gcn error is : %f, Average VAL poly error is : %f " % (avg_epoch_loss, avg_gcn_loss, avg_poly_loss))
        

        # Validation WandB Storage.
        wandb.log({'gcn-val-avg_iou-epoch':avg_iou,'epoch':self.epoch})
        wandb.log({'gcn-val-avg_poly-epoch':avg_poly_loss,'epoch':self.epoch})
        wandb.log({'gcn-val-avg_gcn_loss-epoch':avg_gcn_loss,'epoch':self.epoch})
        wandb.log({'gcn-val-avg_acc-epoch':avg_acc,'epoch':self.epoch})
        
        # Other Metrics 
        wandb.log({'gcn-val-IOU-epoch':100*np.mean(np.array(list(final_ious.values())).astype(np.float)),'epoch':self.epoch})
        wandb.log({'gcn-val-Accuracy-epoch':np.mean(np.array(list(final_acc.values())).astype(np.float)),'epoch':self.epoch})
        wandb.log({'gcn-val-HD-epoch':np.mean(np.array(list(final_hd.values())).astype(np.float)),'epoch':self.epoch})
        wandb.log({'gcn-val-HD95-epoch':np.mean(np.array(list(final_hd95.values())).astype(np.float)),'epoch':self.epoch})

        self.model.train()


if __name__ == '__main__':
    args = get_args()
    opts = json.load(open(args.exp, 'r'))
    trainer = Trainer(args, opts)
    trainer.loop()


          
