# Generic Libraries
import os 
import cv2
import sys
import math
import skfmm
import numpy as np
from collections import OrderedDict
from scipy.interpolate import splprep, splev

# Torch Imports
import torch
import torch.nn as nn

# File Imports 

# MCNN final model path must be changed . 
from models.MCNN.mcnn import Model as enc_model
from models.AGCN.poly_gnn import PolyGNN
from utilities.contourization import testing_hull



# GPU Settings
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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



def get_hull(hull, bbox, w1, h1, feat_w1, feat_h1):
    original_hull = []
    binary_hull = []
    feature_hull = []

    w = float(bbox[2])
    h = float(bbox[3])

    h1 = float(h1)
    w1 = float(w1)

    for i in hull:
        original_hull.append([int((i[1])), int((i[0]))])
        binary_hull.append([i[1]/h, i[0]/w])
        feature_hull.append([math.floor(i[1] * feat_h1 / h), math.floor(i[0] * feat_w1 / w)])

    return original_hull, binary_hull, feature_hull

class Model(nn.Module):

    def __init__(self, opts):
        super(Model, self).__init__()
        self.opts = opts

        # MCNN Initialisation 
        self.edgemodel=enc_model(None,None,3)
        # encoder_state_dict=torch.load(self.opts['encoder_weight_path'])


        if(os.path.isfile(self.opts['encoder_weight_path'])):
            self.modelpath=self.opts['encoder_weight_path']
            self.edgemodel.load_state_dict(torch.load(self.modelpath)["gcn_state_dict"])
            print('Model weights successfully loaded ... ')
        else: 
            print('Error in Encoder ModelWgt Loading...Exiting')
            sys.exit()


        # encoder_new_state_dict = OrderedDict()

        # ----------- Freezing full Encoder -----------
        if self.opts["enc_freeze"]: 
            ct = 0
            for child in self.edgemodel.children():
                ct += 1
                if ct >0:
                    for param in child.parameters():
                        param.requires_grad = False
        
        # GCN inputs a 120 dimensional feature vector.
        # Changing GCN 's device to device 0

        state_dim = 120

        # self.gcn_model=PolyGNN(state_dim=state_dim,
        #                                   n_adj=self.opts['n_adj'],
        #                                   cnn_feature_grids=self.opts['cnn_feature_grids'],
        #                                   coarse_to_fine_steps=self.opts['coarse_to_fine_steps'],
        #                                   ).to(device)

        self.gcn_model=PolyGNN(state_dim=state_dim,
                                          n_adj=self.opts['n_adj'],
                                          cnn_feature_grids=self.opts['cnn_feature_grids'],
                                          coarse_to_fine_steps=self.opts['coarse_to_fine_steps'],
                                          ).to(device0)


    def forward(self,img,bbox,mode):
        class_prob=0
        bbox_cp = bbox.tolist()
        cp = 0

        # Getting Encoder's Output - Device 1
        # tg2, poly_logits= self.edgemodel(img.to(device))
        tg2, poly_logits= self.edgemodel(img.to(device1))

        # Send them to device 0 for processing work 
        # tg2 = tg2.to(device)
        # poly_logits = poly_logits.to(device)
        tg2 = tg2.to(device0)
        poly_logits = poly_logits.to(device0)

        feat_h1, feat_w1 = tg2[:,0,:,:].shape[1],tg2[:,0,:,:].shape[2]
        h1,w1 = poly_logits[:,0,:,:].shape[1],poly_logits[:,0,:,:].shape[2]

        original_hull = []
        binary_hull = []
        feature_hull = []
        listpp = []
        listpp11 = []
        new_list = []

        for i in range(poly_logits.shape[0]):            
            hull1 = testing_hull(poly_logits,class_prob,bbox)
            new_list1 = np.asarray(hull1)
            new_list.append(new_list1)
        new_list = np.asarray(new_list)

        for i in range(poly_logits.shape[0]):
            original_hull_i, binary_hull_i, feature_hull_i = get_hull(new_list[i], bbox[i], w1, h1, feat_w1, feat_h1)
            original_hull.append(original_hull_i)
            binary_hull.append(binary_hull_i)
            feature_hull.append(feature_hull_i)

        feature_hull = torch.from_numpy(np.asarray(feature_hull))
        original_hull = torch.from_numpy(np.asarray(original_hull))
        binary_hull = torch.from_numpy(np.asarray(binary_hull))
        bbox = torch.from_numpy(np.asarray(bbox_cp))

        # This is obviously going to device 0 ~ device as per the existign codebase .
        output_dict = self.gcn_model(tg2, feature_hull, original_hull,binary_hull, bbox, mode)

        return output_dict, poly_logits














        
