# Generic Libraries
import os
import sys
import cv2
import csv
import json
import copy
import math
import imageio
import argparse
import warnings
import numpy as np
import sklearn.metrics as sm
from datetime import datetime
from collections import defaultdict,OrderedDict
from tqdm import tqdm


# Library Parameters
np.set_printoptions(threshold=sys.maxsize)
warnings.filterwarnings("ignore")
cv2.setNumThreads(0)

# Torch Imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

# File Imports
from utilities import utils
from losses.fm_maps import compute_edts_forPenalizedLoss
import datasets.edge_imageprovider_test as image_provider
from models.combined_model import Model


# Global Parameters
classes = ['Hole(Physical)','Hole(Virtual)','Character Line Segment', 'Character Component','Picture','Decorator','Library Marker', 'Boundary Line', 'Physical Degradation']

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
        pathVis=path+'/vis/'
        pathRes=path+'/res/'
        os.system('mkdir -p %s' % path)
        os.system('mkdir -p %s' % pathRes)
        os.system('mdkir -p %s' % pathVis)
        print('Experiment folder created ')
    except Exception as ex : 
        print('Folder Creation Error , Exiting : {}'.format(ex))
        sys.exit()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str)
    parser.add_argument('--model_weights', type=str, default=None)
    parser.add_argument('--expdir',type=str,default=None)
    parser.add_argument('--vis',type=bool,default=False)
    parser.add_argument('--optfile',type=str,help='Writing the instances to the JSON File',default=None)
    parser.add_argument('--metricfile',type=str,help='CSV File storing relevant metrics',default=None)
    args = parser.parse_args()
    return args


def get_data_loaders(opts, DataProvider):
    print('Building Test Dataloader')
    test_split ='test'
    dataset_test = DataProvider(split=test_split, opts=opts[test_split], mode=test_split)
    test_loader = DataLoader(dataset_test, batch_size=opts[test_split]['batch_size'],
                            shuffle=False, num_workers=opts[test_split]['num_workers'],
                            collate_fn=image_provider.collate_fn)
    return test_loader

class Tester(object):
    
    def __init__(self, args, opts):

        # Make directories     
        create_folder(args)

        self.global_step=0
        self.epoch=0

        self.opts=opts
        self.args=args

        self.model_path=self.args.model_weights
        self.model = Model(self.opts)
        self.model.load_state_dict(torch.load(self.model_path)["gcn_state_dict"])

        # Dataloader
        self.test_loader = get_data_loaders(self.opts['dataset'], image_provider.DataProvider)

    def loop(self):
        self.testing()

    
    def testing(self):
        print('Testing the model ...')
        self.model.eval()

        countFalse=countTrue=0
        globalList=[]


        # Metrics 
        avg_acc = 0.0
        avg_iou = 0.0
        avg_hd=0.0
        avg_hd95=0.0

        final_ious = OrderedDict()
        final_acc = OrderedDict()
        final_hd = OrderedDict()
        final_hd95 = OrderedDict()

        testcount = {}
        testarr=[]

        # IoU alone is a list , why ? 
        for clss in classes: 
            final_ious[clss] = 0.0
            final_acc[clss] = 0.0
            final_hd[clss] = 0.0
            final_hd95[clss] = 0.0
        
        with torch.no_grad():
            for step, data in enumerate(self.test_loader):

                try:
                    id=data['id']
                    img = data['img']
                    img = torch.cat(img)
                    img = img.view(-1, img.shape[0], img.shape[1], 3)
                    img = torch.transpose(img, 1, 3)
                    img = torch.transpose(img, 2, 3)
                    img = img.float()


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
                    # self.poly_loss_fn = nn.BCEWithLogitsLoss(pos_weight = self.p_n2,reduction = 'none')

                    poly_mask = torch.from_numpy(np.asarray(poly_mask)).to(device0)
                    poly_mask = poly_mask.view(1,poly_mask.shape[0],poly_mask.shape[1]).to(device0)

                                    

                    gt_label = torch.tensor(data["gt_label"]).to(device0)


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
                    
                    # if(self.args.vis):
                    #     # ----------- Saving Images -----------
                    #     cv2.fillPoly(palm_leaf_pred, np.int32([pred]), (210,0,0))
                    #     cv2.addWeighted(palm_leaf_pred, 0.2, palm_leaf_pred1, 1 - 0.3, 0, palm_leaf_pred1)
                    #     cv2.polylines(palm_leaf_pred1, np.int32([dp7]), True, [255,255,255], thickness=1)
                    #     cv2.polylines(palm_leaf_pred1, np.int32([pred]), True, (210,0,0), thickness=1)
                    #     for point in pred:
                    #         cv2.circle(palm_leaf_pred1, (int(point[0]), int(point[1])), 1, (0, 0, 210), -1)
                    #     imageio.imwrite(self.args.expdir+'/vis/'+str(data['id'])+'_'+str(step) + ".jpg", palm_leaf_pred1, quality=100)


                    avg_iou += iou1
                    avg_acc += accuracy1
                    avg_hd += hd1
                    avg_hd95+=hd951

                    class_lab = data['label'][0]
            
                    final_acc[class_lab] += accuracy1
                    final_ious[class_lab]+=iou1
                    final_hd[class_lab] += hd1
                    final_hd95[class_lab] += hd951

                    testarr.append(class_lab)

                    # Forming the dictionary and all its components.
                    finalDict={}
                    outputs={}
                    metrics={'iou':iou1,'hd':hd1,'hd95':hd951}
            
                    imagePath=str(data["image_url"][0])
                    # Replace address for easy tool inegration . 
                    imagePath=imagePath.replace('data/ASR_Images/','example/data/BoxSupervised/ASR_Images/')
                    imagePath=imagePath.replace('data/Bhoomi_data/','example/data/BoxSupervised/Bhoomi_data/')
                    imagePath=imagePath.replace('data/jain-mscripts/','example/data/BoxSupervised/jain-mscripts/')
                    imagePath=imagePath.replace('data/penn-in-hand/','example/data/BoxSupervised/penn-in-hand/')
                    imagePath=imagePath.replace('data/penn_in_hand/','example/data/BoxSupervised/penn_in_hand/')

                    # TBA 
                    finalDict['id']=str(id[0])
                    finalDict['imagePath']=imagePath
                    finalDict['metrics']=metrics
                    finalDict['regionLabel']=data["cm_label"][0]
                    finalDict['bbox']=bbox[0].tolist()
                    outputs['poly']=(dp101[0].cpu().numpy()).tolist()
                    outputs['encoder_output']=encoder_output.tolist()
                    outputs['gcn_output']= pred.tolist()
                    finalDict['outputs']=outputs

                    # Append it to the list 
                    countTrue+=1
                    globalList.append(finalDict)

                    print('Step --- {} , Epoch --- {} , Metrics ---- {}'.format(step,self.epoch,metrics))
                    
                except Exception as exp : 
                    print('Testing for the instance failed - {} , {} '.format(exp,str(id[0])))
                    countFalse+=1
                    continue


        with open(args.expdir+'/res/'+args.optfile,'w', encoding='utf-8') as outfile:
            json.dump(globalList, outfile,ensure_ascii=False)

        # Metrics Part
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

        
        print(' Dataloader Stats --- CountFalse  : {} ----- CountTrue : {}'.format(countFalse,countTrue))


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

        # Create a dictionary here , and write the dictionary onto the output file 
        fields=['model','iou', 'hd', 'hd95','acc'] 
        metrics_dict={}
        for item in classes:
            metrics={'iou':final_ious[item],'hd':final_hd[item],'hd95':final_hd95[item],'acc':final_acc[item]}
            metrics_dict[item]=metrics
        
         # Writing classwise onto a csv file .
        try:
            with open(args.expdir+'/res/'+'classwise_metrics_'+args.metricfile,'w') as csvfile: 
                writer = csv.DictWriter(csvfile, fieldnames=fields)
                writer.writeheader()
                for key,values in metrics_dict.items():
                    csvfile.write("%s,%s,%s,%s,%s\n"%(key,metrics_dict[key]['iou'],metrics_dict[key]['hd'],metrics_dict[key]['hd95'],metrics_dict[key]['acc']))
                
        except IOError:
            print("I/O error")
        
        avg_metrics_dict={'metrics':'values','avg_iou':avg_iou /len(globalList),'avg_hd':avg_hd/len(globalList),'avg_hd95':avg_hd95/len(globalList),'avg_acc':avg_acc/len(globalList)}
      
        # Writing avg metrics onto a csv file .
        with open(args.expdir+'/res/'+'avg_metrics_'+args.metricfile,'w') as f: 
            for key in avg_metrics_dict.keys():
                f.write("%s,%s\n"%(key,avg_metrics_dict[key]))

        print('GCN Testing Done ...')

            

if __name__ == '__main__':
    args = get_args()
    opts = json.load(open(args.exp, 'r'))
    tester = Tester(args, opts)
    tester.loop()


