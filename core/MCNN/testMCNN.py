# Libraries Involved .
import torch
import csv
import json
import os
import argparse
import copy
import imageio
import skfmm
from  scipy import ndimage
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from datetime import datetime
from collections import defaultdict
from skimage.io import imsave
from tqdm import tqdm
import cv2
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)
import math
import warnings
warnings.filterwarnings('ignore')


# File Imports 
from utilities import utils
from utilities import contourization
from models.MCNN.newEncoder  import Model
import datasets.edge_imageprovider_test as image_provider
from losses.fm_maps import compute_edts_forPenalizedLoss


# Global Variables 
classes = ['Hole(Physical)','Hole(Virtual)','Character Line Segment', 'Character Component','Picture','Decorator','Library Marker', 'Boundary Line', 'Physical Degradation']

# GPU Settings .
if torch.cuda.is_available():
    print("CUDA Acquired !")
    print('__CUDNN VERSION__:', torch.backends.cudnn.version())
    print('__Number CUDA Devices__:', torch.cuda.device_count())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
else:
    print("No GPU Allocated ... Exiting !")
    sys.exit()


def create_folder(args):
    path = args.expdir
    pathResults=path+str('/results')
    os.system('mkdir -p %s' % path)
    os.system('mkdir -p %s' % pathResults)

    if(args.vis):
        pathVis=path+str('/visualization')
        pathtest=pathVis+str('/test_enc_pred')
        pathModelLogits=pathtest+str('/model_polylogit')
        pathModelEncoderOpts=pathtest+str('/encoder_opts')
        pathSmoothEncoderOpts=pathtest+str('/smoothened_encoder_opts')

        os.system('mkdir -p %s' % pathtest)
        os.system('mkdir -p %s' % pathModelLogits)
        os.system('mkdir -p %s' % pathModelEncoderOpts)
        os.system('mkdir -p %s' % pathSmoothEncoderOpts)


    print('Experiment folder created !')
    
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str,help='Respective Experiment JSON')
    parser.add_argument('--expdir', type=str,help='Respective Experiment Folder')
    parser.add_argument('--modelfile', type=str,help='MCNN Weight File Path')
    parser.add_argument('--vis',type=bool,default=False,help='Flag for storing all the images')
    parser.add_argument('--optfile',type=str,help='Writing the instances to the JSON File')
    parser.add_argument('--metricfile',type=str,help='CSV File storing relevant metrics',default=None)
    parser.add_argument('--resume', type=str, default=None)
    args = parser.parse_args()
    return args

# Only Testing , so we can avoid train related parameters .
def get_data_loaders(opts, DataProvider):
    print('Building TestSet Dataloader..')
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

        # Model Declaration - MCNN only 
        self.model=Model(3,3,None)
        if(os.path.isfile(args.modelfile)):
            self.modelpath=args.modelfile
            self.model.load_state_dict(torch.load(self.modelpath)["gcn_state_dict"])
            print('Model weights successfully loaded ... ')
        else: 
            print('Error in ModelWgtLoading...Exiting')
            sys.exit()
        
        # Load the dataloader 
        self.test_loader=get_data_loaders(self.opts['dataset'], image_provider.DataProvider)

        # Setting Model to eval form 
        self.model.eval()
    
    def test(self):
        print('Intiating Testing process ....')
        self.model.eval()
        
        # Empty variable initialisation for different variables ..
        avg_acc = 0.0
        avg_iou = 0.0
        avg_hd95 =0.0
        avg_epoch_loss = 0.0
        avg_poly_loss = 0.0

        final_ious = {} 
        final_acc = {}
        final_hd = {} 
        final_hd95 = {}
        testcount = {}

        losses = []
        gcn_losses = []
        poly_losses = []
        class_losses = []
        testarr=[]
        globalList=[]

        for clss in classes: 
            final_ious[clss] = 0.0
            final_acc[clss] = 0.0
            final_hd[clss] = 0.0
            final_hd95[clss] = 0.0

        countTrue=countFalse=0
        
        with torch.no_grad():
            for step, data in enumerate(tqdm(self.test_loader)):
                
                try :
                    id=data['id']
                    img = data['img']
                    img = torch.cat(img)
                    img = img.view(-1, img.shape[0], img.shape[1], 3)
                    img = torch.transpose(img, 1, 3)
                    img = torch.transpose(img, 2, 3)
                    img = img.float()
                except Exception as e : 
                    print('LoadingImageIssue : {}'.format(e))
                    countFalse+=1
                    break

                try: 
                    bbox = torch.from_numpy(np.asarray(data['bbox'])).to(device)
                except Exception as ex : 
                    print('Bbox Loading Error : {}'.format(ex))
                    countFalse+=1
                    continue
                   
                # Loading the image parameters..
                w1 = torch.tensor(data["w"]).view(-1,1).to(device).float()
                h1 = torch.tensor(data["h"]).view(-1,1).to(device).float()


                # Ground Truth Polygon 
                dp_poly = data['actual_gt_poly']
                dp101 = data['actual_gt_poly11'].detach().clone()
                dp = torch.tensor(data['actual_gt_poly'], dtype = torch.float32).to(device)

                # Model Input 
                try : 
                    tg2, poly_logits,_ = self.model(img.to(device))
                    poly_logits_copy=copy.deepcopy(poly_logits)
                except Exception as e : 
                    print('ModelInputIssue : {}'.format(e))
                    print('Exiting ... ')
                    break

                # Mask Extraction for metrics .
                try:
                    poly_logits88 = torch.sigmoid(poly_logits[0,0,:,:]).cpu().numpy()
                    yy = poly_logits88 > 0.5
                    yy = yy+0
                    poly_logits88 = yy.astype(np.float32)
                    poly_logits89 = (poly_logits88*255).astype(np.uint8)
                    
                    if(self.args.vis):
                        # Block to save the mask logits generated from the image . 
                        # Essentially save this poly_logits89 
                        imageio.imwrite(args.expdir+"/visualization/test_enc_pred/model_polylogit/" + str(step) + ".jpg", poly_logits89, quality=100)
                        
                except Exception as e1: 
                    print('Mask Metrics Computation Block Issue : {}'.format(e1))
                    countFalse+=1
                    continue
                    # sys.exit()

                # Morphological Operations on the mask .
                kernel2 = np.ones((3,3),np.uint8)
                # This is the mask image that will be morphed with the kernel .
                arrs2 = cv2.morphologyEx(poly_logits89, cv2.MORPH_CLOSE, kernel2)

                # H_contour block replace by these 2 lines 
                try:
                    h_smooth=contourization.testing_hull(poly_logits_copy,0, bbox)
                    h_smooth = np.asarray(h_smooth,dtype=np.int32)
                    h_contour=h_smooth
                

                # # Largest Contour Finding - the contour without smoothening part .
                except Exception as ex:
                    max_area=0
                    largest_contour_index=-1
                    contours, hierarchy = cv2.findContours(arrs2,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                    cont_list = []
                    # Growing through all the detected contours in the mask .. 
                    for i in range(len(contours)):
                        cont=contours[i]
                        area=cv2.contourArea(cont)
                        if(area>max_area):
                            max_area=area
                            largest_contour_index=i
                    
                    # After you get the largest contour ..
                    try : 
                        h_contour=contours[largest_contour_index]
                    except Exception as exp : 
                        print('Unresolvable : {}'.format(exp))
                        continue
                    # The Try-Except block for this assigns the largest contour as the corner points of the image itself. 
                    # So nope , we aren't using that shit .

                    # So post processing on this on the h_coutour 
                    h_contour = np.squeeze(h_contour)
                    h_contour = h_contour[::-1]
                    h_contour = np.asarray(h_contour,dtype=np.int32)

                    countFalse+=1
                    print('Contour Block Issue ! : {}'.format(ex))

                # Saving images with these contours drawn.
                if(self.args.vis):
                    try: 
                        palm_leaf_pred=copy.deepcopy(data['img_orig'][0])
                        h = palm_leaf_pred.shape[0]
                        w = palm_leaf_pred.shape[1]
                        # print('HContourShape : {}'.format(h_contour.shape))
                        outputimage=cv2.polylines(np.float32(palm_leaf_pred),[h_contour],True,(255, 0, 0), thickness = 1)
                        # print('Shape of HCOUNTOUR : {}'.format(np.asarray([h_contour],dtype=np.int32).shape))
                        imageio.imwrite(args.expdir+"/visualization/test_enc_pred/encoder_opts/" + str(step) + ".jpg", outputimage, quality=100)                  
                    except Exception as ex:
                        print(' Issue with the polylines visualisation : {}'.format(ex))

                # Prediction Mask processing 
                try :
                    # An empty mask with poly_logits shape 
                    pred_mask58 = np.zeros((poly_logits[:,0,:,:].shape[1], poly_logits[:,0,:,:].shape[2]),np.float32)
                    # Fill the mask with h_contour's polygon , and fill it . 
                    cv2.fillPoly(pred_mask58, np.int32([h_contour]),[1])

                except Exception as ex: 
                    print('Prediction Mask is not getting filled ! :{}'.format(ex))
                    countFalse+=1
                    # sys.exit()
            
                # Extraction of Ground Truth Masks for loss & metrics computations.
                try:
                    # Poly Mask 
                    poly_mask = np.zeros((poly_logits[:,0,:,:].shape[1], poly_logits[:,0,:,:].shape[2]),np.float32)
                    poly_mask  = utils.get_poly_mask(dp.cpu().numpy()[0],poly_mask)
                    # Edge Mask 
                    edge_mask77 = np.zeros((poly_logits[:,0,:,:].shape[1], poly_logits[:,0,:,:].shape[2]),np.float32)
                    edge_mask77  = utils.get_edge_mask(dp.cpu().numpy()[0],edge_mask77)

                    if(self.args.vis):
                        imageio.imwrite(args.expdir+"/visualization/test_enc_pred/poly_mask/" + str(step) + ".jpg", poly_mask, quality=100)
                        imageio.imwrite(args.expdir+"/visualization/test_enc_pred/edge_mask/" + str(step) + ".jpg",  edge_mask77, quality=100)

                except Exception as ex : 
                    print('Edge Mask or Poly Mask failed ! : {}'.format(ex))
                    countFalse+=1
                    continue
                    # sys.exit()

                # What's this exacty 
                n_poly = (np.sum(poly_mask)).astype(np.float32)
                back_mask = 1.0-poly_mask
                n_back = (np.sum(back_mask)).astype(np.float32)

                try: 
                    DT_mask = compute_edts_forPenalizedLoss(edge_mask77)
                    DT_mask = torch.from_numpy(np.asarray(DT_mask)).to(device)
                    DT_mask = DT_mask.float()
                except Exception as ex : 
                    print('DT Mask Issue : {}'.format(ex))
                    countFalse+=1
                    continue
                    # sys.exit()
                
                w1,h1 = poly_logits[:,0,:,:].shape[1],poly_logits[:,0,:,:].shape[2]
                # Creating a weight matrix - same dimenstion of image , but weighted .
                self.p_n2 = torch.ones((w1,h1), dtype= torch.float32).to(device)
                self.p_n2 = self.p_n2*(n_back/n_poly)

                
                # Polygon Loss Computation
                self.poly_loss_fn = nn.BCEWithLogitsLoss(pos_weight = self.p_n2, reduction = 'none')

                poly_mask = torch.from_numpy(np.asarray(poly_mask)).to(device)
                # Change shape - ( 1, w , h)
                poly_mask = poly_mask.view(1,poly_mask.shape[0],poly_mask.shape[1]).to(device)

                try : 
                    # Loss computation.
                    poly_loss1 = self.poly_loss_fn(poly_logits[:,0,:,:], poly_mask.to(device))
                    poly_loss1 = poly_loss1*DT_mask
                    poly_loss1 = torch.mean(poly_loss1)
                    print('Poly loss is : {} , Step :  {}'.format(poly_loss1,step))
                except Exception as ex : 
                    print('Poly Loss Computation Failed ! : {}'.format(ex))
                    countFalse+=1
                    continue
                

                # Mask Computation - ioU and HD 
                palm_leaf_pred = copy.deepcopy(data['img_orig'][0])
                
                pred_mask = (pred_mask58*255).astype(np.uint8)
                gt_mask = ((poly_mask*255)[0].cpu().numpy()).astype(np.uint8)


                # IoU & Hausdorff Metric Computation 
                try : 
                    iou, accuracy = utils.compute_iou_and_accuracy(pred_mask, gt_mask)
                    hd = utils.hd(pred_mask, gt_mask)
                    hd95= utils.hd95(pred_mask, gt_mask)
                except Exception as ex : 
                    print(' Metrics Computation Failed : {}'.format(ex))
                    countFalse+=1
                    continue
                

                # Loss Summation 
                avg_iou += iou
                avg_acc += hd
                avg_hd95+=hd95  

                # Metrics Storage 
                class_lab = data['label'][0]

                final_ious[class_lab] += iou
                final_acc[class_lab] += accuracy

                final_hd[class_lab] += hd
                final_hd95[class_lab] += hd95

                # Loss
                poly_loss=poly_loss1
                loss_sum=1*poly_loss1
                poly_loss_sum = poly_loss1
                loss=loss_sum

                # Loss List 
                losses.append(loss)
                poly_losses.append(poly_loss_sum)
                testarr.append(class_lab)


                # Output JSON File Write .

                finalDict={}
                outputs={}
                metrics={'iou':iou,'hd':hd,'hd95':hd95}
        
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
                outputs['encoder_output']=h_contour.tolist()
                # outputs['gcn_output']=h_smooth.tolist()
                finalDict['outputs']=outputs

                # Append it to the list 
                countTrue+=1
                print('Step : {} ------ CountFalse  : {} ----- CountTrue : {}'.format(step,countFalse,countTrue))
                globalList.append(finalDict)

      
        # JSON file for output stuff
        with open(args.expdir+'/results/'+args.optfile,'w', encoding='utf-8') as f:
            json.dump(globalList,f,ensure_ascii=False)
        
        for i in range(len(losses)):
            avg_epoch_loss += losses[i]
            avg_poly_loss += poly_losses[i]

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
                    final_acc[key] /=  testcount[key]
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

        # Create a dictionary here , and write the dictionary onto the output file 
        fields=['iou', 'hd', 'hd95','acc'] 
        metrics_dict={}
        for item in classes:
            metrics={'iou':final_ious[item],'hd':final_hd[item],'hd95':final_hd95[item],'acc':final_acc[item]}
            metrics_dict[item]=metrics

        # Writing classwise onto a csv file .
        try:
            with open(args.expdir+'/results/'+'classwise_metrics_'+args.metricfile,'w') as csvfile: 
                writer = csv.DictWriter(csvfile, fieldnames=fields)
                writer.writeheader()
                for key,values in metrics_dict.items():
                    csvfile.write("%s,%s,%s,%s,%s\n"%(key,metrics_dict[key]['iou'],metrics_dict[key]['hd'],metrics_dict[key]['hd95'],metrics_dict[key]['acc']))
                
        except IOError:
            print("I/O error")
        
        avg_metrics_dict={'avg_epoch_loss': (avg_epoch_loss / len(losses)).cpu().detach().numpy(),'avg_poly_loss':(avg_poly_loss/len(losses)).cpu().detach().numpy(),'avg_iou':avg_iou / len(losses),'avg_hd':avg_acc / len(losses),'avg_hd95':avg_hd95/len(losses)}
      
        # Writing avg metrics onto a csv file .
        with open(args.expdir+'/results/'+'avg_metrics_'+args.metricfile,'w') as f: 
            for key in avg_metrics_dict.keys():
                f.write("%s,%s\n"%(key,avg_metrics_dict[key]))

        avg_epoch_loss = avg_epoch_loss / len(losses)
        avg_poly_loss = avg_poly_loss / len(losses)
        avg_iou = avg_iou / len(losses)
        avg_hd = avg_acc / len(losses)
        avg_hd95=avg_hd95/len(losses)
         
        print(" Avg IOU : ", avg_iou)
        print(" Avg HD  : ", avg_hd)
        print(" Avg HD 95  : ", avg_hd)

        print("Average VAL error is : %f, Average VAL poly error is : %f " % (avg_epoch_loss,avg_poly_loss))
        print(' Final : CountFalse  : {} ----- CountTrue : {}'.format(countFalse,countTrue))
        
if __name__ == '__main__':
    args = get_args()
    opts = json.load(open(args.exp, 'r'))
    tester = Tester(args, opts)
    tester.test()
