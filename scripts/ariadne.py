import cv2
import torch
import numpy as np
import os
import arrow 
from termcolor import cprint

# ariadne
import scripts.core as network
from scripts.utils.dataset import BasicDataset
from scripts.core.tripletnet import TripletNet
from scripts.core.crossnet import CrossNet
from scripts.core.core import Ariadne, AriadnePath
from scripts.core.curvature import CurvatureVonMisesPredictor
from scripts.utils.spline import Spline, SplineMask



class AriadnePlus():

    def __init__(self, main_folder, num_segments, type_model = "STANDARD"):
        
        cprint("Initializing Ariadne+ model...", "white")

        self.num_segments = num_segments

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load Model - DEEPLAB
        if type_model == "STANDARD":
            checkpoint_deeplab = os.path.join(main_folder, 'checkpoints/model_deeplab.pth')
        elif type_model == "SYNTHETIC":
            checkpoint_deeplab = os.path.join(main_folder, 'checkpoints/model_deeplab_synt.pth')
        else:
            print("type model not known!")

        self.model = network.deeplabv3plus_resnet101(num_classes=1, output_stride=16)
        network.convert_to_separable_conv(self.model.classifier)
        checkpoint = torch.load(checkpoint_deeplab, map_location=torch.device('cpu'))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval() 

        # Load Model - TRIPLETNET
        checkpoint_path2 = os.path.join(main_folder, 'checkpoints/model_tripletnet_aug.ckpt')
        self.tripletnet = TripletNet()
        state_dict2 = torch.load(checkpoint_path2, map_location=torch.device('cpu') )['state_dict']
        self.tripletnet.load_state_dict(state_dict2)
        self.tripletnet.eval()
        self.tripletnet.to(self.device)

        # Load Model - CROSSNET
        checkpoint_path = os.path.join(main_folder, 'checkpoints/model_crossnet_aug.ckpt')
        self.crossnet = CrossNet()
        state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))['state_dict']
        self.crossnet.load_state_dict(state_dict)
        self.crossnet.eval()
        self.crossnet.to(self.device)

        cprint("Done!", "green")


    def predictImg(self, net, img, device):

        img = torch.from_numpy(BasicDataset.pre_process(np.array(img)))
        img = img.unsqueeze(0)
        img = img.to(device=device, dtype=torch.float32)

        with torch.no_grad():
            output = net(img)
            probs = torch.sigmoid(output)
            probs = probs.squeeze(0).cpu()
            full_mask = probs.squeeze().cpu().numpy()

        return full_mask


    def runAriadne(self, img, debug=False):

        t0 = arrow.utcnow()
        if debug: print("-> Computing image binary mask ... ")
        ##################################
        # Semantic Segmentation
        ##################################
        mask = self.predictImg(net=self.model, img=img, device=self.device) 
        result = (mask * 255).astype(np.uint8)
        img_mask = result.copy()
        img_mask[img_mask < 127] = 0
        img_mask[img_mask >=127] = 255

        t1 = arrow.utcnow()

        ##################################
        # Initialization GRAPH
        ##################################
        if debug: print("-> Generating image graph ... ")
        img_bgr = img[:,:,::-1] 
        ariadne = Ariadne(img_bgr, img_mask, num_segments=self.num_segments)

        ##################################
        # Paths Discovery
        ##################################
        if debug: print("-> Discovering paths ... ")
        ariadnePathCNN = AriadnePath(   ariadne, 
                                        triplet_net=self.tripletnet, 
                                        cross_net=self.crossnet, 
                                        curvature_pred=CurvatureVonMisesPredictor(ariadne),
                                        device=self.device)
        result_all = ariadnePathCNN.mainPathFinder()

        t2 = arrow.utcnow()

        ##################################
        # Spline Msg
        ##################################
        if debug: print("-> Building spline model ... ")

        # spline output msg
        spline_model = Spline(ariadne)
        result_spline = spline_model.genereteOutputSplinesMsg(result_all)
    

        # colorized output mask
        spline = SplineMask(ariadne)
        masks_labeled_dict = spline.generateSingleLabels(result_all)
    
        indeces_dict = ariadnePathCNN.crossnetPredFast(result_all, masks_labeled_dict)
        mask_final = spline.drawFinalMaskSpline(result_all, indeces_dict)
        
        t3 = arrow.utcnow()
        if debug:
            print("segmentation: \t\t", (t1-t0).total_seconds())
            print("paths discovery: \t", (t2-t1).total_seconds())
            print("output generation: \t", (t3-t2).total_seconds())

            print("RESULT ARIADNE PATHS NODES: ")
            for p in result_all: print(p)


        return {"spline_msg": result_spline, 
                "img_mask": img_mask, 
                "img_final": mask_final, 
                "time": (t3-t0).total_seconds()}
        

