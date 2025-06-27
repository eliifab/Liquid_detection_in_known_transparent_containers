# Apply inference to single mask in single image.
#...............................Imports..................................................................
import os
import numpy as np
import SegmentROIRegion_UsingAttentionRegion_Pretrained_VesselContent.Visuallization as vis
import cv2
import shutil
import json
import torch
##################################Input paramaters#########################################################################################
# ####################################################################################################
MinImSize=200 # min image height/width smaller images will be resize
MaxImSize=1000#  max image height/width smaller images will be resize

MinClassThresh=0.3 # Threshold for assigning class to image
MinInsPixels=250 # Min pixels in prediction smaller instances will not be registered

#********************************************************************************************************************************************


def InferSingleVessel(Net,ImagePath,VesMaskPath,OutDirInst,OutDirSemantic, img_idx, object_name, OutDirInstDiplay="",OutDirSemanticDisplay="",ClassesToUse="",Display=False,UseGPU=False): # Run Predict on a single vessel in an image
    if not os.path.exists(OutDirInst): os.mkdir(OutDirInst)
    if not os.path.exists(OutDirSemantic): os.mkdir(OutDirSemantic)
    if not os.path.exists(OutDirInstDiplay) and OutDirInstDiplay!="": os.mkdir(OutDirInstDiplay)
    if not os.path.exists(OutDirSemanticDisplay) and OutDirSemanticDisplay!="": os.mkdir(OutDirSemanticDisplay)

    saved_liquid_instance = 0
    saved_liquid_semantic = 0

    Image = cv2.imread(ImagePath)  # Load Image
    Img,  w0, h0=vis.ResizeToLimit(Image,MinImSize,MaxImSize,interpolation=cv2.INTER_NEAREST) # Resize if image to large or small
   # Img=Img[:,:,::-1]
    Img = np.expand_dims(Img, axis=0)
    VesMask = (cv2.imread(VesMaskPath, 0) > 0).astype(np.uint8) # Read vessel mask
    VesMask, w0, h0 = vis.ResizeToLimit(VesMask, MinImSize, MaxImSize, interpolation=cv2.INTER_NEAREST) # Resize if to big or small
    VsMsk=np.expand_dims(VesMask,axis=0)
    with torch.no_grad(): # Run Prediction
        ProbInst, LbInst, ProbSemantic, LbSemantic = Net.forward(Images=Img,ROI=VsMsk,TrainMode=False,PredictSemantic=True,PredictInstance = True,UseGPU=UseGPU) # Run net inference and get prediction
    Net.zero_grad()

    #cv2.imshow("img",Image)
    #cv2.waitKey(0)

    #cv2.imshow("mask",VesMask)
    #cv2.waitKey(0)
    ################################Set Instance Class###########################################################################################################
    ################################Filter and Set Instance Class by comparing instances to semantic maps###########################################################################################################
    NumInst=0
    InstMasks={}
    InstClasses = {}
    InstProbClass = {}
    VesMask = cv2.resize(VesMask, (int(w0), int(h0)), interpolation=cv2.INTER_NEAREST)
    for InsMsk in LbInst:  # Find overlap between instance and semantic maps to deterine class
        InsMsk = InsMsk[0]#.data.cpu().numpy()
        SumAll=InsMsk.sum()
        if SumAll>MinInsPixels:
            NumInst+=1
            InstClasses[NumInst] = []
            InstProbClass[NumInst] = {}
            for nm in LbSemantic:
                if not nm in ClassesToUse: continue
                SemMask = LbSemantic[nm][0]#.data.cpu().numpy()
                Fract=float(((InsMsk*SemMask).sum()/SumAll).data.cpu().numpy()) # find overlap between instace and semantic maps
                if Fract>MinClassThresh: # If overlap exceed thresh assign class to instance
                    InstClasses[NumInst].append(nm)
                    InstProbClass[NumInst][nm]=Fract


            InstMasks[NumInst]=InsMsk.data.cpu().numpy().astype(np.uint8)

        ################################Write and Display###########################################################################################################
        # OutSemDir = OutDir+"/Semantic/"
        # OutInstanceDir = OutDir + "/Instance/"
        # OutSemDirOverlay = OutDir + "/SemanticOverlay/"
        # OutInstanceDirOverlay = OutDir + "/InstanceOverlay/"
        # if os.path.exists(OutDir): shutil.rmtree(OutDir)
        # os.mkdir(OutDir)
        # os.mkdir(OutSemDir)
        # os.mkdir(OutInstanceDir)
        # os.mkdir(OutSemDirOverlay)
        # os.mkdir(OutInstanceDirOverlay)
    #---------------Save instance categories-----------------------------------------------
        with open(OutDirInst+'/InstanceClassList.json', 'w') as fp: # List of classes for instance
            json.dump(InstClasses, fp)
        with open(OutDirInst+'/InstanceClassProbability.json', 'w') as fp: # Class probability  for instance
            json.dump(InstProbClass, fp)
        # cv2.imwrite(OutDir+"/Image.jpg",Image)
       # cv2.imwrite(OutDirInst + "/Mask.png",VesMask)
    # ------------------------DisplayInstances----------------------------------------

        I = Image.copy()
        I1 = I.copy()
        I1[:, :, 0][VesMask > 0] = 0
        I1[:, :, 1][VesMask > 0] = 0
        I1[:, :, 2][VesMask > 0] = 255
        if Display == True:
            vis.show(np.concatenate([I, I1], axis=1), "Vessel  ")


        for ins in InstMasks: # Save instance maps
            Mask=InstMasks[ins]
            Mask = cv2.resize(Mask, (int(w0), int(h0)),interpolation=cv2.INTER_NEAREST)
            if Mask.sum()==0: continue
            if nm == 'Vessel': continue

            saved_liquid_instance = 1
#-------------Save instance and display-------------------------------------------------------------
            I2 = I1.copy()
            I2[:, :, 0][Mask > 0] = 0
            I2[:, :, 1][Mask > 0] = 255

            cv2.imwrite(OutDirInst+ object_name + "_" + str(ins)+ "_" + str(img_idx)+".png",Mask.astype(np.uint8))
            #Overlay=np.concatenate([I, I1, I2, vis.GreyScaleToRGB(Mask * 255)],axis=1).astype(np.uint8)
            Mask_vis = vis.GreyScaleToRGB(Mask * 255)
            if OutDirInstDiplay != "": # Instance maps overlay on image
                cv2.imwrite(OutDirInstDiplay + object_name + "_" + str(ins) + "_" + str(img_idx) + ".png", Mask_vis)
                #cv2.imwrite(OutDirInstDiplay + "/" + str(ins) + "B.png", I2)
            if Display: vis.show(Mask_vis,"Instace:" + str(InstClasses[ins])+" "+str(InstProbClass[ins]))

            return Mask_vis


        # ---------------Display  and save Semantic maps-----------------------------------------------------------------------------------------------
        I = Image.copy()

        for nm in LbSemantic:
            Mask = LbSemantic[nm][0].data.cpu().numpy()
            Mask = cv2.resize(Mask, (int(w0), int(h0)), interpolation=cv2.INTER_NEAREST)
            if Mask.sum() == 0: continue
            if nm == 'Vessel': continue
            if nm == 'Liquid':
                saved_liquid_semantic = 1
                I2 = I1.copy()
                I2[:, :, 0][Mask > 0] = 0
                I2[:, :, 1][Mask > 0] = 255

                #cv2.imwrite(OutDirSemantic + object_name + "_" + nm + str(img_idx) + ".png", Mask.astype(np.uint8))
                #Overlay = np.concatenate([I, I1, I2, vis.GreyScaleToRGB(Mask * 255)], axis=1)
                Mask_vis = vis.GreyScaleToRGB(Mask * 255)
                #if OutDirInstDiplay!="": # Semantic Map overlay on image
                #    cv2.imwrite(OutDirSemanticDisplay + object_name + "_" + nm + str(img_idx) + ".png", Mask_vis)
                #    #cv2.imwrite(OutDirSemanticDisplay + "/" + nm + "B.png", I2)
                #if Display: vis.show(Mask_vis,"Semantic:" + nm)



    '''
    if not saved_liquid_instance:
        cv2.imwrite(OutDirInst + object_name + "_" + str(1) + "_" + str(img_idx) + ".png", np.zeros_like(Image, dtype=np.uint8))
        cv2.imwrite(OutDirInstDiplay + object_name + "_" + str(1) + "_" + str(img_idx) + ".png", np.zeros_like(Image, dtype=np.uint8))
    
    if not saved_liquid_semantic:
        cv2.imwrite(OutDirSemantic + object_name + "_" + 'Liquid' + str(img_idx) + ".png", np.zeros_like(Image, dtype=np.uint8))
        cv2.imwrite(OutDirSemanticDisplay + object_name + "_" + 'Liquid' + str(img_idx) + ".png",
                    np.zeros_like(Image, dtype=np.uint8))
    '''

    return np.zeros_like(Image, dtype=np.uint8)

    #print('saved' + str(img_idx))