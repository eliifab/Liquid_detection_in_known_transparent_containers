# Run example on single image and single mask (should run out of the box for a the example image and mask)
#...............................Imports..................................................................
import os
import SegmentROIRegion_UsingAttentionRegion_Pretrained_VesselContent.FCN_NetModel as NET_FCN # The net Class
import torch
import shutil
import SegmentROIRegion_UsingAttentionRegion_Pretrained_VesselContent.InferenceSingle as Infer
import SegmentROIRegion_UsingAttentionRegion_Pretrained_VesselContent.ClassesGroups
##################################Input paramaters#########################################################################################


def RunExample(object_name, img_path,mask_path, img_idx):


    OutDir="SegmentROIRegion_UsingAttentionRegion_Pretrained_VesselContent/OutPrediction/"  # Output prediction dir
    Trained_model_path="SegmentROIRegion_UsingAttentionRegion_Pretrained_VesselContent/logs/Defult.torch" # Pretrain model path


    #########################Create output folders#####################################################
    #if os.path.exists(OutDir): shutil.rmtree(OutDir)
    #os.mkdir(OutDir)


    UseGPU=False # use GPU

    OutDirInst=OutDir+"/Instance/"
    OutDirSemantic=OutDir+"/Semantic/"
    OutDirInstDiplay=OutDir+"/InstanceOverlay/"
    OutDirSemanticDisplay=OutDir+"/SemanticOverlay/"
    #**************************Load model and create net******************************************************************************************************************
    ClassToUse=SegmentROIRegion_UsingAttentionRegion_Pretrained_VesselContent.ClassesGroups.VesselContentClasses
    Net=NET_FCN.Net(ClassList=ClassToUse) # Create net and load pretrained
    Net.load_state_dict(torch.load(Trained_model_path,map_location=torch.device('cpu')))
    if UseGPU: Net=Net.cuda().eval()


    #...................................Run prediction and save results....................................................................
    return Infer.InferSingleVessel(Net,img_path,mask_path,OutDirInst,OutDirSemantic,img_idx, object_name, OutDirInstDiplay,OutDirSemanticDisplay, ClassesToUse=ClassToUse,UseGPU=UseGPU)





