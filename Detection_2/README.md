In main.py: modify paths in CONFIGURATION SECTION  
3D-DAT: https://github.com/v4r-tuwien/3D-DAT  
Dataset: https://researchdata.tuwien.at/records/3b85h-y6t96  

## Folder structure:  
Detection_2/  
├── output_img/ *--- Folder for output images*  
├── full_model.py  
├── img_edge_prediction.py  
├── main.py  
├── optimization_z_torch.py  
├── sam_vit_h_4b8939.pth  *--- [download ViT-H SAM model](https://github.com/facebookresearch/segment-anything?tab=readme-ov-file#model-checkpoints)*  
├── SegmentROIRegion_UsingAttentionRegion_Pretrained_VesselContent/  *--- [download pretrained](https://drive.google.com/file/d/1uqvTqEFqMDXCuoEDlfm3yjNnmpw2bX-b/view)*  
**Modified files, update after downloading ROI-NN code:** 
>>├── FCN_NetModel.py  
>>├── InferenceSingle.py  
>>├── RunExample.py  
>>└── ... remaining files  

