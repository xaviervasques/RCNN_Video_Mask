# RCNN_Video_Mask
Mask R-CNN for object detection in videos

# Installation

    git clone https://github.com/xaviervasques/RCNN_Video_Mask.git
  
    git clone https://github.com/xaviervasques/Mask_RCNN.git

Go to Mask_RCNN folder and install dependencies: 

    pip3 install -r requirements.txt

Run setup from the repository root directory

    python3 setup.py install
  
If you want a pre-trained model, download the pre-trained COCO weights (mask_rcnn_coco.h5): https://github.com/matterport/Mask_RCNN/releases and copy it into the main folder (RCNN_Video_Mask)

You can also uncomment the following lines in Capture_Mask_RCNN.py: 

    if not os.path.exists(COCO_MODEL_PATH):
  
      utils.download_trained_weights(COCO_MODEL_PATH)

If you want to train your own dataset, go to https://github.com/xaviervasques/Mask_RCNN.git

In the main folder, create a folder with the name "videos" in which you will put your videos to capture and mask. Put a video in the folder. 

To run the code:

    python3 Capture_Mask_RCNN.py 
  
    python3 Make_video.py
  
 Find the output in the "videos" folder
 
 Here is an exemple, driving in the Montpellier City (France): 
 
 ![Example](https://github.com/xaviervasques/RCNN_Video_Mask/example.jpg)
 
 https://www.youtube.com/watch?v=Rw9nOIhVf9A

    
      
 Sources
 
 https://github.com/xaviervasques/colab-mask-rcnn.git
 
 https://github.com/xaviervasques/Mask_RCNN.git
 





