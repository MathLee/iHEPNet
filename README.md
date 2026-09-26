# iHEPNet
[TCSVT2026] [iHEPNet] Exploring Information Entropy-driven Interaction and Hierarchical Edge Perception for Lightweight ORSI Salient Object Detection [PDF](https://github.com/MathLee/mathlee.github.io/blob/main/PDF/2026_TCSVT_Gongyang.pdf)|[Homepage](https://mathlee.github.io/)

# Network Architecture
   <div align=center>
   <img src="https://github.com/MathLee/iHEPNet/blob/main/image/iHEPNet.png">
   </div>
   
   
# Requirements
   python 3.8 + pytorch 1.13.1


# Saliency maps
   We provide saliency maps of [our iHEPNet](https://github.com/MathLee/iHEPNet/tree/main/SaliencyMap), [lightweight methods](https://pan.baidu.com/s/1r_4FeeOJ4h-klBshg89oCA) (code: frem), and [normal-size methods](https://pan.baidu.com/s/13FVHe5kLilEKtBvC-ZwBxw) (code: frem) on the ORSSD, EORSSD, and ORSI-4199 datasets.


      
   ![Image](https://github.com/MathLee/iHEPNet/blob/main/image/table.png)

   
# Training
   We use data_aug.m for data augmentation. 
   
   Modify paths of datasets, then run train_iHEPNet.py.

Note: Our main model is under './model/GeleNet_models.py'. Our code is built on GeleNet. So in this code, GeleNet refers to our iHEPNet.



# Pre-trained model and testing
1. We provide the pre-trained models in './models/'.

2. Modify paths of pre-trained models and datasets.

3. Run test_iHEPNet.py.

   
# Evaluation Tool
   You can use the [evaluation tool (MATLAB version)](https://github.com/MathLee/MatlabEvaluationTools) to evaluate the above saliency maps.


# [ORSI-SOD_Summary](https://github.com/MathLee/ORSI-SOD_Summary)
   
# Citation
        @ARTICLE{Li_2026_iHEPNet,
                  author={Yihua Tu and Wenqi Si and Gongyang Li and Chengjun Han and Yun Sui and Weisi Lin},
                  title={Exploring Information Entropy-driven Interaction and Hierarchical Edge Perception for Lightweight {ORSI} Salient Object Detection}, 
                  journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
                  year={2026},
                  pages={1-13},
                }
                
                
If you encounter any problems with the code, want to report bugs, etc.

Please contact me at lllmiemie@163.com or ligongyang@shu.edu.cn.


