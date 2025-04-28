# AI within deepfake detection
This project uses a CNN + LSTM model to try and predict the differece between real and manipulated images. The model is run through command line using "streamlit run Deepfake_Det,py" and opens in a browser. The current accuracy is 56% AUC.


# tools/libraries
-streamlit
-tensorflow
-keras
-matplotlib
-CV2
-shutil
-sklearn
-numpy

# references
@inproceedings{Celeb_DF_cvpr20,
   author = {Yuezun Li, Xin Yang, Pu Sun, Honggang Qi and Siwei Lyu},
   title = {Celeb-DF: A Large-scale Challenging Dataset for DeepFake Forensics},
   booktitle= {IEEE Conference on Computer Vision and Patten Recognition (CVPR)},
   year = {2020}
}
