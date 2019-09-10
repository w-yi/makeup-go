There should be 2 folders in this directory.

--|-01FaceDetection
  |-02Model

=========
01FaceDetection
=========
This folder contains the code to detect the faces in an image, crop it and save the cropped image.
Please refer to the README.md in that folder for more information.

=========
02Model
=========
This folder contains the code to train the model and perform validation.

=========
Instructions
=========
<------------------------------ QUICKEST GUIDE ------------------------------>
1> Quick Validation 
We have put some cropped and beautified images in <./02model/data/EECS442_Makeup_Go/result_test>. In order to
get the model's output on those beautified images, change directory to <./02Model> and run the following code

'''
python test.py -lp test
'''

You should be able to see the output under <./02model/output/>.
<------------------------------ QUICKEST GUIDE ------------------------------>

2> Makeup-Go on your own image
In order to start from the beginning, you should prepare a portrait and put it in <./01Dataset/faces> and run

'''
python3 detect_crop.py
'''

This will detect the face in the portrait and crop the image to the desired size. The output will be in <./01Dataset/result>.
Then, you can use your own image beautification APP to perform whitening and skin-smoothing on the cropped image.
In order to perform Makeup-Go, put the beautified image in <./02model/data/EECS442_Makeup_Go/result_test>, change directory to
<./02Model> and run the following code

'''
python test.py -lp test
'''

You should be able to see the output under <./02model/output/>.

3> Train your own model
In order to train your own model, please change directory to <./02model> and run

'''
python train.py -lp XX<XX.yml in param/> -op XX.XX=XX<optional>
'''
For example:
'''
python train.py -lp train -op network.kwargs.nonlinear=ReLU,optimizer.name=Adam
'''

We have included a small portion of training dataset so that you can run the code. If you want to train on your own dataset, here are some high level steps.
i)   Face Detection and Image Cropping using the instructions in <./01FaceDetection>
ii)  Beautify the cropped images.
iii) Put the original cropped images and beautified cropped images in the same directory but under two folders.
iv)  Copy the <./02Model/PCA_preprocess.py> to that folder and run
'''
python PCA_preprocess.py -o <original_image_directory> -b <beautified_image_directory>
'''
And copy the yielding <.t> files to <./02Model>.
v)   Copy the original images and beautified images to <./02Model/data/EECS442_Makeup_Go/result_original> and <./02Model/data/EECS442_Makeup_Go/result_beautified> respectively.
vi)  Run the training code

'''
python train.py -lp XX<XX.yml in param/> -op XX.XX=XX<optional>
'''
For example:
'''
python train.py -lp train -op network.kwargs.nonlinear=ReLU,optimizer.name=Adam
'''

=========
Questions
=========
If you have any question, send an email to kinmark@umich.edu and we will respond to you asap. Enjoy.