# **Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: https://raw.githubusercontent.com/ruanvdm11/Ruan_CARND_Term1_PROJ5/master/Reference_Images/HOGFeatures.png "HOG Features"
[image3]: ./examples/sliding_windows.jpg
[image4]: https://raw.githubusercontent.com/ruanvdm11/Ruan_CARND_Term1_PROJ5/master/Reference_Images/PipelineExamples.PNG "Pipeline Examples"
[image5]: https://raw.githubusercontent.com/ruanvdm11/Ruan_CARND_Term1_PROJ5/master/Reference_Images/PipelineHeatDetect.PNG "Heat Map Detection"
[image7]: https://raw.githubusercontent.com/ruanvdm11/Ruan_CARND_Term1_PROJ5/master/Reference_Images/BoundingBox.png "Output Boxes"
[image8]: https://raw.githubusercontent.com/ruanvdm11/Ruan_CARND_Term1_PROJ5/master/Reference_Images/VehicleDetectScreenshot.PNG "Detection Video"
[video1]: ./project_video.mp4

### [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
## 1. Histogram of Oriented Gradients (HOG)

### 1.1 Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the Second code cell of the IPython notebook. The use of HOG, from the `sklearn` library, enables the counting of gradients in an image with the goal of matching predefined templates to similar objects in an images.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=11`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

### 1.2 Explain how you settled on your final choice of HOG parameters.

I tried numerous variations of the parameters that define HOG and settled on the following numbers due to the accuracy it showed in the project video. Here is how the HOG features were calculated.

|Paramater || Vehicles|
| :------|| :-------|
| Image  || This is the image used for the calculation of HOG features|
| Orienations|| 10.5|
| Pixels per Cell|| 8|
| Cells per Block|| (2, 2)|


```python
def hogFeatures(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
                        
    if vis == True:
    
        features, hog_image = hog(img, orientations=orient, 
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), 
                                  transform_sqrt=False, 
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    else:   
    
        features = hog(img, orientations=orient, 
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), 
                       transform_sqrt=False, 
                       visualise=vis, feature_vector=feature_vec)
        return features
```

## 2. Classifier Details
### Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using the GTI and KITTI datasets with the following composition:

|Dataset || Vehicles|| Non Vehicles|
| :------|| :-------|| :-----------|
| GTI  || 3491|| 3900|
| KITTI|| 4000||  N/A|

Therefore, the Vehicle total is more than the non-vehicle total. Even though the rule of the states that the two datasets should be similar, this configuration gave the best results.

I ustilised the HOG features and the histogram of color paramaters in order to train the model. The spatial features lead to quite a bit of false postives that could not be filtered. The SVM was trained with the following code:

```python

svc = LinearSVC(C=1.7, class_weight=None, dual=True, fit_intercept=True,
intercept_scaling=1, loss='squared_hinge',
multi_class='ovr', penalty='l2', random_state=None, tol=0.00001,
verbose=0)
```

## 3. Sliding Window Search

The sliding window search was done by specifying the region that must be searched as well as the size of the search window. A scale was also imposed such that features could be searched for depending on their size ie. far or near to the camera. Also, it is necessary to implement a stepover value so that features were not missed.

### 3.1 Examples on images of how the pipeline is functioning. 

The optimisations that were done on the pipeline involved implementing different scale values, different orientation values, different sets of data, and attempting different kernels for the SVM.

Eventually the pipeline is using four different scale sizes and a Linear SVM. Here follows some examples of the images generated by the pipeline.

![alt text][image4]
---

## 4. Video Implementation

### 4.1 Video Result
The followiing link can be used to download the video. Also, a view link has been supplied. 
Here's a [link to my video result](https://github.com/ruanvdm11/Ruan_CARND_Term1_PROJ5/blob/master/p16Result.mp4)
(https://youtu.be/tdKI_zMfT8U)

[![alt text][image8]](https://www.youtube.com/watch?v=tdKI_zMfT8U)

### 4.2 Filters Implemented for Minimising False Positives

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used thresholds to identify 'individual' blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video and the bounding boxes then overlaid over a frame of the video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]


### Here the resulting bounding boxes are drawn onto the last frame in the series:
It is seen that the rusulting white boundary boxes have successfully been drawn over the two vehicles.
![alt text][image7]



---

## 5. Discussion

The biggest issue that was encountered was detection of vehicles when the are further away from the camara and also to a side.It is seen in the video that there is a brief section where the one vehicle is not fully detected. It is however deemed that this is okay because the vehicle is quite far away and not an immediate threat. When the vehicle gets closer again it is accurately picked up.

The second issue is solving time. It took quite a while for a video to be generated which is something that is not implementable in real life. Therefore, the two major adjustments would be to accurate detection at a distance as well as decreasing the calculation time.
 

