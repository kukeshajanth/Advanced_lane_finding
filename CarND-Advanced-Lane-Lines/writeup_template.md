Advanced Lane Detection
===

[//]: # (Image References)

[image1]: ./project_video.gif

[image2]: ./camera_cal/calibration2.jpg

[image3]: ./output_images/Undistorted_image.jpg

[image4]: ./output_images/sobel_x.jpg

[image5]: ./output_images/sobel_y.jpg

[image6]: ./output_images/Thresholded_Magnitude.jpg

[image7]: ./output_images/Thresholded_Gradient_Direction.jpg

[image8]: ./output_images/Thresholded_S.jpg

[image9]: ./output_images/combined_threshold_img.jpg

[image10]: ./output_images/warped_img.jpg

[image11]: ./output_images/out_img_1.jpg

[image12]: ./output_images/result.jpg

[image13]: ./output_images/Final_img.jpg

[image14]: ./output_images/finalized.jpg

[image15]: ./test_images/test1.jpg

[image16]: ./project_solution.mp4





---
In this Advanced Lane Detection project, I have applied computer vision techniques to create a pipeline to identify the lane boundaries in a video from camera on a car.

![alt text][image1]

The steps involved in this project are :

- Step 0 - Importing the required packages.
- Step 1 - Compute the camera calibration matrix and applying distortion correction to new images.
- Step 2 - Creating threshold binary images.
- Step 3 - Combined thresholds to detect the line.
- Step 4 - Perspective Transform ("birds-eye view").
- Step 5 - Creating a histogram.
- Step 6 - Detect the lane pixels to find the lane boundary.
- Step 7 - Warping the fit from rectified image onto the original image
- Step 8 - Calculating curvature and camera offset.
- Step 9 - Add Curvature and offset to the image.
- Step 10 - Run the process in the video


## Step 0 - Importing the required packages:

I have imported the following packages to assist me with this project.

- Numpy 
- Matplotlib
- cv2
- moviepy
- IPython

## Step 1 - Compute the camera calibration matrix and applying distortion correction to new images.

Next step, to get the calibration matrix for camera. I have used chessboard images to get the calibration matrix using `cv2.findChessboardCorners()` and `cv2.calibrateCamera`.

Finally, undistorting the image using `cv2.undistort()`. This is used to convert the image to undistored form using the calculated camera matrix.

### Distorted Image and undistorted image
![alt text][image3]





## Step 2 - Creating threshold binary images.

### Original image:

![alt text][image15]

I have defined following functions to create thresholded binary image. 

- `abs_sobel()`  - to calculate x and y direction gradient
- `mag_thresh()` - to calculate gradient magnitude
- `dir_threshold()` - to calculate gradient direction
- `hls_select()` - to calculate color threshold

Finally to combine all these, I have defined a `threshold_combined()` function.

Results of the functions are as follows:

        
- X directional gradient and Y directional graient

 ```python
 def abs_sobel(img, orient = 'x', thresh_min = 0, thresh_max = 255):
    
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1 , 0)
    if orient == 'y':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0 , 1)
        
    abs_sobel = np.absolute(sobel)

    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    
    sxbinary = np.zeros_like(scaled_sobel)
    
    sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
    
    return sxbinary 
``` 

 ![alt text][image4]
 
 ![alt text][image5]
 
 
- Calculating Gradient magnitude

```python
   def mag_thresh(img , sobel_kernel =3 , mag_thresh = (0,255)):
    
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1 , 0, ksize = sobel_kernel)
    
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0 , 1, ksize = sobel_kernel)
        
    magnitude = np.sqrt(sobelx**2 + sobely**2)

    scaled_sobel = np.uint8(255 * magnitude / np.max(magnitude))
    
    magbinary = np.zeros_like(scaled_sobel)
    
    magbinary[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1])] = 1
    
    return magbinary
```

 ![alt text][image6]
 
- Calculating gradient direction
```python
 def dir_threshold(img, sobel_kernel = 3, thresh= (0, np.pi/2)):
    
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize = sobel_kernel)
    
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize = sobel_kernel)
    
    dirgrad = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    
    dirbinary = np.zeros_like(dirgrad)
    
    dirbinary[(dirgrad >= thresh[0]) & (dirgrad <= thresh[1])] = 1
    
    return dirbinary
```
 ![alt text][image7]
 
 
- Calculating color threshold using hls color space

```python
 def hls_select(img, threshs=(0,255), threshl = (0,255)):
    
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    
    s = hls[:,:,2]
    
    l = hls[:,:,1]
    hls_binary_s = np.zeros_like(s)
    
    #hls_binary_l = np.zeros_like(l)
    
    #hls_binary = np.zeros_like(s)
    
    hls_binary_s[(s> threshs[0]) & (s <= threshs[1])] = 1
    
    #hls_binary_l[(l> threshl[0]) & (l <= threshl[1])] = 1
    
    #hls_binary[((hls_binary_s == 1) | (hls_binary_l == 1) )] = 1
    
    return hls_binary_s
```

  ![alt text][image8]
  
- Finally, combining all these thresholds to get the desired output image with detected edges.

```python
 def threshold_combined(sobelx, sobely, magbinary, dirbinary, hls_binary):
    
    combined = np.zeros_like(dirbinary)
    
    combined[((sobelx == 1) & (sobely ==1)) | ((magbinary == 1) & (dirbinary==1)) | (hls_binary == 1)] = 1
    
    return combined
```
  
  ![alt text][image9]
  
  
## Step 4 - Perspective Transform ("birds-eye view"):

To get the curvature, I have converted the sample image to birds_eye view using perspective transform.

I have followed the following steps:

- First, define coordinates for a polygon, so that , this polygon will be used to mask the image . To focus the attention to the part of the image, where the probability of finding the required line is high.

- Then, define destination coordinates, to view the polygon from bird's eye view. As that, polygon appears as rectangle in warped space.

- Finally,  I have used `cv2.getPerspectiveTransform()` to get the perspective transform M and inverse perspective transform Minv. Then, M, Minv is used in `cv2.warpPerspective()` to get the warped image.

```python
 def perspective_transform(img, src, dst):
    
    img_size = (img.shape[1], img.shape[0])
    
    M = cv2.getPerspectiveTransform(src, dst)
    
    Minv = cv2.getPerspectiveTransform(dst, src)
    
    warped = cv2.warpPerspective( img, M, img_size, flags = cv2.INTER_LINEAR)
    
    return warped, Minv

src = np.float32([[240,720],[570,480],[730,480],[1180,720]])

dst = np.float32([[250,720],[250,0],[1100,0],[1100,720]])

warped_img, Minv = perspective_transform(combined_threshold_img, src ,dst)
                   
src_pts =np.array([[240,720],[570,480],[730,480],[1180,720]], np.int32)

src_pts = src_pts.reshape((-1,1,2))

img_copy = img.copy()

cv2.polylines(img_copy, [src_pts], True, (255,0,0), thickness = 5)

visualization(img_copy, 'Source image', warped_img , 'warped_img')
```

 ![alt text][image10]
 

## Step 5 - Creating a histogram:

 Create a histogram based on summation of pixel values along vertical axis. Since the sum of bright pixels are higher, this would help us differentiate left and right lanes.
 
```python
 def histo(img):
    
    histogram = np.sum(img[img.shape[0]//2:,:], axis = 0)
    
    return histogram
```

##  Step 6 - Detect the lane pixels to find the lane boundary:

Starting point for left and right lane is selected based on the maximum  values to the left and right of the histogram midpoint.

Then, using sliding window, whose height is decided based on number of windows and its width a based on assumed margin. 

Based on the non_zero pixels in the window region and by taking mean of the non zero pixels in the window, x position of the lane is determined within the window.

After doing this for every window, these non zero pixels locations are obtained. Then, we use `np.polyfit()` to calculate the polynomial that is passing through the points and a line is ploted on the image.

```python
 def finding_lane_pixels(img, out_img):
    
    hist = histo(img)
    
    midpoint = np.int(hist.shape[0]//2)
    
    leftx_base = np.argmax(hist[:midpoint])
    
    rightx_base = np.argmax(hist[midpoint:]) + midpoint
    
    nwindows = 9
    
    minpix = 50
    
    margin = 100
    
    window_height = np.int(img.shape[0]//nwindows)
    
    nonzero = img.nonzero()
    
    nonzerox = np.array(nonzero[1])
    
    nonzeroy = np.array(nonzero[0])
    
    leftx_current = leftx_base
    
    rightx_current = rightx_base
    
    left_lane_inds = []
    
    right_lane_inds = []
    
    for i in range(nwindows):
        
        y_low = img.shape[0] - (i+1)*window_height
        
        y_high = img.shape[0] - i*window_height
        
        xleft_low    = leftx_current  - margin
        xleft_high   = leftx_current  + margin
        xright_low   = rightx_current - margin
        xright_high  = rightx_current + margin
        
        left_inds = ((nonzeroy >= y_low) & (nonzeroy < y_high) & 
        (nonzerox >= xleft_low) &  (nonzerox < xleft_high)).nonzero()[0]
        right_inds = ((nonzeroy >= y_low) & (nonzeroy < y_high) & 
        (nonzerox >= xright_low) &  (nonzerox < xright_high)).nonzero()[0]
        
        cv2.rectangle(out_img,(xleft_low,y_low),
        (xleft_high,y_high),(0,255,0), 8) 
        cv2.rectangle(out_img,(xright_low,y_low),
        (xright_high,y_high),(0,255,0), 8) 
        
        left_lane_inds.append(left_inds)
        right_lane_inds.append(right_inds)
        
        if len(left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[left_inds]))
        if len(right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[right_inds]))

    
    
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    
    leftfit = np.polyfit(lefty, leftx ,2)
    rightfit = np.polyfit(righty, rightx ,2)
    
    ploty = np.linspace(0, img.shape[0]-1, img.shape[0])
    
    try:
        
        leftfitx = leftfit[0]*ploty**2 + leftfit[1]*ploty + leftfit[2]
        
        rightfitx = rightfit[0]*ploty**2 + rightfit[1]*ploty + rightfit[2]
        
    except:
        
        print('function failed')
        
        leftfitx = 1*ploty**2 + 1*ploty
        
        rightfitx = 1*ploty**2 + 1*ploty
        
    out_img[lefty,leftx] = [255,0,0]
    
    out_img[righty, rightx] = [0,0,255]
    
    plt.plot(leftfitx, ploty, color = 'yellow')
    
    plt.plot(rightfitx, ploty, color = 'yellow')
    
    
    return out_img, (leftfit, rightfit), (leftx, lefty, rightx, righty), ploty,( leftfitx,rightfitx) 
    

```

 ![alt text][image11]
 
 
Once the lines are selected, these can be used detect similar lines in the remaining frame of the video, this is done using `search_around()` function.

```python
 def search_around(img, out_img, linefit = None):
    
    if linefit is None:
        _ , linefit , xy_points, ploty , fitx = finding_lane_pixels(img, out_img)
    
    margin = 100
    
    nonzero = img.nonzero()
    
    nonzerox = np.array(nonzero[1])
    
    nonzeroy = np.array(nonzero[0])
    
    leftfit, rightfit = linefit
    
    left_lane_inds = ((nonzerox > (leftfit[0]*(nonzeroy**2)+ leftfit[1]*nonzeroy +
                                   leftfit[2] - margin)) & (nonzerox <(leftfit[0]*(nonzeroy**2)+ 
                                   leftfit[1]*nonzeroy + leftfit[2] + margin))).nonzero()[0]
    right_lane_inds = ((nonzerox > (rightfit[0]*(nonzeroy**2)+ rightfit[1]*nonzeroy +
                                   rightfit[2] - margin)) & (nonzerox <(rightfit[0]*(nonzeroy**2)+ 
                                   rightfit[1]*nonzeroy + rightfit[2] + margin))).nonzero()[0]
    
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    
    if (leftx.size == 0 or rightx.size == 0):
        return finding_lane_pixels(warped_img, out_warped_img)
        
    leftfit = np.polyfit(lefty, leftx, 2)
    rightfit = np.polyfit(righty, rightx,2)
    
    ploty = np.linspace(0, img.shape[0]-1, img.shape[0])
        
    leftfitx = leftfit[0]*ploty**2 + leftfit[1]*ploty + leftfit[2]
        
    rightfitx = rightfit[0]*ploty**2 + rightfit[1]*ploty + rightfit[2]
        
    window_img = np.zeros_like(out_img)
    
    out_img[lefty,leftx] = [255,0,0]
    
    out_img[righty, rightx] = [0, 0,255] 
    
    left_line_window1 = np.array([np.transpose(np.vstack([leftfitx - margin, ploty]))])
    
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([leftfitx + margin, ploty])))])
    
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
     
    right_line_window1 = np.array([np.transpose(np.vstack([rightfitx - margin, ploty]))])
    
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([rightfitx + margin, ploty])))])
    
    right_line_pts = np.hstack((right_line_window1, right_line_window2))
    
    cv2.fillPoly(window_img, np.int_([left_line_pts]),(0,255,0))
    
    cv2.fillPoly(window_img, np.int_([right_line_pts]),(0,255,0))
    
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    
    plt.plot(leftfitx, ploty, color='yellow')
    
    plt.plot(rightfitx, ploty, color='yellow')
    
    fitx = (leftfitx,rightfitx)
    
    return result, linefit, ploty , fitx
    
    
```

 ![alt text][image12]


## Step 7 - Warping the fit from rectified image onto the original image:

In this step, the detected lane will be drawn on to the warped image. Then, using Minv and `cv2.warpPerspective()` , the warped image is brought back to its original shape and this new image is add to the original image using `cv2.addWeighted()`.

```python

 def boundary(img, warp, leftfitx, rightfitx, ploty, Minv):
    
    warp_zero = np.zeros_like(warp).astype(np.uint8)
    
    color_warp = np.dstack((warp_zero,warp_zero,warp_zero))
    
    left_line = np.array([np.transpose(np.vstack([leftfitx, ploty]))])
    
    right_line = np.array([np.flipud(np.transpose(np.vstack([rightfitx, ploty])))])
    
    pts = np.hstack((left_line, right_line))
    
    cv2.fillPoly(color_warp,np.int_([pts]),(0,255,0))
    
    img_shape = (img.shape[1], img.shape[0])
    
    newwarp = cv2.warpPerspective(color_warp, Minv, img_shape)
    
    final = cv2.addWeighted(img, 1, newwarp, 0.3, 0)
    
    return final
     
```

 ![alt text][image13]
 
 
##  Step 8 - Calculating curvature and camera offset:


Here, we calculate the curvature of lane and the offset of camera in the car to the lane. 

To calcualte the curvature, first using polyfit and changing the values from pixel world to real world using 

                `ym_per_pix = 30/720 
                 xm_per_pix = 3.7/700`

I found the polynomial and then using this formula to calcualte the curvature.

`R curve = (1+(2Ay+B)**2)**3/2 / ∣2A∣` 

Then, I found the offset using difference between the centre of the image and the centre of the lane.

```python

 def curvature(ploty, leftfitx, rightfitx):
    
    ym_per_pix = 30/720 
    
    xm_per_pix = 3.7/700
    
    yeval = np.max(ploty)
    
    leftfit = np.polyfit(ploty*ym_per_pix, leftfitx*xm_per_pix,2)
    
    rightfit = np.polyfit(ploty*ym_per_pix, rightfitx*xm_per_pix,2)
    
    left_curvature = ((1+(2*leftfit[0]*yeval*ym_per_pix + leftfit[1])**2)**1.5)/np.absolute(2*leftfit[0])
    
    right_curvature = ((1+(2*rightfit[0]*yeval*ym_per_pix + rightfit[1])**2)**1.5)/np.absolute(2*rightfit[0])
    
    return left_curvature, right_curvature
```
```python

 def camera_offset(leftx, rightx, img_shape = img.shape):
    
    camera_location = img.shape[1]//2
    
    lane_centre = (leftx[-1] + rightx[-1])/2
    
    offset = (camera_location - lane_centre) * (3.7/700)
    
    return offset

```

## Step 9 - Add Curvature and offset to the image:

I have added the curvature and offset to the camera using the following function.

```python

def add_info(img, leftx, rightx, leftfitx, rightfitx, ploty):
    
    left_curve, right_curve = curvature(ploty, leftfitx, rightfitx)
    
    cameraoffset = camera_offset(leftx, rightx)
    
    final_copy = img.copy()
    
    cv2.putText(final_copy, 'Left lane curvature: {}m'.format(np.round(left_curve)), 
                (170, 60), cv2.FONT_HERSHEY_SIMPLEX , 2, (50,255,50), 5)
    
    cv2.putText(final_copy, 'Right lane curvature: {}m'.format(np.round(right_curve)), 
                (170, 130), cv2.FONT_HERSHEY_SIMPLEX , 2, (50,255,50), 5)
    
    cv2.putText(final_copy, 'Camera offset Dist: {:.3f}m'.format(cameraoffset), 
                (170, 200), cv2.FONT_HERSHEY_SIMPLEX , 2, (255,255,50), 5)
    
    return final_copy

```
![alt text][image14]


## Step 10 - Run the process in the video:


I have tired 2 different approches to apply the process to the given video. First, I defined a class so that I can recall the `linefit` value again and pass it to the `search_around()` function. In the other method, I have used `finding_lane_pixels()` to find the lines and their polynomial fit in each frame of video. This is the basic difference between these two framework. This is followed application of all the functions which we have seen above. I have used first method in the first less-challenging video , as this method improves the speed at the same time adding deviation to the detection. Second method is slower, but performs a better job in challenging video compared to the first one. 

First pipeline:


```python

 
class Pipeline:
    
    def __init__(self,image):
        
        image = glob.glob(image)
        
        self.ret, self.mtx, self.dist, self.revecs, self.tvecs = calibrate_camera_img(img)
        
        self.linefit = None
        
    def __call__(self, img):

        

        undist = cv2.undistort(img, mtx, dist, None, mtx)

        sobelx = abs_sobel(img, orient = 'x', thresh_min = 20, thresh_max = 100)

        sobely = abs_sobel(img, orient = 'y', thresh_min = 20, thresh_max = 100)

        magbinary = mag_thresh(img, sobel_kernel = 9, mag_thresh =(60,180))

        dirbinary = dir_threshold(img, sobel_kernel = 19, thresh= (0.7,1.3))

        hls_binary = hls_select(img, threshs= (145,255), threshl = (120,255))

        combined_threshold_img = threshold_combined(sobelx, sobely, magbinary, dirbinary, hls_binary)

        src = np.float32([[240,720],[570,470],[730,470],[1180,720]])

        dst = np.float32([[250,720],[250,0],[1100,0],[1100,720]])

        warped_img, Minv = perspective_transform(combined_threshold_img, src ,dst)

        #out_warped_img, _ = perspective_transform(img, src, dst)

        #out_img, linefit , xy_points, ploty , fitx = finding_lane_pixels(warped_img, out_warped_img)

        out_warped_img1, _ = perspective_transform(img, src, dst)

        result, self.linefit, ploty , fitx = search_around(warped_img, out_warped_img1, self.linefit)

        final_img = boundary(img, warped_img, fitx[0], fitx[1], ploty, Minv)

        finalized = add_info(final_img,  fitx[0], fitx[1], ploty)

        return finalized

```

Second pipeline:

```python

def process_image(img, mtx = mtx, dist = dist):
    
    #ret, mtx, dist, revecs, tvecs = calibrate_camera_img(img)

    undist = cv2.undistort(img, mtx, dist, None, mtx)
    
    sobelx = abs_sobel(img, orient = 'x', thresh_min = 20, thresh_max = 100)
    
    sobely = abs_sobel(img, orient = 'y', thresh_min = 20, thresh_max = 100)

    magbinary = mag_thresh(img, sobel_kernel = 9, mag_thresh =(60,180))

    dirbinary = dir_threshold(img, sobel_kernel = 19, thresh= (0.7,1.3))

    hls_binary = hls_select(img, threshs= (145,255), threshl = (120,255))

    combined_threshold_img = threshold_combined(sobelx, sobely, magbinary, dirbinary, hls_binary)

    src = np.float32([[240,720],[570,470],[730,470],[1180,720]])

    dst = np.float32([[250,720],[250,0],[1100,0],[1100,720]])

    warped_img, Minv = perspective_transform(combined_threshold_img, src ,dst)
 
    out_warped_img, _ = perspective_transform(img, src, dst)
    
    out_img, linefit , xy_points, ploty , fitx = finding_lane_pixels(warped_img, out_warped_img)

    out_warped_img1, _ = perspective_transform(img, src, dst)

    #result = search_around(warped_img, out_warped_img1)
    
    final_img = boundary(img, warped_img, fitx[0], fitx[1], ploty, Minv)

    finalized = add_info(final_img, fitx[0], fitx[1], ploty)

    return finalized
    
```

Finally, I have used 

`from moviepy.editor import VideoFileClip
 from IPython.display import HTML` & `fl_image`
 
as in first project to apply this pipeline to the video.

You can find the final video in the folder as "./project_solution.mp4" 




---

## Discussion:


My pipeline has detected lanes in the project video, but it does not perform well on situations were there is more bright objects in the frame, which tricks the system into thinking, that there is a line over there. To overcome this, I have to find a way to remove the outliers and other than that I feel it does a good job in highway with less traffic. Better option would be to annotate the video and train the neural networks, which theoritically would perform better than this model.


