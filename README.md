# Single Object Tracking Using Meanshift

My research in the field of visible object detection begins with the tracking of visual features from moving object using Mean Shift. Peter Meer et al. proposed a method for [Real-time tracking of non-rigid objects using mean shift](https://ieeexplore.ieee.org/document/854761). After reading this paper carefully, I wrote codes using python to realize the ideas of the authors. The codes have two versions, one with **opencv** built-in functions **calcBackProject** and **meanShift** and the other with purely **numpy**. Here are my thoughts for this paper and inspirations from it:

- Visual Feature Representation

Representation of visual features, such as color or texture, of a moving object is crucial, when we use a RGB camera to track it. Because the statistical distributions of visual features can characterize the object of interest. The authors employed the color histogram as color density estimation for its low computational cost imposed by real-time processing. In my implementation, a 2D color histogram for hue and saturation results in faster convergence and higher FPS than the 1D histogram for hue. This setting can also be imitated and implemented in processing multispectral imaging. When running a multispectral video frame by frame, a 2D histogram for hue and infrared or histogram for mid-wave-infrared and long-wave-infrared is computed as estimation for multispectral features for both target model and target candidates.

- Line Search Optimization Method

The optimal location of the target candidate center **y** in the current frame is found by maximizing the Bhattacharyya coefficient. This optimization problem is according to the paper equals to maximizing the weighted Kernel Density Estimation (KDE) among all the pixels in the candidate window w.r.t. pixel position **y**. Under the assumption that the band width of the kernel always stays within the range of the candidate window, we can only turn this optimization problem into maximizing the weighted KDE among all the pixels in the whole current frame w.r.t. pixel position **y**. If converged, the result leads to the global optimum. That is the reason why I choose the 2D Epanechnikov kernel function in my codes and solve this optimization problem using Line Search method (see the flow chart below), instead of simply iterating among all the pixels.

![image](https://github.com/1996JCZhou/Single-Object-Tracking/blob/master/line.png)

What interests me most in this [paper](https://ieeexplore.ieee.org/document/854761) is that the author proposed a theorem **Theorem 1** on the Page 2 to provide a sufficient convergence condition for the optimization problem. This **Theorem 1** provides a theoretical basis for setting the step length to be the length of the Mean Shift vector in every iteration.

- Improvements of the method

When running a video, we can often clearly find out that the size of the target object will change frame by frame. That means in this situation, the estimation of the color distribution for the target object will no longer work for the following frames. So, the size of the target candidate and the kernel band width need to change according the changing target model. This problem can be solved by the **camshift** method. My another ongoing work is to introduce the **Kalman Filter** in the whole Mean Shift setting. With the **Kalman Filter**, the target object can still be tracked, even though it moves faster or hides behind another object. I am truely excited to realize it.
