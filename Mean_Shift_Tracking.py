import cv2, argparse
import numpy as np
from numba   import jit
from time    import time


class MSTracking:
    def init(self,
             frame,
             rect,
             dim,
             num_bins      =15,
             threshold     =1e-9,
             epsilon       =1e-5,
             max_iterations=200):
        """
        Parameter Initialization.

        Args:
            frame          (Numpy array): Current video frame.
            rect           (Numpy array): Defined target window (ROI) position and size.
            dim            (Integer)    : Dimension of the histogram.
            num_bins       (Integer)    : Number of value intervalls to display the histogram.
            threshold      (Float)      : Threshold of the target candidate histogram to prevent zeros in the denominator.
            epsilon        (Float)      : Termination criterion.
            max_iterations (Integer)    : Maximal iteration steps to find a higher Bhattacharyya coefficient.
        """

        """Parameter initialization."""

        # Read the size of the current frame.
        self.video_height, self.video_width, self.channels = frame.shape

        # Read the position and size of the defined ROI.
        x, y, w, h = np.round(rect)
        roi_center = np.round(np.array([x + w / 2, y + h / 2]))
        self.roi_width = w
        self.roi_height = h

        # Read the dimension of the histogram.
        self.DIMENSION = dim

        # Read the number of value intervalls to display the histogram.
        # If every pixel Hue value has to be considered in the histogram,
        # then we have 180 bins.
        self.bins = num_bins

        # Read the threshold of the target candidate histogram
        # to prevent zeros in the denominator.
        self.threshold = threshold

        # Read the termination criteria.
        self.epsilon = epsilon
        self.max_iterations = max_iterations
    ### ----------------------------------------------------------------------------
    ### ----------------------------------------------------------------------------
        """Calculate the estimation of the color representation of the target model.
           (histogram with kernel function, probability of the color in the target model)
           Paper: (19) - (20).
        """
        # Take the ROI out from the current video frame.
        # Compute the estimated target model color probability distribution (histogram)
        # given the center and size of the ROI.
        target_toi = self.extract_image_patch(frame, roi_center, w, h)
        self.target_model = self.color_distribution(target_toi)
        # Consider the current center of the target model (ROI)
        # as the starting point for the tracking in the next frame.
        self.prev_center = roi_center


    def track(self, frame):
        """
        Find the optimal location of the target candidate in the current video frame.

        Args:
            frame (Numpy array): The current video frame.

        Returns:
            bounding box (Numpy array): The positions of the top left and
                                        bottom right pixels of the bonding box.
        """

        while True:
            roi_width, roi_height = self.roi_width, self.roi_height
            video_height, video_width = self.video_height, self.video_width
    ### ----------------------------------------------------------------------------
    ### ----------------------------------------------------------------------------
            """STEP 1: Calculate the estimation of the color probability distribution (histogram) of the target candidate."""

            """Paper: (21) - (22)."""
            img_patch = self.extract_image_patch(frame, self.prev_center, roi_width, roi_height)
            candidate_model = self.color_distribution(img_patch)

            """Evaluate the Bhattacharyya coefficient. Paper: (17)."""
            b_coefficient = self.compute_bhattacharyya_coefficient(self.target_model, candidate_model)
    ### ----------------------------------------------------------------------------
    ### ----------------------------------------------------------------------------
            """STEP 2: Derive the data weights. Paper: (25)."""
            weights = self.compute_weights(img_patch, self.target_model, candidate_model)
    ### ----------------------------------------------------------------------------
    ### ----------------------------------------------------------------------------
            """STEP 3: Compute the mean shift vector, using Epanechnikov kernel,
               and derive the new location of the target.
               Paper: (26).
            """
            new_Location = self.compute_meanshift_vector(img_patch, weights)

            """Evaluate the Bhattacharyya coefficient. Paper: (17)."""
            new_img_patch = self.extract_image_patch(frame, new_Location, roi_width, roi_height)
            new_color_model = self.color_distribution(new_img_patch)
            new_b_coefficient = self.compute_bhattacharyya_coefficient(self.target_model, new_color_model)

            """Check if the Bhattacharyya coefficient is higher than the previous center.
               If not, take the half between both centers continuously
               until we find a higher Bhattacharyya coefficient.
               Paper: Algorithm (4).
            """
            Iteration_Steps = 1
            while new_b_coefficient < b_coefficient and Iteration_Steps <= self.max_iterations:
                Iteration_Steps += 1
                new_Location = 0.5 * (self.prev_center + new_Location)
                new_img_patch = self.extract_image_patch(frame, new_Location, roi_width, roi_height)
                new_color_model = self.color_distribution(new_img_patch)
                new_b_coefficient = self.compute_bhattacharyya_coefficient(self.target_model, new_color_model)
    ### ----------------------------------------------------------------------------
    ### ----------------------------------------------------------------------------
            """STEP 4: Continuously search for the optimal target candidate
               until the termination criterion are broken.
            """

            """If two centers stay within the same pixel,
               then we have found the location of the optimal target candidate
               as 'self.prev_center'.
               Paper: Algorithm (5).
            """
            # 'np.linalg.norm()': Calculate the norm of a vector.
            if np.linalg.norm(new_Location - self.prev_center, ord=1) < self.epsilon:
            # if np.sqrt(np.power(new_Location[0]-self.prev_center[0], 2) + \
            #     np.power(new_Location[1]-self.prev_center[1], 2)) < self.epsilon:
                self.prev_center = new_Location
                break

            self.prev_center = new_Location

        Left_Top_x = np.round(max(self.prev_center[0] - roi_width / 2, 0))
        Left_Top_y = np.round(max(self.prev_center[1] - roi_height / 2, 0))
        Right_Bottom_x = np.round(min(video_width-1, Left_Top_x + roi_width))
        Right_Bottom_y = np.round(min(video_height-1, Left_Top_y + roi_height))

        return np.array([Left_Top_x, Left_Top_y, Right_Bottom_x, Right_Bottom_y])


    def extract_image_patch(self, image, center, width, height):
        """
        Extract the image patch from the current frame
        given the center position and the size of the ROI
        to prepare for the calculation of the color histogram.
        (no guarantee the same size as the initial ROI)

        Args:
            image  (Numpy array): The current video frame.
            center (Numpy array): The position (x_coordinate, y_coordinate) of the ROI.
            width  (Integer)    : Width of the ROI.
            height (Integer)    : Height of ROI.

        Returns:
            image[h:h2+1, w:w2+1, :] (Numpy array): The ROI of the current video frame.
        """

        x = center[0] - width/2
        y = center[1] - height/2

        video_height, video_width = self.video_height, self.video_width
        h1 = int(min(video_height - 1, y + height))
        w1 = int(min(video_width - 1, x + width))

        # h = int(max(y, 1))
        # w = int(max(x, 1))
        h = int(max(y, 0))
        w = int(max(x, 0))

        return image[h:h1+1, w:w1+1, :]


    @jit(cache=True)
    def compute_weights(self, img_patch, q_target, p_current):
        """
        Derive the data weights.
        Paper: (25).

        Args:
            img_patch [Numpy array]: The current image patch as target candidate.
            q_target  [Numpy array]: Target model histogram.
            p_current [Numpy array]: Target candidate histogram.

        Returns:
            weights [Numpy array]: Data weights for every pixel position.
        """

        bins = self.bins
        h, w, c = img_patch.shape

        """Compute the ratio vector between both histograms,
           which is independent of the pixel position.
           'q_target'.shape = 'p_current'.shape = (bins, bins).
        """
        # As denominator, there must be non zero values in the target candidate histogram,
        # when we employ the Taylor expansion around the target candidate histogram.
        ratio = np.sqrt(q_target / p_current)                                              
        ratio[p_current <= self.threshold] = 0
        # Do not consider 'fake' zero values in the target candidate histogram!

        """Compute weights over all the image patch for every pixel."""
        weights = np.zeros((h, w))

        if self.DIMENSION == 1:
            img_patch = cv2.cvtColor(img_patch, cv2.COLOR_BGR2HSV)

            """Check: in which bin the current pixel value lies in."""
            hue_patch = (img_patch[:, :, 0] / 180 * bins).astype('int') # 'hue_patch'.shape = (h, w).
            hue_patch[hue_patch == bins] = bins - 1 # 'hue_patch' take values from 0 to 'bins'-1.
            weights += ratio[hue_patch]

        if self.DIMENSION == 2:
            img_patch = cv2.cvtColor(img_patch, cv2.COLOR_BGR2HSV)

            """Check: in which bin the current pixel value lies in."""
            hue_patch = (img_patch[:, :, 0] / 180 * bins).astype('int') # 'hue_patch'.shape = (h, w).
            sat_patch = (img_patch[:, :, 1] / 255 * bins).astype('int') # 'sat_patch'.shape = (h, w).
            hue_patch[hue_patch == bins] = bins - 1 # 'hue_patch' takes values from 0 to 'bins'-1.
            sat_patch[sat_patch == bins] = bins - 1 # 'sat_patch' takes values from 0 to 'bins'-1.
            weights += ratio[hue_patch, sat_patch]

        return weights


    def compute_meanshift_vector(self, img_patch, weights):
        """
        Compute the new location of the target candidate based on the mean shift vector.
        (no guarantee the new location is optimal)
        Paper: (26).

        Args:
            img_patch [Numpy array]: The current image patch as target candidate.
            weights   [Numpy array]: Weight for every pixel position in the image patch.

        Returns:
            new_Location [Numpy array]: Rounded new location of the target candidate.
        """

        h, w, c = img_patch.shape
        # Calculate the center pixel position of the current image patch. 
        center = np.floor(np.array([h, w]) / 2)

        # get array of coordinates
        x, y = np.meshgrid(np.linspace(1, w, w), np.linspace(1, h, h))
        # 这里减去 sz，就是公式中的减去候选区域中的中心点
        x = x - center[1] + self.prev_center[0]
        y = y - center[0] + self.prev_center[1]
        z = np.array([np.sum(x * weights), np.sum(y * weights)] / np.sum(weights))

        return np.round(z)


    @staticmethod
    def compute_bhattacharyya_coefficient(p, q):
        """Paper: (17)."""
        return np.sum(np.sqrt(p * q))


    @jit(cache=True)
    def color_distribution(self, img_patch):
        """
        Calculate the estimated color probability distribution (histogram)
        of the extracted image patch.
        Paper: (19) - (20) and (21) - (22).

        Args:
            img_patch (Numpy array): The extracted image patch.

        Returns:
            histogram (Numpy array): The estimated color probability distribution
                                     (histogram) of the extracted image patch.
        """

        bins = self.bins
        h, w, c = img_patch.shape

        """Compute distances between each pixel position inside the image patch
           and the center of the image patch.
        """
        # Center of the image patch being one of the lattice nodes.
        center = np.round(np.array([w/2, h/2]))
        # Construct a lattice for the image patch.
        x, y = np.meshgrid(np.linspace(1, w, w), np.linspace(1, h, h))
        # Compute distances.
        dist = np.sqrt( np.power(x-center[0], 2) + np.power(y-center[1], 2) )
        dist = dist / np.max(dist) # Normalize distances with 'h' = 'np.max(dist)'.

        img_patch = cv2.cvtColor(img_patch, cv2.COLOR_BGR2HSV)
        if self.DIMENSION == 1:
            """Build a 1D count 'cd' only for hue."""
            cd = np.zeros(bins)
        else:
            """Build a 2D count 'cd' for hue and saturation."""
            cd = np.zeros((bins, bins))

        if self.DIMENSION == 1:
            for i in range(h):
                for j in range(w):

                    """Consider location weights:
                       Employ a 2D Epanechnikov kernel profile (k),
                       which is convex and monotonic decreasing,
                       to each pixel position.
                       (d=2, cd=np.pi*(1**2))
                    """
                    if dist[i, j] ** 2 <= 1:
                        KE_Value = 2 / np.pi * (1 - dist[i, j] ** 2)
                    else:
                        KE_Value = 0

                    """Build a 1D count for hue."""
                    # Hue.
                    h_index = int(img_patch[i, j, 0] / 180 * bins) # 'img_patch[i, j, 0]': Hue value ([0, 180]) of the pixel (i, j).
                    h_index = h_index - 1 if h_index == bins else h_index # 'img_patch[i, j, 0]' == 180, then 'h_index' = 'h_index' - 1.
                    # 'h_index' = 0, ..., bins-1, in total 'bins' bins.

                    cd[h_index] += KE_Value

        if self.DIMENSION == 2:
            for i in range(h):
                for j in range(w):

                    """Consider location weights:
                       Employ a 2D Epanechnikov kernel profile (k),
                       which is convex and monotonic decreasing,
                       to each pixel position.
                       (d=2, cd=np.pi*(1**2))
                    """
                    if dist[i, j] ** 2 <= 1:
                        KE_Value = 2 / np.pi * (1 - dist[i, j] ** 2)
                    else:
                        KE_Value = 0

                    """Build a 2D count for hue and saturation."""
                    # Hue.
                    h_index = int(img_patch[i, j, 0] / 180 * bins) # 'img_patch[i, j, 0]': Hue value ([0, 180]) of the pixel (i, j).
                    h_index = h_index - 1 if h_index == bins else h_index # 'img_patch[i, j, 0]' == 180, then 'h_index' = 'h_index' - 1.
                    # 'h_index' = 0, ..., bins-1, in total 'bins' bins.

                    # Saturation.
                    s_index = int(img_patch[i, j, 1] / 255 * bins) # 'img_patch[i, j, 1]': Saturation value ([0, 255]) of the pixel (i, j).
                    s_index = s_index - 1 if s_index == bins else s_index # 'img_patch[i, j, 1]' == 255, then 's_index' = 's_index' - 1.
                    # 's_index' = 0, ..., bins-1, in total 'bins' bins.

                    cd[h_index, s_index] += KE_Value

        """Normalize the count to build a histogram."""
        cd = cd / np.sum(cd)
        cd[cd <= self.threshold] = self.threshold # To ensure non zero values.

        return cd


### ----------------------------------------------------------------------------
### ----------------------------------------------------------------------------
"""Draw the target window."""
Select_Object = False
Initialize_Tracker = False
Tracking_On = False
ix, iy, cx, cy = -1, -1, -1, -1
w, h = 0, 0


def draw_target_window(event, x, y, flags, param):

    global Select_Object, Initialize_Tracker, Tracking_On, ix, iy, cx, cy, w, h

    # If we press the left mouse button.
    if event == cv2.EVENT_LBUTTONDOWN:
        # Then we begin to select the target window.
        Select_Object = True
        # But we still do not begin tracking.
        Tracking_On = False

        # 'ix', 'iy': Pixel position of the top left pixel has been found.
        ix, iy = x, y
        # 'cx', 'cy': Pixel position to find the botton right pixel.
        cx, cy = x, y

    # If we press the left mouse button
    # and move the mouse without releasing the pressed left button.
    elif event == cv2.EVENT_MOUSEMOVE:
        cx, cy = x, y

    # If we release the pressed left button.
    elif event == cv2.EVENT_LBUTTONUP:
        # Then we stop selecting the target.
        Select_Object = False

        # If the target window is big enough.
        if abs(x - ix) > 10 and abs(y - iy) > 10:
            # Then we have found the top left pixel position of the target window (ix, iy)
            # and its height and width (w, h). 
            w, h = abs(x - ix), abs(y - iy)
            ix, iy = min(x, ix), min(y, iy)
            # After selecting the target window, we begin to initailize the tracker.
            Initialize_Tracker = True
        # If the target window is not big enough, then we do not begin to track.
        else:
            Tracking_On = False
### ----------------------------------------------------------------------------
### ----------------------------------------------------------------------------
"""Hyperparameter configuration."""
def config_parser():
    parser = argparse.ArgumentParser()
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
    """Basic options."""
    parser.add_argument("--name_window", type=str,
                        default="tracking",
                        help="Name of the window for display.")
    parser.add_argument('--video_path', type=str,
                        default="D:\\Python Spyder\\Objekterkennung\\Meanschift\\slow_traffic_small.mp4",
                        help='Video file path.')
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
    """Tracking."""
    parser.add_argument("--dim_histogram", type=int,
                        default=2,
                        help="Dimension of the histogram.")
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
    """Display."""
    parser.add_argument("--interval_frames", type=int,
                        default=30,
                        help="Every 'interval_frames' to display a video frame, \
                              controlling the video display speed.")
    parser.add_argument("--duration", type=int,
                        default=0.01,
                        help="Duration for calculation.")

    return parser


if __name__ == '__main__':

    """Hyperparameter configuration."""
    parser = config_parser()
    args = parser.parse_args()
### ----------------------------------------------------------------------------
### ----------------------------------------------------------------------------
    """Open a window for display."""
    cv2.namedWindow(args.name_window, cv2.WINDOW_KEEPRATIO)
    cv2.setMouseCallback(args.name_window, draw_target_window)
### ----------------------------------------------------------------------------
### ----------------------------------------------------------------------------
    """Instantiate a Mean Shift Tracker without initialization."""
    ms_tracker = MSTracking()
### ----------------------------------------------------------------------------
### ----------------------------------------------------------------------------
    """Load video."""
    cap = cv2.VideoCapture(args.video_path)

    while True:

        """Run every frame of it until it ends ('ret' = False)."""
        ret, frame = cap.read()
        if not ret:
            break
    ### ----------------------------------------------------------------------------
    ### ----------------------------------------------------------------------------
        # Selecting the target.
        # If we press the left mouse button, then we begin to select the target.
        if Select_Object:

            """Choose a target window."""
            cv2.rectangle(frame, (ix, iy), (cx, cy), (0, 255, 255), 2)
    ### ----------------------------------------------------------------------------
    ### ----------------------------------------------------------------------------
        # Initialize the tracker after selecting the target.
        # If we release the pressed left mouse button, then we begin to initialize the tracker.
        elif Initialize_Tracker:

            """Draw the selected target window."""
            cv2.rectangle(frame, (ix, iy), (ix + w, iy + h), (0, 255, 255), 2)

            """Initialize the Tracker instance."""
            ms_tracker.init(frame=frame, rect=np.array([ix, iy, w, h]), dim=args.dim_histogram)

            # Since we initialize the tracker only once,
            # we begin to track the target window in the following frames.
            Initialize_Tracker = False
            Tracking_On = True
    ### ----------------------------------------------------------------------------
    ### ----------------------------------------------------------------------------
        # Since we initialize the tracker only once,
        # we begin to track the target window in the following frames.
        elif Tracking_On:

            """Begin to record calculation time."""
            t0 = time()

            """Track the target window in the current video frame
               and output the optimal target candidate window."""
            bounding_box = ms_tracker.track(frame)

            """Record calculation time."""
            t1 = time()

            """Draw the optimal target candidate window."""
            bounding_box = list(bounding_box.astype('int'))
            cv2.rectangle(frame, (bounding_box[0], bounding_box[1]), (bounding_box[2], bounding_box[3]), (0, 255, 255), 2)

            duration = 0.8 * args.duration + 0.2 * (t1 - t0)
            # duration = t1 - t0
            cv2.putText(frame, 'FPS: ' + str(1 / duration)[:4].strip('.'), (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
### ----------------------------------------------------------------------------
### ----------------------------------------------------------------------------
        # Refresh the window "tracking"
        # with the current video frame
        # and all the drawing on the current video frame,
        # neglecting all the drawing on the previous video frames.
        # (different from the static image).
        cv2.imshow(args.name_window, frame)

        c = cv2.waitKey(args.interval_frames) & 0xFF
        if c == 27 or c == ord('q'):
            break
### ----------------------------------------------------------------------------
### ----------------------------------------------------------------------------
    cap.release()
    cv2.destroyAllWindows()
