import cv2
import numpy as np
import matplotlib.pyplot as plt

from utils import get_parking_spots_bboxes, empty_or_not


def calc_diff(img1, img2):
    """returns the difference between two images

    Args:
        img1 (_type_): image 1
        img2 (_type_): image 2

    Returns:
        _type_: _description_
    """
    return np.mean(img1) - np.mean(img2)

# mask = "./mask_crop.png"
mask = "./mask_1920_1080.png"

# video_path = "./samples/parking_crop_loop.mp4"
video_path = "./samples/parking_1920_1080_loop.mp4"

# 0 is for grayscale
mask = cv2.imread(mask, 0)

# visualize video
cap = cv2.VideoCapture(video_path)

# 4 is for 4-connected components
# cv2.CV_32S is for 32-bit signed integer
connected_components = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)

# get the bounding boxes of the parking spots
spots = get_parking_spots_bboxes(connected_components)

# print the bbox of the first parking spot
#print(spots[0])

# initialize the status of the parking spots as None initially for all parking spots
spots_status = [None for j in spots]

diffs = [None for j in spots]

previous_frame = None 
ret = True
step = 30
frame_num = 0

while ret:
    # read the frame
    ret, frame = cap.read()
    
    if frame_num % step == 0 and previous_frame is not None :
        for spot_idx, spot in enumerate(spots):
            # get the location of the parking spot
            x1, y1, w, h = spot
            
            # crop the parking spot from the frame
            spot_crop = frame[y1:y1 + h, x1:x1 + w, :]
            
            # compute the difference between the current frame and the previous frame and store it in the diffs array
            diffs[spot_idx] = calc_diff(spot_crop, previous_frame[y1:y1 + h, x1:x1 + w, :])
            
        #print([diffs[j] for j in np.argsort(diffs)][::-1])
        # plt.figure()
        # plt.hist([diffs[j] / np.amax(diffs) for j in np.argsort(diffs)][::-1], bins=20)

        # if frame_num == 300:
        #     plt.show()
    
    
    if frame_num % step == 0: # OPTIMIZATION: update the status of parking spot once every 30 frames instead of every frame
        if previous_frame is None:
            arr_ = range(len(spots))
        else: # set the threshold to 0.4 from the histogram plots
            arr_ = [j for j in np.argsort(diffs) if diffs[j] / np.amax(diffs) > 0.4]
        
        # draw the bounding boxes around the parking spots
        # for spot_idx, spot in enumerate(spots):
        for spot_idx in arr_:
            
            # get the status of the parking spot
            spot = spots[spot_idx]
            # get the location of the parking spot
            x1, y1, w, h = spot
            
            # crop the parking spot from the frame
            spot_crop = frame[y1:y1 + h, x1:x1 + w, :]
            
            # get the status of the parking spot
            spot_status = empty_or_not(spot_crop)
            
            # update the status of the parking spot
            spots_status[spot_idx] = spot_status
    
    if frame_num % step == 0:
        previous_frame = frame.copy()
    
    
    for spot_idx, spot in enumerate(spots):
        # get the status of the parking spot
        spot_status = spots_status[spot_idx]
        # get the bbox location of the parking spot
        x1, y1, w, h = spots[spot_idx]
        
        if spot_status:
            # parking spot is empty; display green bounding box
            frame = cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)
        else: # parking spot is not empty; display red bounding box
            frame = cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 0, 255), 2)
    
    # add a black rectangle to display the number of available parking spots
    cv2.rectangle(frame, (80, 20), (550, 80), (0, 0, 0), -1)
    # display the number of available parking spots
    cv2.putText(frame, "Available spots: {} / {}".format(str(sum(spots_status)), str(len(spots_status))), (100, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    # fit the frame size to the screen
    cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
    # show the output frame
    cv2.imshow("frame", frame)
    # quit the video by pressing q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    frame_num += 1
    
# release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()