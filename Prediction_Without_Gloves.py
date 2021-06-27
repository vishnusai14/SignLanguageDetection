from keras.models import load_model
import numpy as np
import cv2

classifier = load_model('./modelfit.h5')

background = None
accumulated_weight = 0.5
ROI_top = 100
ROI_bottom = 300
ROI_right = 150
ROI_left = 350


def cal_accum_avg(frame, accumulated_weight):
    global background
    
    if background is None:
        background = frame.copy().astype("float")
        return None
    cv2.accumulateWeighted(frame, background, accumulated_weight)


def segment_hand(frame, threshold=25):
    global background
    
    diff = cv2.absdiff(background.astype("uint8"), frame)

    
    _ , thresholded = cv2.threshold(diff, threshold, 255,cv2.THRESH_BINARY)
    
     #Fetching contours in the frame (These contours can be of handor any other object in foreground) …

    contours, hierarchy = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    # If length of contours list = 0, means we didn't get anycontours...
    if len(contours) == 0:
        return None
    else:
        # The largest external contour should be the hand 
        hand_segment_max_cont = max(contours, key=cv2.contourArea)
        
        # Returning the hand segment(max contour) and thethresholded image of hand...
        return (thresholded, hand_segment_max_cont)




def ret_result(result):
    
   
    if result[0][0] == 1:
            return '1'
    elif result[0][1] == 1:
            return '2'
    elif result[0][2] == 1:
            return '3'
    elif result[0][3] == 1:
            return '4'
    elif result[0][4] == 1:
            return '5'
    elif result[0][5] == 1:
            return '6'
    elif result[0][6] == 1:
            return '7'
    elif result[0][7] == 1:
            return '8'
    elif result[0][8] == 1:
            return '9'
    elif result[0][9] == 1:
            return 'A'
    elif result[0][10] == 1:
            return 'All Gone'
    elif result[0][11] == 1:
            return 'B'
    elif result[0][12] == 1:
            return 'C'
    elif result[0][13] == 1:
            return 'D'
    elif result[0][14] == 1:
            return 'E'
    elif result[0][15] == 1:
            return 'F'
    elif result[0][16] == 1:
            return 'Friend'
    elif result[0][17] == 1:
            return 'G'
    elif result[0][18] == 1:
            return 'H'
    elif result[0][19] == 1 :
            return 'Hang'
    elif result[0][20] == 1:
            return 'House'
    elif result[0][21] == 1:
            return 'I'
    elif result[0][22] == 1:
            return 'J'
    elif result[0][23] == 1:
            return 'K'
    elif result[0][24] == 1:
            return 'L'
    elif result[0][25] == 1:
            return 'M'
    elif result[0][26] == 1 :
            return 'Middle'
    elif result[0][27] == 1:
            return 'Money'
    elif result[0][28] == 1:
            return 'N'
    elif result[0][29] == 1:
            return 'O'
    elif result[0][30] == 1:
            return 'Opposite'
    elif result[0][31] == 1:
            return 'P'
    elif result[0][32] == 1:
            return 'Q'
    elif result[0][33] == 1:
            return 'R'
    elif result[0][34] == 1:
            return 'S'
    elif result[0][35] == 1:
            return 'T'
    elif result[0][36] == 1:
            return 'U'
    elif result[0][37] == 1:
            return 'V'
    elif result[0][38] == 1:
            return 'W'
    elif result[0][39] == 1:
            return 'X'
    elif result[0][40] == 1:
            return 'Y'
    elif result[0][41] == 1:
            return 'Z'


cam = cv2.VideoCapture(0)
num_frames =0
while True:
    ret, frame = cam.read()

    # flipping the frame to prevent inverted image of capturedframe...
    
    frame = cv2.flip(frame, 1)

    frame_copy = frame.copy()

    # ROI from the frame
    roi = frame[ROI_top:ROI_bottom, ROI_right:ROI_left]

    gray_frame = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.GaussianBlur(gray_frame, (9, 9), 0)


    if num_frames < 70:
        
        cal_accum_avg(gray_frame, accumulated_weight)
        
        cv2.putText(frame_copy, "FETCHING BACKGROUND...PLEASE WAIT", (80, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
    
    else: 
        # segmenting the hand region
        hand = segment_hand(gray_frame)
        
        # Checking if we are able to detect the hand...
        if hand is not None:
            
            thresholded, hand_segment = hand

            # Drawing contours around hand segment
            cv2.drawContours(frame_copy, [hand_segment + (ROI_right,ROI_top)], -1, (255, 0, 0),1)
            
            cv2.imshow("Thesholded Hand Image", thresholded)
            
            thresholded = cv2.resize(thresholded, (64, 64))
            thresholded = cv2.cvtColor(thresholded,cv2.COLOR_GRAY2RGB)
            thresholded = np.reshape(thresholded,(1,thresholded.shape[0],thresholded.shape[1],3))
            
            pred = classifier.predict(thresholded)
            print(pred)
            cv2.putText(frame_copy, ret_result(pred),(170, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            
    # Draw ROI on frame_copy
    cv2.rectangle(frame_copy, (ROI_left, ROI_top), (ROI_right,
    ROI_bottom), (255,128,0), 3)

    # incrementing the number of frames for tracking
    num_frames += 1

    # Display the frame with segmented hand
    cv2.putText(frame_copy, "Sign recognition_ _ _",
    (10, 20), cv2.FONT_ITALIC, 0.5, (51,255,51), 1)
    cv2.imshow("Sign Detection", frame_copy)


    # Close windows with Esc
    k = cv2.waitKey(1) & 0xFF

    if k == 27:
        break

# Release the camera and destroy all the windows
cam.release()
cv2.destroyAllWindows()
