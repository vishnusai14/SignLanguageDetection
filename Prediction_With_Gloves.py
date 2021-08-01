from keras.models import load_model
import numpy as np
import cv2
classifier = load_model('./modelfit.h5')
ROI_top = 100
ROI_bottom = 300
ROI_right = 150
ROI_left = 350


#For Using Different Colors Use 
#This Website To Detect The Lower And Upper Limit
#https://alloyui.com/examples/color-picker/hsv.html



BLUE_LOW_LIMIT=np.array([90,0,0])
BLUE_HIGH_LIMIT=np.array([120,255,255]) 
GREEN_LOW_LIMIT=np.array([40,0,0])
GREEN_HIGH_LIMIT=np.array([70,255,255])
PINK_LOW_LIMIT=np.array([140,0,0])
PINK_HIGH_LIMIT=np.array([170,255,255])


def getGlovesImage(image, color):
  if(color == None):
    return
  low_limit = None
  high_limit = None
  if color == 'blue' :
    low_limit = BLUE_LOW_LIMIT
    high_limit = BLUE_HIGH_LIMIT
  elif color == 'green':
    low_limit = GREEN_LOW_LIMIT
    high_limit = GREEN_HIGH_LIMIT
  elif color == 'pink':
    low_limit = PINK_LOW_LIMIT
    high_limit = PINK_HIGH_LIMIT
    
 
  hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
  mask=cv2.inRange(hsv,low_limit,high_limit)
  return mask


def predict_image(result):
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

while True:
    ret, frame = cam.read()

    # flipping the frame to prevent inverted image of capturedframe...
    
    frame = cv2.flip(frame, 1)

    frame_copy = frame.copy()

    # ROI from the frame
    roi = frame[ROI_top:ROI_bottom, ROI_right:ROI_left]
    
    #Getting The Maked Gloves Only
    
    masked_image = getGlovesImage(roi, 'pink')
    
    cv2.rectangle(frame_copy, (ROI_left, ROI_top), (ROI_right,
    ROI_bottom), (255,128,0), 3)
    

    
    #Prediction
    
    resized_masked_image = cv2.resize(masked_image, (64, 64))
    resized_masked_image = cv2.cvtColor(resized_masked_image,cv2.COLOR_GRAY2RGB)
    reshaped_masked_image = np.reshape(resized_masked_image,(1,resized_masked_image.shape[0],resized_masked_image.shape[1],3))
    
    result = classifier.predict(reshaped_masked_image)

    cv2.putText(frame_copy, predict_image(result),(170, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    cv2.imshow("Masked Image", masked_image)
    cv2.imshow('Original Image', frame_copy)
    
    # Close windows with Esc
    k = cv2.waitKey(1) & 0xFF

    if k == 27:
        break

# Release the camera and destroy all the windows
cam.release()
cv2.destroyAllWindows()
