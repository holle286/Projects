import numpy as np
from PIL import ImageGrab
import pandas as pd
import cv2
import time
import subprocess
import pyautogui
import pytesseract
from keyboardKeys import PressKey, ReleaseKey, Up, Down, Left, Right, Enter, X, C, Z, L

teal = [0, 107, 107, 255]
blue = [0, 0, 144, 255]
yellow = [206, 186, 4, 255]
green = [0, 90, 0, 255]
red = [90, 0, 0, 255]
purple = [79, 8, 142, 255]

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


def process_img(image):
    original_image = image
    # processed_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image = cv2.Canny(image, threshold1=200, threshold2=300)
    return image

# src = r'C:\Users\n1ck1\Desktop\Projects\Pokemon\level.png'
# imageFrame = cv2.imread(src)
# hsvFrame = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2HSV)

def main():
    start_game()

    while True:
        imageFrame =  np.array(ImageGrab.grab(bbox=(20,40,660,530)))
        scoreFrame = np.array(ImageGrab.grab(bbox=(478,180,593,215)))
        hsvScore = cv2.cvtColor(scoreFrame, cv2.COLOR_RGB2HSV)
        levelFrame = np.array(ImageGrab.grab(bbox=(478,236,593,268)))
        hsvLevel = cv2.cvtColor(levelFrame, cv2.COLOR_RGB2HSV)
        score_msk = cv2.inRange(hsvScore, (99, 100, 200), (110, 235, 255))
        level_msk = cv2.inRange(hsvLevel, (99, 100, 200), (110, 235, 255))
        krn = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
        score_dlt = cv2.dilate(score_msk, krn, iterations=1)
        score_thr = 255 - cv2.bitwise_and(score_dlt, score_msk)
        score = pytesseract.image_to_string(score_thr, config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789')
        level_dlt = cv2.dilate(level_msk, krn, iterations=1)
        level_thr = 255 - cv2.bitwise_and(level_dlt, level_msk)
        level = pytesseract.image_to_string(level_thr, config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789')


        # imageFrame =  np.array(ImageGrab.grab(bbox=(250,120,465,503)))
        #print('Frame took {} seconds'.format(time.time()-last_time))
        #cv2.imshow('window', new_screen)
        hsvFrame = cv2.cvtColor(imageFrame, cv2.COLOR_RGB2HSV)


        # red_lower = np.array([0, 100, 100], np.uint8)
        # red_upper = np.array([2, 255, 255], np.uint8)
        # red_mask = cv2.inRange(hsvFrame, red_lower, red_upper)
    
        # # Set range for green color and 
        # # define mask
        # green_lower = np.array([59, 100, 100], np.uint8)
        # green_upper = np.array([60, 255, 255], np.uint8)
        # green_mask = cv2.inRange(hsvFrame, green_lower, green_upper)
    
        # # Set range for blue color and
        # # define mask
        # blue_lower = np.array([118, 100, 100], np.uint8)
        # blue_upper = np.array([122, 255, 255], np.uint8)
        # blue_mask = cv2.inRange(hsvFrame, blue_lower, blue_upper)

        # # Set range for teal color and 
        # # define mask
        # teal_lower = np.array([88, 100, 100], np.uint8)
        # teal_upper = np.array([92,255,255], np.uint8)
        # teal_mask = cv2.inRange(hsvFrame, teal_lower, teal_upper)

        # # Set range for yellow color and 
        # # define mask
        # yellow_lower = np.array([26, 100, 100], np.uint8)
        # yellow_upper = np.array([28, 255, 255], np.uint8)
        # yellow_mask = cv2.inRange(hsvFrame, yellow_lower, yellow_upper)

        # # Set range for purple color and 
        # # define mask
        # purple_lower = np.array([134, 100, 100], np.uint8)
        # purple_upper = np.array([138, 255, 255], np.uint8)
        # purple_mask = cv2.inRange(hsvFrame, purple_lower, purple_upper)
        
        
        # # Morphological Transform, Dilation
        # # for each color and bitwise_and operator
        # # between imageFrame and mask determines
        # # to detect only that particular color
        # kernel = np.ones((5, 5), "uint8")
        
        # # For red color
        # red_mask = cv2.dilate(red_mask, kernel)
        # res_red = cv2.bitwise_and(imageFrame, imageFrame, 
        #                         mask = red_mask)
        
        # # For green color
        # green_mask = cv2.dilate(green_mask, kernel)
        # res_green = cv2.bitwise_and(imageFrame, imageFrame,
        #                             mask = green_mask)
        
        # # For blue color
        # blue_mask = cv2.dilate(blue_mask, kernel)
        # res_blue = cv2.bitwise_and(imageFrame, imageFrame,
        #                         mask = blue_mask)
        
        # # For teal color
        # teal_mask = cv2.dilate(teal_mask, kernel)
        # res_teal = cv2.bitwise_and(imageFrame, imageFrame,
        #                             mask = teal_mask)
        
        # # For yellow color
        # yellow_mask = cv2.dilate(yellow_mask, kernel)
        # res_yellow = cv2.bitwise_and(imageFrame, imageFrame,
        #                                 mask = yellow_mask)
        
        # # For purple color
        # purple_mask = cv2.dilate(purple_mask, kernel)
        # res_purple = cv2.bitwise_and(imageFrame, imageFrame,
        #                                 mask = purple_mask)
    
        # # Creating contour to track red color
        # contours, hierarchy = cv2.findContours(red_mask,
        #                                     cv2.RETR_TREE,
        #                                     cv2.CHAIN_APPROX_SIMPLE)
        
        # for pic, contour in enumerate(contours):
        #     area = cv2.contourArea(contour)
        #     if(area > 300):
        #         x, y, w, h = cv2.boundingRect(contour)
        #         imageFrame = cv2.rectangle(imageFrame, (x, y), 
        #                                 (x + w, y + h), 
        #                                 (90, 0, 0), 2)
                
        #         cv2.putText(imageFrame, "R", (x, y),
        #                     cv2.FONT_HERSHEY_SIMPLEX, 1.0,
        #                     (90, 0, 0))    
    
        # # Creating contour to track green color
        # contours, hierarchy = cv2.findContours(green_mask,
        #                                     cv2.RETR_TREE,
        #                                     cv2.CHAIN_APPROX_SIMPLE)
        
        # for pic, contour in enumerate(contours):
        #     area = cv2.contourArea(contour)
        #     if(area > 300):
        #         x, y, w, h = cv2.boundingRect(contour)
        #         imageFrame = cv2.rectangle(imageFrame, (x, y), 
        #                                 (x + w, y + h),
        #                                 (0, 90, 0), 2)
                
        #         cv2.putText(imageFrame, "G", (x, y),
        #                     cv2.FONT_HERSHEY_SIMPLEX, 
        #                     1.0, (0, 90, 0))
    
        # # Creating contour to track blue color
        # contours, hierarchy = cv2.findContours(blue_mask,
        #                                     cv2.RETR_TREE,
        #                                     cv2.CHAIN_APPROX_SIMPLE)
        # for pic, contour in enumerate(contours):
        #     area = cv2.contourArea(contour)
        #     if(area > 300):
        #         x, y, w, h = cv2.boundingRect(contour)
        #         imageFrame = cv2.rectangle(imageFrame, (x, y),
        #                                 (x + w, y + h),
        #                                 (0, 0, 144), 2)
                
        #         cv2.putText(imageFrame, "B", (x, y),
        #                     cv2.FONT_HERSHEY_SIMPLEX,
        #                     1.0, (0, 0, 144))
                
        # # Creating contour to track teal color
        # contours, hierarchy = cv2.findContours(teal_mask,
        #                                     cv2.RETR_TREE,
        #                                     cv2.CHAIN_APPROX_SIMPLE)
        
        # for pic, contour in enumerate(contours):
        #     area = cv2.contourArea(contour)
        #     if(area > 300):
        #         x, y, w, h = cv2.boundingRect(contour)
        #         imageFrame = cv2.rectangle(imageFrame, (x, y), 
        #                                 (x + w, y + h), 
        #                                 (0, 107, 107), 2)
                
        #         cv2.putText(imageFrame, "T", (x, y),
        #                     cv2.FONT_HERSHEY_SIMPLEX, 1.0,
        #                     (0, 107, 107))
                
        # # Creating contour to track yellow color
        # contours, hierarchy = cv2.findContours(yellow_mask,
        #                                     cv2.RETR_TREE,
        #                                     cv2.CHAIN_APPROX_SIMPLE)
        
        # for pic, contour in enumerate(contours):
        #     area = cv2.contourArea(contour)
        #     if(area > 300):
        #         x, y, w, h = cv2.boundingRect(contour)
        #         imageFrame = cv2.rectangle(imageFrame, (x, y), 
        #                                 (x + w, y + h), 
        #                                 (206, 186, 4), 2)
                
        #         cv2.putText(imageFrame, "Y", (x, y),
        #                     cv2.FONT_HERSHEY_SIMPLEX, 1.0,
        #                     (206, 186, 4)) 
                
        # # Creating contour to track purple color
        # contours, hierarchy = cv2.findContours(purple_mask,
        #                                     cv2.RETR_TREE,
        #                                     cv2.CHAIN_APPROX_SIMPLE)
        
        # for pic, contour in enumerate(contours):
        #     area = cv2.contourArea(contour)
        #     if(area > 300):
        #         x, y, w, h = cv2.boundingRect(contour)
        #         imageFrame = cv2.rectangle(imageFrame, (x, y), 
        #                                 (x + w, y + h), 
        #                                 (79, 8, 142), 2)
                
        #         cv2.putText(imageFrame, "P", (x, y),
        #                     cv2.FONT_HERSHEY_SIMPLEX, 1.0,
        #                     (79, 8, 142))    


                
        cv2.imshow('window',cv2.cvtColor(imageFrame, cv2.COLOR_BGR2RGB))
        cv2.imshow('window2',cv2.cvtColor(scoreFrame, cv2.COLOR_BGR2RGB))
        cv2.imshow('window3',cv2.cvtColor(levelFrame, cv2.COLOR_BGR2RGB))
        print("Score: ", score)
        print("Level: ", level)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break


  
# Start a while loop
# while(1):
#     # _, imageFrame = webcam.read()
  
#     # # Convert the imageFrame in 
#     # # BGR(RGB color space) to 
#     # # HSV(hue-saturation-value)
#     # # color space
#     # hsvFrame = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2HSV)
#     # Set range for red color and 
#     # define mask
#     red_lower = np.array([0, 100, 100], np.uint8)
#     red_upper = np.array([10, 255, 255], np.uint8)
#     red_mask = cv2.inRange(hsvFrame, red_lower, red_upper)
  
#     # Set range for green color and 
#     # define mask
#     green_lower = np.array([50, 100, 100], np.uint8)
#     green_upper = np.array([70, 255, 255], np.uint8)
#     green_mask = cv2.inRange(hsvFrame, green_lower, green_upper)
  
#     # Set range for blue color and
#     # define mask
#     blue_lower = np.array([110, 100, 100], np.uint8)
#     blue_upper = np.array([128, 255, 255], np.uint8)
#     blue_mask = cv2.inRange(hsvFrame, blue_lower, blue_upper)

#     # Set range for teal color and 
#     # define mask
#     teal_lower = np.array([80, 100, 100], np.uint8)
#     teal_upper = np.array([100,255,255], np.uint8)
#     teal_mask = cv2.inRange(hsvFrame, teal_lower, teal_upper)

#     # Set range for yellow color and 
#     # define mask
#     yellow_lower = np.array([17, 100, 100], np.uint8)
#     yellow_upper = np.array([37, 255, 255], np.uint8)
#     yellow_mask = cv2.inRange(hsvFrame, yellow_lower, yellow_upper)

#     # Set range for purple color and 
#     # define mask
#     purple_lower = np.array([129, 100, 100], np.uint8)
#     purple_upper = np.array([146, 255, 255], np.uint8)
#     purple_mask = cv2.inRange(hsvFrame, purple_lower, purple_upper)
    
      
#     # Morphological Transform, Dilation
#     # for each color and bitwise_and operator
#     # between imageFrame and mask determines
#     # to detect only that particular color
#     kernel = np.ones((5, 5), "uint8")
      
#     # For red color
#     red_mask = cv2.dilate(red_mask, kernel)
#     res_red = cv2.bitwise_and(imageFrame, imageFrame, 
#                               mask = red_mask)
      
#     # For green color
#     green_mask = cv2.dilate(green_mask, kernel)
#     res_green = cv2.bitwise_and(imageFrame, imageFrame,
#                                 mask = green_mask)
      
#     # For blue color
#     blue_mask = cv2.dilate(blue_mask, kernel)
#     res_blue = cv2.bitwise_and(imageFrame, imageFrame,
#                                mask = blue_mask)
    
#     # For teal color
#     teal_mask = cv2.dilate(teal_mask, kernel)
#     res_teal = cv2.bitwise_and(imageFrame, imageFrame,
#                                  mask = teal_mask)
    
#     # For yellow color
#     yellow_mask = cv2.dilate(yellow_mask, kernel)
#     res_yellow = cv2.bitwise_and(imageFrame, imageFrame,
#                                     mask = yellow_mask)
    
#     # For purple color
#     purple_mask = cv2.dilate(purple_mask, kernel)
#     res_purple = cv2.bitwise_and(imageFrame, imageFrame,
#                                     mask = purple_mask)
   
#     # Creating contour to track red color
#     contours, hierarchy = cv2.findContours(red_mask,
#                                            cv2.RETR_TREE,
#                                            cv2.CHAIN_APPROX_SIMPLE)
      
#     for pic, contour in enumerate(contours):
#         area = cv2.contourArea(contour)
#         if(area > 10):
#             x, y, w, h = cv2.boundingRect(contour)
#             imageFrame = cv2.rectangle(imageFrame, (x, y), 
#                                        (x + w, y + h), 
#                                        (0, 0, 90), 2)
              
#             cv2.putText(imageFrame, "Red Colour", (x, y),
#                         cv2.FONT_HERSHEY_SIMPLEX, 1.0,
#                         (0, 0, 90))    
#             print("FOUND RED")
  
#     # Creating contour to track green color
#     contours, hierarchy = cv2.findContours(green_mask,
#                                            cv2.RETR_TREE,
#                                            cv2.CHAIN_APPROX_SIMPLE)
      
#     for pic, contour in enumerate(contours):
#         area = cv2.contourArea(contour)
#         if(area > 10):
#             x, y, w, h = cv2.boundingRect(contour)
#             imageFrame = cv2.rectangle(imageFrame, (x, y), 
#                                        (x + w, y + h),
#                                        (0, 90, 0), 2)
              
#             cv2.putText(imageFrame, "Green Colour", (x, y),
#                         cv2.FONT_HERSHEY_SIMPLEX, 
#                         1.0, (0, 90, 0))
#             print("FOUND GREEN")
  
#     # Creating contour to track blue color
#     contours, hierarchy = cv2.findContours(blue_mask,
#                                            cv2.RETR_TREE,
#                                            cv2.CHAIN_APPROX_SIMPLE)
#     for pic, contour in enumerate(contours):
#         area = cv2.contourArea(contour)
#         if(area > 10):
#             x, y, w, h = cv2.boundingRect(contour)
#             imageFrame = cv2.rectangle(imageFrame, (x, y),
#                                        (x + w, y + h),
#                                        (144, 0, 0), 2)
              
#             cv2.putText(imageFrame, "Blue Colour", (x, y),
#                         cv2.FONT_HERSHEY_SIMPLEX,
#                         1.0, (144, 0, 0))
#             print("FOUND BLUE")  
            
#     # Creating contour to track teal color
#     contours, hierarchy = cv2.findContours(teal_mask,
#                                            cv2.RETR_TREE,
#                                            cv2.CHAIN_APPROX_SIMPLE)
      
#     for pic, contour in enumerate(contours):
#         area = cv2.contourArea(contour)
#         if(area > 50):
#             x, y, w, h = cv2.boundingRect(contour)
#             imageFrame = cv2.rectangle(imageFrame, (x, y), 
#                                        (x + w, y + h), 
#                                        (107, 107, 0), 8)
              
#             cv2.putText(imageFrame, "Teal Colour", (x, y),
#                         cv2.FONT_HERSHEY_SIMPLEX, 1.0,
#                         (107, 107, 0))
#             print("FOUND TEAL")       
            
#     # Creating contour to track yellow color
#     contours, hierarchy = cv2.findContours(yellow_mask,
#                                            cv2.RETR_TREE,
#                                            cv2.CHAIN_APPROX_SIMPLE)
      
#     for pic, contour in enumerate(contours):
#         area = cv2.contourArea(contour)
#         if(area > 10):
#             x, y, w, h = cv2.boundingRect(contour)
#             imageFrame = cv2.rectangle(imageFrame, (x, y), 
#                                        (x + w, y + h), 
#                                        (206, 186, 4), 2)
              
#             cv2.putText(imageFrame, "Yellow Colour", (x, y),
#                         cv2.FONT_HERSHEY_SIMPLEX, 1.0,
#                         (206, 186, 4)) 
#             print("FOUND YELLOW")
            
#     # Creating contour to track purple color
#     contours, hierarchy = cv2.findContours(purple_mask,
#                                            cv2.RETR_TREE,
#                                            cv2.CHAIN_APPROX_SIMPLE)
      
#     for pic, contour in enumerate(contours):
#         area = cv2.contourArea(contour)
#         if(area > 10):
#             x, y, w, h = cv2.boundingRect(contour)
#             imageFrame = cv2.rectangle(imageFrame, (x, y), 
#                                        (x + w, y + h), 
#                                        (142, 8, 79), 2)
              
#             cv2.putText(imageFrame, "Purple Colour", (x, y),
#                         cv2.FONT_HERSHEY_SIMPLEX, 1.0,
#                         (142, 8, 79))    
#             print("FOUND PURPLE")
            
#     cv2.imshow('window',cv2.cvtColor(hsvFrame, cv2.COLOR_HSV2BGR))
#     if cv2.waitKey(25) & 0xFF == ord('q'):
#         cv2.destroyAllWindows()
#         break       
            
              
    # Program Termination
    # cv2.imshow("Multiple Color Detection in Real-TIme", imageFrame)
    # if cv2.waitKey(10) & 0xFF == ord('q'):
    #     cap.release()
    #     cv2.destroyAllWindows()
    #     break




# def main():
#     start_game()
#     # for i in list(range(4))[::-1]:
#     #     PressKey(L)
#     #     time.sleep(5)
#     #     ReleaseKey(L)
#     #     time.sleep(5)
#     #     restart_game()

#     last_time = time.time()
#     while True:
#         screen =  np.array(ImageGrab.grab(bbox=(20,60,660,540)))
#         #print('Frame took {} seconds'.format(time.time()-last_time))
#         last_time = time.time()
#         new_screen = process_img(screen)
#         #cv2.imshow('window', new_screen)
#         cv2.imshow('window',cv2.cvtColor(screen, cv2.COLOR_BGR2RGB))
#         if cv2.waitKey(25) & 0xFF == ord('q'):
#             cv2.destroyAllWindows()
#             break

def restart_game():
    print('restarting game')
    PressKey(X)
    ReleaseKey(X)
    time.sleep(2)
    PressKey(X)
    ReleaseKey(X)
    time.sleep(.3)
    PressKey(X)
    ReleaseKey(X)
    time.sleep(.3)
    PressKey(X)
    ReleaseKey(X)
    time.sleep(8.5)

def start_game():
    subprocess.Popen(['C:\\n64\Project64 3.0\Project64.exe'])
    time.sleep(1)
    pyautogui.keyDown('enter')
    pyautogui.keyUp('enter')
    time.sleep(10)
    print('starting game')
    PressKey(Enter)
    ReleaseKey(Enter)
    time.sleep(1)
    print('skipping intro')
    PressKey(Enter)
    ReleaseKey(Enter)
    time.sleep(2.5)
    print('navigating menu')
    PressKey(Left)
    ReleaseKey(Left)
    time.sleep(.3)
    PressKey(Left)
    ReleaseKey(Left)
    time.sleep(.3)
    PressKey(Left)
    ReleaseKey(Left)
    time.sleep(.3)
    print('selecting game mode')
    PressKey(X)
    ReleaseKey(X)
    time.sleep(2)
    PressKey(Right)
    ReleaseKey(Right)
    time.sleep(.3)
    PressKey(X)
    ReleaseKey(X)
    time.sleep(1.5)
    print('selecting difficulty')
    PressKey(X)
    ReleaseKey(X)
    time.sleep(.3)
    PressKey(X)
    ReleaseKey(X)
    time.sleep(.3)
    PressKey(X)
    ReleaseKey(X)
    print('entering game')
    time.sleep(7)






# # def main():
# #     for i in list(range(4))[::-1]:
# #             print(i+1)
# #             time.sleep(1)
# #     PressKey(Enter)
# #     ReleaseKey(Enter)
# #     PressKey(Down)
# #     time.sleep(1)
# #     ReleaseKey(Down)
# #     PressKey(X)
# #     ReleaseKey(X)
# #     PressKey(Enter)
# #     ReleaseKey(Enter)

if __name__ == '__main__':
    main()