import cv2
import numpy as np
import RPi.GPIO as GPIO
import os
import time
import DetectChars
import DetectPlates
import PossiblePlate
import urllib.request
import pyrebase
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)

SCALAR_BLACK = (0.0, 0.0, 0.0)
SCALAR_WHITE = (255.0, 255.0, 255.0)
SCALAR_YELLOW = (0.0, 255.0, 255.0)
SCALAR_GREEN = (0.0, 255.0, 0.0)
SCALAR_RED = (0.0, 0.0, 255.0)


showSteps = False

camera = cv2.VideoCapture(0)
dis = int()
vechile_number = []
x=" "
def main():
    
    def  ultra():
        TRIG = 2
        ECHO = 3
        GPIO.setwarnings(False)
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(TRIG,GPIO.OUT)
        GPIO.setup(ECHO,GPIO.IN)
        
        while(True):
            GPIO.output(TRIG,False)
            time.sleep(2)

            GPIO.output(TRIG,True)
            time.sleep(0.00001)
            GPIO.output(TRIG,False)

            while(GPIO.input(ECHO)==0):
                start= time.time()

            while(GPIO.input(ECHO)==1):
                end= time.time()

            pulse_duration= end-start

            distance= pulse_duration*17150
            dis= round(distance,2)
            
            print("distance is ",dis)
            break
            

    def servo():
        mot1 = 18
        GPIO.setwarnings(False)
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(mot1,GPIO.OUT)
        p= GPIO.PWM(mot1,50)
        p.start(7.5)
        p.ChangeDutyCycle(7.5)
        time.sleep(5)
        p.ChangeDutyCycle(2.5)
        time.sleep(1)
        GPIO.cleanup()
        
    def fire():
        import pyrebase
        config = {
           "apiKey": "AIzaSyCyjVHHvDlkW_OcGjRub-nZTH5G8yWTCS8",
    "authDomain": "registration-ed760.firebaseapp.com",
    "databaseURL": "https://registration-ed760.firebaseio.com",
    "projectId": "registration-ed760",
    "storageBucket": "registration-ed760.appspot.com",
    "messagingSenderId": "302826735996"
        };

        firebase=pyrebase.initialize_app(config);
        vechile_number=[]
        db=firebase.database()
        a=0
        while(1):
            
            users = db.child("carin").get()
            temp=dict(users.val())
            key_data=list(users.val())
          
            if(len(key_data)!=a):
                
                for i in range(len(key_data)):
                    vechile_number.insert(i,temp[key_data[i]]['carvechile'])
                    
                
                for j in range(len(key_data)):
                    print(vechile_number[j])
                print("-----------------------------------------------")
                a=len(key_data)     
            if(x in vechile_number):
                servo()
                break
            else:
                  car_driver=input("Driver name:")
                  car_mobile=input("Mobile number:")
                  car_vechile=x
                  data={'cardriver':car_driver,'carmobile':car_mobile,'carvechile':car_vechile}
                  db.child("carin").push(data)
                  print("edit successfully")
                  
                
    blnKNNTrainingSuccessful = DetectChars.loadKNNDataAndTrainKNN()      

    if blnKNNTrainingSuccessful == False:                              
        print ("\nerror: KNN traning was not successful\n")              
        return

    ultra()
    if(dis<50):
        
        res=urllib.request.urlopen('http://192.168.43.185:8080//shot.jpg')
        data=np.array(bytearray(res.read()),dtype=np.uint8)
        
        imgOriginalScene=cv2.imdecode(data,-1)
     
        if imgOriginalScene is None:                        
            print ("\nerror: image not read from file \n\n")   
            os.system("pause")                                 
            return                                          

        listOfPossiblePlates = DetectPlates.detectPlatesInScene(imgOriginalScene)          

        listOfPossiblePlates = DetectChars.detectCharsInPlates(listOfPossiblePlates)      

        cv2.imshow("imgOriginalScene",imgOriginalScene)            

        if len(listOfPossiblePlates) == 0:                        
            print ("\nno license plates were detected\n")            
        else:                                                    
                  
            listOfPossiblePlates.sort(key = lambda possiblePlate: len(possiblePlate.strChars), reverse = True)

                   
            licPlate = listOfPossiblePlates[0]

            cv2.imshow("imgPlate", licPlate.imgPlate)         
            cv2.imshow("imgThresh", licPlate.imgThresh)

            if len(licPlate.strChars) == 0:                   
                print ("\nno characters were detected\n\n")     
                return                                      
       

            drawRedRectangleAroundPlate(imgOriginalScene, licPlate)            
            
            x = licPlate.strChars
            print("license plate is" + x + "\n")

            fire()    

            writeLicensePlateCharsOnImage(imgOriginalScene, licPlate)         

            cv2.imshow("imgOriginalScene", imgOriginalScene)              

            cv2.imwrite("imgOriginalScene.png", imgOriginalScene)       

       

        #cv2.waitKey(0)					
        cv2.destroyAllWindows()
        return
# end main

def drawRedRectangleAroundPlate(imgOriginalScene, licPlate):

    p2fRectPoints = cv2.boxPoints(licPlate.rrLocationOfPlateInScene)           

    cv2.line(imgOriginalScene, tuple(p2fRectPoints[0]), tuple(p2fRectPoints[1]), SCALAR_RED, 2)        
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[1]), tuple(p2fRectPoints[2]), SCALAR_RED, 2)
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[2]), tuple(p2fRectPoints[3]), SCALAR_RED, 2)
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[3]), tuple(p2fRectPoints[0]), SCALAR_RED, 2)

def writeLicensePlateCharsOnImage(imgOriginalScene, licPlate):
    ptCenterOfTextAreaX = 0                           
    ptCenterOfTextAreaY = 0

    ptLowerLeftTextOriginX = 0                         
    ptLowerLeftTextOriginY = 0

    sceneHeight, sceneWidth, sceneNumChannels = imgOriginalScene.shape
    plateHeight, plateWidth, plateNumChannels = licPlate.imgPlate.shape

    intFontFace = cv2.FONT_HERSHEY_SIMPLEX                     
    fltFontScale = float(plateHeight) / 30.0                 
    intFontThickness = int(round(fltFontScale * 1.5))          

    textSize, baseline = cv2.getTextSize(licPlate.strChars, intFontFace, fltFontScale, intFontThickness)        

         
    ( (intPlateCenterX, intPlateCenterY), (intPlateWidth, intPlateHeight), fltCorrectionAngleInDeg ) = licPlate.rrLocationOfPlateInScene

    intPlateCenterX = int(intPlateCenterX)             
    intPlateCenterY = int(intPlateCenterY)

    ptCenterOfTextAreaX = int(intPlateCenterX)        

    if intPlateCenterY < (sceneHeight * 0.75):                                                
        ptCenterOfTextAreaY = int(round(intPlateCenterY)) + int(round(plateHeight * 1.6))     
    else:                                                                                      
        ptCenterOfTextAreaY = int(round(intPlateCenterY)) - int(round(plateHeight * 1.6))      
    

    textSizeWidth, textSizeHeight = textSize             
    ptLowerLeftTextOriginX = int(ptCenterOfTextAreaX - (textSizeWidth / 2))        
    ptLowerLeftTextOriginY = int(ptCenterOfTextAreaY + (textSizeHeight / 2))         
            
    cv2.putText(imgOriginalScene, licPlate.strChars, (ptLowerLeftTextOriginX, ptLowerLeftTextOriginY), intFontFace, fltFontScale, SCALAR_YELLOW, intFontThickness)



if __name__ == "__main__":
    while 1:
        main()
