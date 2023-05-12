import cv2
import pandas as pd
import numpy as np
img_path = "C:/Users/Tushar/Downloads/colorimage.jpeg"
img = cv2.imread(img_path)

index = ["color", "color_name", "hex", "R", "G", "B"]
csv = pd.read_csv('C:/Users/Tushar/OneDrive/Documents/Colors.csv', names=index, header=None)

def getColorName(R,G,B):
    minimum = 10000
    for i in range(len(csv)):
        d = abs(R- int(csv.loc[i,"R"])) + abs(G- int(csv.loc[i,"G"]))+ abs(B- int(csv.loc[i,"B"]))
        if(d<=minimum):
            minimum = d
            color_name = csv.loc[i,"color_name"]
    return color_name

clicked = False
r = g = b = xpos = ypos = 0

def mouse_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        global b,g,r,xpos,ypos, clicked
        clicked = True
        xpos = x
        ypos = y
        b,g,r = img[y,x]
        b = int(b)
        g = int(g)
        r = int(r)

cv2.namedWindow('Color Detection App')
cv2.setMouseCallback('Color Detection App', mouse_click)

while(1):
    cv2.imshow("Color Detection App",img)
    if (clicked):
        #cv2.rectangle(image, startpoint, endpoint, color, thickness) -1 thickness fills rectangle entirely
        cv2.rectangle(img,(20,20), (750,60), (b,g,r), -1)

        #Creating text string to display ( Color name and RGB values )
        color_name = getColorName(r,g,b) + ' R='+ str(r) + ' G='+ str(g) + ' B='+ str(b)

        #cv2.putText(img,text,start,font(0-7), fontScale, color, thickness, lineType, (optional bottomLeft bool) )
        cv2.putText(img, color_name,(50,50),2,0.8,(255,255,255),2,cv2.LINE_AA)
  #For very light colours we will display text in black colour
        if(r+g+b>=600):
            cv2.putText(img, color_name,(50,50),2,0.8,(0,0,0),2,cv2.LINE_AA)

        clicked=False

    #Break the loop when user hits 'esc' key 
    if cv2.waitKey(20) & 0xFF ==27:
        break

cv2.destroyAllWindows()

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
color = np.uint8([[[b, g, r]]])
hsv_color = cv2.cvtColor(color,cv2.COLOR_BGR2HSV)
lower_range = np.array([hsv_color[0][0][0] - 10, 100, 100])
upper_range = np.array([hsv_color[0][0][0] + 10, 255, 255])

mask = cv2.inRange(hsv, lower_range, upper_range)

cv2.imshow('mask', mask)
cv2.waitKey(0)
cv2.destroyAllWindows()
