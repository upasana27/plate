import cv2 
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import pytesseract
img = cv2.imread('10.jpg', 1)
kernel =np.ones((3,23),np.uint8)
imgblur = cv2.GaussianBlur(img, (5,5), 0)
gray = cv2.cvtColor(imgblur, cv2.COLOR_BGR2GRAY)
sx = cv2.Sobel(gray,cv2.CV_8U,1,0,ksize=3)
ret2,th2 = cv2.threshold(sx,6,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
th2= cv2.dilate(th2,kernel,iterations = 1)
th2, contours, hierarchy = cv2.findContours(th2,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
maxarea=0.0
for c in contours:
    rect = cv2.boundingRect(c)
    x,y,w,h = rect
    area = cv2.contourArea(c)
    if w-100<h: continue 
    if area>maxarea:
	maxarea=area
	i=c
x1,y1,w1,h1 = cv2.boundingRect(i)
cv2.rectangle(img,(x1,y1),(x1+w1,y1+h1),(0,0,255),2)
#----------------for part detected rectangles-----------------------------------#
for cnt in contours:
    rect = cv2.boundingRect(cnt)
    x,y,w,h = rect
    if (y>=y1-5) and (y<=y1+5):
        if w>h:
           cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)


#----------------------using mser---------------------------#
roi=img[y1:y1+h1, x1:x1+w1]
roigray = cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY) 


vis = roi.copy() #detect regions in gray scale image
mser = cv2.MSER_create()
regions, _ = mser.detectRegions(roigray)
hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
mask = np.zeros(vis.shape, np.uint8)
#cv2.drawContours(roi,[hulls[1]],0,(0,0,255),2)

for h in hulls:
    
    [x,y,w,h] = cv2.boundingRect(h)
    area = w
    if area<2000 and w<h :
        cv2.rectangle(mask,(x,y),(x+w,y+h),(255,255,255),-1)
        cv2.rectangle(mask,(x,y),(x+w,y+h),(255,255,255),10)

kernel1 = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
kernel2= cv2.getStructuringElement(cv2.MORPH_CROSS,(3,5))
#dil= cv2.dilate(mask,kernel1,iterations =5)
dil= cv2.erode(mask,kernel2,iterations =2)
#dil= cv2.dilate(dil,kernel1,iterations =2)
dilgray=cv2.cvtColor(dil, cv2.COLOR_BGR2GRAY)
th, cont, hierarchy = cv2.findContours(dilgray,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
print cont
maxar=0.0
for c in cont:
    rect = cv2.boundingRect(c)
    x,y,w,h = rect
    area = w*h
    print area
    cv2.rectangle(dil,(x,y),(x+w,y+h),(0,0,255),1)
    if area>maxar:
	maxar=area
	j=c
        
rect2 = cv2.boundingRect(j)
x2,y2,w2,h2 = rect2
cv2.rectangle(dil,(x2,y2),(x2+w2,y2+h2),(255,0,0),2)
xfinal=x2
ffinal=x2+w2

for c in cont:
    rect = cv2.boundingRect(c)
    x,y,w,h = rect
    f=x+w
    if y>=y2-1 and y<=y2+1:
       print 0
       cv2.rectangle(dil,(x,y),(x+w,y+h),(255,0,0),2)
       xfinal=min(xfinal,x)
       ffinal=max(ffinal,f)
       
#print xfinal,wfinal
cv2.rectangle(dil,(xfinal,y2),(ffinal,y2+h2),(0,0,255),2)
roifinal=roi[y2:y2+h2, xfinal:ffinal]

text = pytesseract.image_to_string(roifinal)
print(text)
   
cv2.imshow("text only",img)

#cv2.imshow('img',vis)
cv2.waitKey(0)
cv2.destroyAllWindows()

