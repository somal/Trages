import numpy as np
import cv2
import cv
#import os
from multiprocessing.pool import ThreadPool
from numpy.linalg import norm
from Tkinter import *
from time import clock
from math import*
from time import *
import random


def hog(test):
    samples=[]
    for img in test:
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
        mag, ang = cv2.cartToPolar(gx, gy)  

        bins=np.int32(16*ang
                      /(2*np.pi))     
                                       
        bin_cells=bins[:10,:10],bins[10:-40,:10],bins[20:-30,:10],bins[30:-20,:10],bins[40:-10,:10],bins[50:,:10],bins[:10,10:-70],bins[10:-40,10:-70],bins[20:-30,10:-70],bins[30:-20,10:-70],bins[40:-10,10:-70],bins[50:,10:-70],bins[:10,20:-60],bins[10:-40,20:-60],bins[20:-30,20:-60],bins[30:-20,20:-60],bins[40:-10,20:-60],bins[50:,20:-60],bins[:10,30:-50],bins[10:-40,30:-50],bins[20:-30,30:-50],bins[30:-20,30:-50],bins[40:-10,30:-50],bins[50:,30:-50],bins[:10,40:-40],bins[10:-40,40:-40],bins[20:-30,40:-40],bins[30:-20,40:-40],bins[40:-10,40:-40],bins[50:,40:-40],bins[:10,50:-30],bins[10:-40,50:-30],bins[20:-30,50:-30],bins[30:-20,50:-30],bins[40:-10,50:-30],bins[50:,50:-30],bins[:10,60:-20],bins[10:-40,60:-20],bins[20:-30,60:-20],bins[30:-20,60:-20],bins[40:-10,60:-20],bins[50:,60:-20],bins[:10,70:-10],bins[10:-40,70:-10],bins[20:-30,70:-10],bins[30:-20,70:-10],bins[40:-10,70:-10],bins[50:,70:-10],bins[:10,80:],bins[10:-40,80:],bins[20:-30,80:],bins[30:-20,80:],bins[40:-10,80:],bins[50:,80:]
        mag_cells=mag[:10,:10],mag[10:-40,:10],mag[20:-30,:10],mag[30:-20,:10],mag[40:-10,:10],mag[50:,:10],mag[:10,10:-70],mag[10:-40,10:-70],mag[20:-30,10:-70],mag[30:-20,10:-70],mag[40:-10,10:-70],mag[50:,10:-70],mag[:10,20:-60],mag[10:-40,20:-60],mag[20:-30,20:-60],mag[30:-20,20:-60],mag[40:-10,20:-60],mag[50:,20:-60],mag[:10,30:-50],mag[10:-40,30:-50],mag[20:-30,30:-50],mag[30:-20,30:-50],mag[40:-10,30:-50],mag[50:,30:-50],mag[:10,40:-40],mag[10:-40,40:-40],mag[20:-30,40:-40],mag[30:-20,40:-40],mag[40:-10,40:-40],mag[50:,40:-40],mag[:10,50:-30],mag[10:-40,50:-30],mag[20:-30,50:-30],mag[30:-20,50:-30],mag[40:-10,50:-30],mag[50:,50:-30],mag[:10,60:-20],mag[10:-40,60:-20],mag[20:-30,60:-20],mag[30:-20,60:-20],mag[40:-10,60:-20],mag[50:,60:-20],mag[:10,70:-10],mag[10:-40,70:-10],mag[20:-30,70:-10],mag[30:-20,70:-10],mag[40:-10,70:-10],mag[50:,70:-10],mag[:10,80:],mag[10:-40,80:],mag[20:-30,80:],mag[30:-20,80:],mag[40:-10,80:],mag[50:,80:]      
        hists = [np.bincount(b.ravel(), m.ravel(), 16) for b, m in zip(bin_cells, mag_cells)]
        hist = np.hstack(hists)
        eps = 1e-7 
        hist /= hist.sum() + eps
        hist = np.sqrt(hist)
        hist /= norm(hist) + eps

        samples.append(hist)
    return np.float32(samples)        

class StatModel(object):
    def load(self, fn):
        self.model.load(fn)
    def save(self, fn):
        self.model.save(fn)

class SVM(StatModel):
    def __init__(self,C=1,gamma=0.5):
        self.params=dict( kernel_type = cv2.SVM_RBF, 
                            svm_type = cv2.SVM_C_SVC,
                            C = C,
                            gamma = gamma )
        self.model=cv2.SVM()
    def train(self,samples,responses):
        self.model=cv2.SVM()
        self.model.train(samples,responses,params=self.params)

    def predict(self,samples):
        return self.model.predict_all(samples).ravel()

def moment(cnt):
    moments = cv2.moments(cnt) # Calculate moments
    cx=cy=0
    if moments['m00']!=0:
        cx = int(moments['m10']/moments['m00']) # cx = M10/M00
        cy = int(moments['m01']/moments['m00'])
        
    return cx,cy

def deskew(img):
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11']/m['mu02']
    M = np.float32([[1, skew, -0.5*SZY*skew], [0, 1, 0]])
    img = cv2.warpAffine(img, M, (SZX, SZY), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    return img
    
def draw_str(dst, (x, y), s):
    cv2.putText(dst, s, (x+1, y+1), cv2.FONT_HERSHEY_PLAIN, 5.0, (0, 255, 255), thickness = 5)   #, lineType=cv2.CV_AA
    cv2.putText(dst, s, (x, y), cv2.FONT_HERSHEY_PLAIN, 5.0, (0, 0, 255), thickness = 3)    

def draw_run_str(dst, (x, y), s):
    cv2.putText(dst, s, (x+1, y+1), cv2.FONT_HERSHEY_PLAIN, 4.0, (0, 255, 255), thickness = 5)   #, lineType=cv2.CV_AA
    cv2.putText(dst, s, (x, y), cv2.FONT_HERSHEY_PLAIN, 4.0, (0, 0, 255), thickness = 3)    


def main():
    '''cv2.namedWindow("animals:D")
    alph=cv2.imread('alphabet.jpg')
    alph=cv2.resize(alph,(648,381))
    cv2.imshow("animals:D",alph)
    alphges=cv2.imread('alphages.png')
    alphges=cv2.resize(alphges,(542,660))
    cv2.imshow("Gestures",alphges)
    shlet=cv2.imread('lets-go.jpg')
    shlet=cv2.resize(shlet,(320,240))
    cv2.imshow("Letters",shlet)'''
    import os
    classifier_fn = 'recognition.dat'
    if not os.path.exists(classifier_fn):
        print '"%s" not found' % classifier_fn
        return

    model = SVM()
    model.load('recognition.dat')

    cap=cv2.VideoCapture(0)
    size=(640,480)
    _,framef=cap.read()
    framef1=np.copy(framef)

    #cv.NamedWindow("ROI", flags= cv.CV_WINDOW_AUTOSIZE)
    
    framef = cv2.cvtColor(framef, cv2.COLOR_BGR2GRAY)
    h, w = framef.shape[:2]

    x0=300
    y0=300
    dx=75
    ch=0
    a=[]
    let=0
    letter=0
    pr=0
    s=''
    j=1
    flag=100
    win=0
    global SZX
    SZX=60
    global SZY
    SZY=90
    while cv2.waitKey(1)==-1:
        x1,y1=x0-dx,y0-dx
        x2,y2=x0+dx,y0+dx
        
        _,frame=cap.read()
        cv2.rectangle(frame,(x0-dx,y0-dx),(x0+dx,y0+dx),(255,0,0))
        cv2.imshow('1',frame)
        
    cv2.destroyAllWindows()
    
    dx=100
    while cv2.waitKey(1)==-1:
        cv2.namedWindow("animals:D")
        alph=cv2.imread('alphabet.jpg')
        alph=cv2.resize(alph,(648,381))
        cv2.imshow("animals:D",alph)
        alphges=cv2.imread('alphages.png')
        alphges=cv2.resize(alphges,(542,660))
        cv2.imshow("Gestures",alphges)
        shlet=cv2.imread('lets-go.jpg')
        shlet=cv2.resize(shlet,(320,240))
        cv2.imshow("Letters",shlet)
        if x0<dx:
            x0=dx
            
        if y0<dx:
            y0=dx
            
        if y0+dx>h:
            y0=h-dx
            
        if x0+dx>w:
            x0=w-dx
            
        x1,y1=x0-dx,y0-dx
        x2,y2=x0+dx,y0+dx
        
        _,frame=cap.read()
        frame1=np.copy(frame)
        output=np.copy(frame)
        
        mmcc = np.zeros((h, w), np.uint8)
        
        frame_diff1 = cv2.absdiff(frame, framef1)
        fr_diff = cv2.cvtColor(frame_diff1, cv2.COLOR_BGR2GRAY)
        fr_diff=cv2.resize(fr_diff,(320,240))
        _, motion_mask_an = cv2.threshold(fr_diff, 10, 255, cv2.THRESH_BINARY)
        motion_mask_an=cv2.blur(motion_mask_an,(5,5))

        mask5=np.copy(motion_mask_an)
        mask4=np.copy(motion_mask_an)
        cv2.rectangle(mask4,((x0-dx)/2,(y0-dx)/2),((x0+dx)/2,(y0+2*dx)/2),(255,0,0))
        
                
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_diff = cv2.subtract(frame,framef)
        
        minv=np.min(gray_diff)
        maxv=np.max(gray_diff)
        thresh=minv+int(sqrt(maxv-minv))
        
        _, motion_mask = cv2.threshold(gray_diff, thresh, 255, cv2.THRESH_BINARY)
        
        kernel = np.ones((5,5),np.uint8)
        motion_mask2 = cv2.erode(motion_mask,kernel,iterations = 1)
        motion_mask3 = cv2.dilate(motion_mask2,kernel,iterations = 2) 
        motion_mask3=cv2.blur(motion_mask3,(5,5))
        
        mask=np.copy(motion_mask3[y1:(y2+1),x1:(x2+1)])
        p=np.sum(mask)            
               
        contours,_ = cv2.findContours(np.copy(mask), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        sumcx=sumcy=0
        count=0
        for cnt in contours:
            cx,cy=moment(cnt)
            sumcx+=cx
            sumcy+=cy
            count+=1
            
        if count>0:
            sumcx/=count
            sumcy/=count
            
            sumcx+=x0-dx
            sumcy+=y0-dx
            t=1
            newcx=int(x0+t*(sumcx-x0))
            newcy=int(y0+t*(sumcy-y0))-30
            
            if np.sum(mask)>1000*255:
                x0=newcx
                y0=newcy
        
        framef1=np.copy(output)  
        framef=np.copy(frame)
        cv2.rectangle(output,(x0-dx,y0-dx),(x0+dx,y0+2*dx),(255,0,0))

        
        cv2.rectangle(mmcc,(x0-dx,y0-dx),(x0+dx,y0+2*dx),(255,0,0),-1)
        mmcc=cv2.resize(mmcc,(320,240))        
        
        motion_mask_an=cv2.bitwise_and(motion_mask_an,motion_mask_an,mask=mmcc)

        contours,_ = cv2.findContours(motion_mask_an.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if win==1:
            #sleep(2)
            if j<=24:
                j+=1
            if j==25:    
                s=random.choice('ACCCHHHLLL')
                win=0
                j=0
        if win==0:
            draw_str(output,(20,60),'Show "%s"' %s)
            if s=='A':
                shlet_n=cv2.imread('a.jpg')
                shlet_n=cv2.resize(shlet_n,(320,240))
                cv2.imshow("Letters",shlet_n)

            if s=='C':
                shlet_n=cv2.imread('c.jpg')
                shlet_n=cv2.resize(shlet_n,(320,240))
                cv2.imshow("Letters",shlet_n)

            if s=='H':
                shlet_n=cv2.imread('h.jpg')
                shlet_n=cv2.resize(shlet_n,(320,240))
                cv2.imshow("Letters",shlet_n)

            if s=='L':
                shlet_n=cv2.imread('l.jpg')
                shlet_n=cv2.resize(shlet_n,(320,240))
                cv2.imshow("Letters",shlet_n)
            
        else:
            draw_str(output,(20,60),'Great!')
            
 

        for cnt in contours:
            area=cv2.contourArea(cnt)
            if area>3000:
                cv2.imwrite('motion_mask.jpg',motion_mask_an)
                img1=cv.LoadImage('motion_mask.jpg')
                x1b,y1b,w1b,h1b=cv2.boundingRect(cnt)
                cv.SetImageROI(img1, (x1b,y1b,w1b,h1b))
                CvRect=cv.GetImageROI(img1)
                dst = cv.CreateImage(( 60,90), img1.depth, img1.nChannels)
                img1=cv.Resize(img1, dst)
                cv.SaveImage("dst.png",dst)
                dst=cv2.imread("dst.png",0)
                dst=np.array(dst)        
                #dst=deskew(dst)  

                tes = np.asarray(dst[:,:])
                cv2.imwrite('tes.png',tes)
                tes=cv2.imread('tes.png',0)
                tes=np.array(tes)          

                sample = hog([tes])
                letter = model.predict(sample)[0]
                if ch<8:
                    a.append(letter)
                    ch +=1
                if ch==8:
                    a.sort()
                    maxkol=0
                    let=0
                    while len(a)>0:
                        g=a[0]
                        kol=a.count(g)
                        if kol>=maxkol:
                            maxkol=kol
                            let=g
                        a=a[kol:]
                    ch=0
        
        framef = frame.copy()
        dictionary={'0':'','1':'A','2':'B','3':'C','4':'D','9':'I','8':'H','12':'L','15':'O'}
        
        #draw_str(output,(20,60),dictionary[str(int(let))])
        if win==0:
            printim=random.choice('12345')
        if dictionary[str(int(let))]==s:
            win=1
            if printim=='1':
                shlet_n=cv2.imread('Good for you!.jpg')
                shlet_n=cv2.resize(shlet_n,(320,240))
                cv2.imshow("Letters",shlet_n)
            if printim=='2':
                shlet_n=cv2.imread('Well done!.jpg')
                shlet_n=cv2.resize(shlet_n,(320,240))
                cv2.imshow("Letters",shlet_n)
            if printim=='3':
                shlet_n=cv2.imread('welldone.jpg')
                shlet_n=cv2.resize(shlet_n,(320,240))
                cv2.imshow("Letters",shlet_n)
            if printim=='4':
                shlet_n=cv2.imread('well-done.jpg')
                shlet_n=cv2.resize(shlet_n,(320,240))
                cv2.imshow("Letters",shlet_n)
            if printim=='5':
                shlet_n=cv2.imread('Excellent!.jpg')
                shlet_n=cv2.resize(shlet_n,(320,240))
                cv2.imshow("Letters",shlet_n)
              
            
        output=cv2.resize(output,(320,240))    
        cv2.imshow("img",output)
        
        x1,y1=x0-dx,y0-dx
        x2,y2=x0+dx,y0+2*dx
        y2=min(y2,h)
        x2=min(x2,w)
        
    cv2.destroyAllWindows()
    cap.release()
    #os.system('main2.py')

if __name__ == '__main__':
    main()
