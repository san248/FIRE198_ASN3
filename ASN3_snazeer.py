from cmath import pi
import cv2
import numpy as np

# Step Two

# Question One
image = cv2.imread('C:\\Users\\nazee\\Desktop\\ASN3\\Frame0064.png')

cv2.imshow('Original image',image)

# Question Two

Ig = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow('Gray image', Ig)

# (a)
Kx = np.array([
  [-1, 0, 1],
  [-2, 0, 2],
  [-1, 0, 1]
])
Gx = cv2.filter2D(Ig, -1, Kx)
Gx = np.uint8(Gx)
cv2.imshow("Horizontal Edges", Gx)

# (b)
Ky = np.array([
  [-1, -2, -1],
  [0, 0, 0],
  [1, 2, 1]
])
Gy = cv2.filter2D(Ig, -1, Ky)
Gy = np.uint8(Gy)
cv2.imshow("Vertical Edges", Gy)

# (c)
G = np.sqrt(np.square(Gx) + np.square(Gy))
G = np.uint8(G)

cv2.imshow("Total Gradient", G)


# Step Three

# Question One
colorImage = cv2.imread('C:\\Users\\nazee\\Desktop\\ASN3\\Frame0064.png', cv2.IMREAD_COLOR)
cv2.imshow('color image', colorImage)
(B, G, R) = cv2.split(colorImage)
cv2.imshow("Red", R)
cv2.imshow("Green", G)
cv2.imshow("Blue", B)

# Question Two
hsvImg = cv2.cvtColor(colorImage, cv2.COLOR_BGR2HSV)
cv2.imshow('HSV image', hsvImg)


cv2.waitKey(0)
cv2.destroyAllWindows()

# Step Four

# Question One
cap = cv2.VideoCapture('C:\\Users\\nazee\\Desktop\\ASN3\\Vid.mp4')

while(True):
    _, frame = cap.read()

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_orange = np.array([0, 130, 100])
    upper_orange = np.array([100, 225, 225])
    
    mask = cv2.inRange(hsv, lower_orange, upper_orange)

    res = cv2.bitwise_and(frame,frame, mask= mask)

    cv2.imshow('frame',frame)
    cv2.imshow('mask',mask)
    cv2.imshow('res',res)
    k = cv2.waitKey(50) & 0xFF
    if k == 27:
        break


cv2.destroyAllWindows()

# Question 3
def gaussian(x, mu, sigma):
    return  (1/(2* pi * abs(sigma)))*np.exp((-1/2)*(np.transpose(x - mu))*(sigma^-1)*(x-mu))


cap = cv2.VideoCapture('C:\\Users\\nazee\\Desktop\\ASN3\\Masks.mp4')
cap2 = cv2.VideoCapture('C:\\Use1rs\\nazee\\Desktop\\ASN3\\Vid.mp4')


while(True):
    _, frame1 = cap.read()
    _, frame2 = cap2.read()

    gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    I = gray * gray2
    image = cv2.cvtColor(I, cv2.COLOR_GRAY2BGR)
    #mu = np.mean(image, axis = 0)
    #sigma = np.cov(image)
    cv2.imshow('image', image)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

