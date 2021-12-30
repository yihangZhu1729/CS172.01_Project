import cv2
import numpy as np
    
def center(image_dir):
    img = cv2.imread(image_dir)
    img = img[15:img.shape[0]-15,15:img.shape[1]-15]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((5,5),np.float32)/25
    gray = cv2.filter2D(gray,-1,kernel)

    up = 0
    for i in range(gray.shape[0]):
        bool1 = 0
        for j in range(gray.shape[1]):
            if gray[i][j] < 200:
                bool1 = 1
                break
        if bool1 == 1:
            break
        up = i

    left = 0
    for i in range(gray.shape[1]):
        bool1 = 0
        for j in range(gray.shape[0]):
            if gray[j][i] < 200:
                bool1 = 1
                break
        if bool1 == 1:
            break
        left = i

    down = gray.shape[0]-1
    for i in range(gray.shape[0]-1,-1,-1):
        bool1 = 0
        for j in range(gray.shape[1]-1,-1,-1):
            if gray[i][j] < 200:
                bool1 = 1
                break
        if bool1 == 1:
            break
        down = i

    right = gray.shape[1]-1
    for i in range(gray.shape[1]-1,-1,-1):
        bool1 = 0
        for j in range(gray.shape[0]-1,-1,-1):
            if gray[j][i] < 200:
                bool1 = 1
                break
        if bool1 == 1:
            break
        right = i

    new_img = img[up:down,left:right]

    gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
    ret,binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    draw_img = cv2.drawContours(new_img.copy(), contours[:int(len(contours)/8)], -1, (255, 255, 255), 3)

    img = draw_img
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((9,9),np.float32)/81
    gray = cv2.filter2D(gray,-1,kernel)

    up = 0
    for i in range(gray.shape[0]):
        bool1 = 0
        for j in range(gray.shape[1]):
            if gray[i][j] < 200:
                bool1 = 1
                break
        if bool1 == 1:
            break
        up = i

    left = 0
    for i in range(gray.shape[1]):
        bool1 = 0
        for j in range(gray.shape[0]):
            if gray[j][i] < 200:
                bool1 = 1
                break
        if bool1 == 1:
            break
        left = i

    down = gray.shape[0]-1
    for i in range(gray.shape[0]-1,-1,-1):
        bool1 = 0
        for j in range(gray.shape[1]-1,-1,-1):
            if gray[i][j] < 200:
                bool1 = 1
                break
        if bool1 == 1:
            break
        down = i

    right = gray.shape[1]-1
    for i in range(gray.shape[1]-1,-1,-1):
        bool1 = 0
        for j in range(gray.shape[0]-1,-1,-1):
            if gray[j][i] < 200:
                bool1 = 1
                break
        if bool1 == 1:
            break
        right = i

    new_img = img[up:down,left:right]

    return new_img

if __name__ == '__main__': 
    nparray = center('0109002_4.png')
