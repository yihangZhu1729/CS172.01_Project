import cv2
import numpy as np

def eliminate_rectangle(image_dir):
    img = cv2.imread(image_dir)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret,binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    draw_img = cv2.drawContours(img.copy(), contours[:25], -1, (0, 0, 255), 3)

    cv2.imshow("img", img)
    cv2.imshow("draw_img", draw_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def center(image_dir):
    img = cv2.imread(image_dir)
    img = img[10:349,10:879]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((5,5),np.float32)/25
    gray = cv2.filter2D(gray,-1,kernel)
    print(gray)
    print(gray.shape)

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
    print(up)

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
    print(left)

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
    print(down)

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
    print(right)

    new_img = img[up:down,left:right]
    cv2.imshow("new_img", new_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def eliminate_edge(image_dir):
    pass

if __name__ == '__main__':
    eliminate_rectangle('0110005_11.png')
    # center('0109002_4.png')