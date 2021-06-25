import cv2
import numpy as np
import pybgs as bgs

def remove_bg(img, BLUR=21, CANNY_THRESH_1=10, CANNY_THRESH_2=200, MASK_DILATE_ITER=10, MASK_ERODE_ITER=10, MASK_COLOR=(0.0, 0.0, 1.0)):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("ouput", gray)
    # cv2.waitKey(0)
    edges = cv2.Canny(gray, CANNY_THRESH_1, CANNY_THRESH_2)
    # cv2.imshow("ouput", edges)
    # cv2.waitKey(0)
    edges = cv2.dilate(edges, None)
    # cv2.imshow("ouput", edges)
    # cv2.waitKey(0)
    edges = cv2.erode(edges, None)
    # cv2.imshow("ouput", edges)
    # cv2.waitKey(0)

    #find contour
    contour_info = []
    # contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    clone = img.copy()
    new = cv2.drawContours(clone, contours, -1, (0, 255, 0), 2)
    cv2.imshow('out', new)
    cv2.waitKey(0)
    return 
    for c in contours:
        contour_info.append((c, cv2.isContourConvex(c), cv2.contourArea(c)))
    contour_info = sorted(contour_info, key=lambda c: c[2], reverse=True)
    
    max_contour = contour_info[0]
    
    mask = np.zeros(edges.shape)
    cv2.fillConvexPoly(mask, max_contour[0], (255))
    cv2.imshow('output', mask)
    cv2.waitKey(0)
    mask = cv2.dilate(mask, None, iterations=MASK_DILATE_ITER)
    mask = cv2.erode(mask, None, iterations=MASK_ERODE_ITER)
    mask = cv2.GaussianBlur(mask, (BLUR, BLUR), 0)
    
    mask_stack = np.dstack([mask]*3)

    mask_stack = mask_stack.astype('float32')/255.0
    img = img.astype('float32')/255.0
    masked = (mask_stack * img) + ((1-mask_stack) * MASK_COLOR)
    masked = (masked * 255).astype('uint8')
    
    c_blue, c_green, c_red = cv2.split(img)
    img_a = cv2.merge((c_red, c_green, c_blue, mask.astype('float32') / 255.0) )
    # cv2.imshow("ouput", img_a)
    # cv2.waitKey(0)
    return img_a.astype('float32') * 255

if __name__ == '__main__':
    test_image = cv2.imread('data/image/street_snapshot.png')

    t_blur = 21
    thresh_1 = 50
    thresh_2 = 70
    dilate = 10
    erode = 6

    img_fin = remove_bg(test_image.copy(), BLUR=t_blur, CANNY_THRESH_1=thresh_1, CANNY_THRESH_2=thresh_2, MASK_DILATE_ITER=dilate, MASK_ERODE_ITER=erode)
    cv2.imshow("ouput", cv2.cvtColor(img_fin, cv2.COLOR_RGBA2BGRA))
    cv2.imshow("ouput", img_fin)
    cv2.waitKey(0)

    # bgs_method = bgs.SuBSENSE()
    # fore = bgs_method.apply(test_image)
    # back = bgs_method.getBackgroundModel()

    # cv2.imshow('123', fore)
    # cv2.waitKey(0)
    # cv2.imshow('123', back)
    # cv2.waitKey(0)
