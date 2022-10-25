import cv2 as cv
import numpy as np

# Doubled intensity yang akan jalan di setiap frame video
# Maka yang diminta adalah framenya, besar intensitas, dan besar threshold
def doubled_intensity(frame, intensity_value, thresh_value):
    
    # Buat matrix untuk mengalikan semua channel frame nya dengan besar 
    # intensitas.
    intensity_matrix = np.ones(frame.shape, dtype="uint8")*intensity_value
    # Tambahkan nilai matrix intensitas ke frame asli
    bright_img = cv.add(frame, intensity_matrix)
    # Threshold hasil penggabungan
    ret, intensity_threshold = cv.threshold(bright_img, thresh_value, 255, cv.THRESH_BINARY)

    return intensity_threshold

# Edge detection akan dijalankan di setiap frame video yang telah di doubled intensity.
# Maka yang akan diminta adalah frame setelah doubled intensity, dan ukuran kernel
def sobel_edge_detection(doubled_int_frame, ksize):
    
    blur_frame = cv.GaussianBlur(doubled_int_frame, (5,5), 0)
    gray_frame = cv.cvtColor(blur_frame, cv.COLOR_BGR2GRAY)
    
    ddepth = cv.CV_16S
    scale = 1
    delta = 0

    grad_x = cv.Sobel(gray_frame, ddepth, 1, 0, ksize=ksize, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)
    grad_y = cv.Sobel(gray_frame, ddepth, 0, 1, ksize=ksize, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)  
    
    abs_grad_x = cv.convertScaleAbs(grad_x)
    abs_grad_y = cv.convertScaleAbs(grad_y)

    grad = cv.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    
    return grad

# Threshold hasil edge detectionnya, nanti hasilnya warna hitam merupakan edge yang terdeteksi dan 
# warna putih adalah jalan/langit. Sehingga apabila di contour yang terdeteksi adalah si warna 
# putih (jalan/langit)
# Maka yang diminta adalah frame setelah edge detection
def thresholding(edge_det_frame):

    ret, thresh = cv.threshold(edge_det_frame, 10, 255, cv.THRESH_BINARY_INV)
    
    return thresh

# ROI untuk mengambil hanya bagian jalanan saja, karena hanya menggunakan beberapa sample video 
# sehingga perlu mendefinisikan ukuran ROI yang berbeda-beda diluar loop video performance
# Maka yang diminta adalah rect (ukuran ROI), dan hasil threshold
def region_of_interest(roi_size, threshold_frame):
    
    # Buat array dari ukuran roi (rect)
    mask_frame = np.array([roi_size], dtype=np.int32)
    
    # Buat array kosong, seukuran frame asli (jadi sebaikanya di resize dari awal)
    zeros_frame = np.zeros((threshold_frame.shape[0], threshold_frame.shape[1]), np.int8)
    
    # Gabungkan hasil roi ke dalam array kosong
    cv.fillPoly(zeros_frame, mask_frame, 255)
    
    # Threshold hasil penggabungan dan definisikan ROI sebagai hasil dari
    # edge detection
    thresh_mask = cv.inRange(zeros_frame, 1, 255)
    roi = cv.bitwise_and(threshold_frame, threshold_frame, mask=thresh_mask)

    return roi

# Setelah melalui ROI dimana fokus contour akan ke jalanan saja
# Selanjutnya mewarnai jalanan dengan mengambil contour area terbesar
# Maka yang diminta adalah frame asli yang sudah di copy, dan roi
def recoloring(copy_frame, roi_frame):
    
    # Temukan contour (putih) pada roi_frame menggunakan cv.findContours
    contour, h = cv.findContours(roi_frame, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    
    # Jadikan area contour terluas atau maksimal sebagai area untuk di recolor
    areas = [cv.contourArea(cont) for cont in contour]
    maxAreaIndex = areas.index(max(areas))
    
    # Recolor area menggunakan cv.drawContours
    return cv.drawContours(copy_frame, contour, maxAreaIndex, (255, 0, 0), -1)

# Concatenate Frame tiap Progress
def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv.cvtColor( imgArray[x][y], cv.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv.cvtColor(imgArray[x], cv.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver



# PERFORMANCE VIDEO

#cap = cv.VideoCapture('Dashcam_0021.mp4')
cap = cv.VideoCapture('Dashcam_0091.mp4')
#cap = cv.VideoCapture('Dashcam_0061.mp4')
#cap = cv.VideoCapture('Dashcam_0071.mp4')
#cap = cv.VideoCapture('Dashcam_0081.mp4')
#cap = cv.VideoCapture('Dashcam_0091.mp4')

# UNTUK Dashcam_21, Dashcam_31, Dashcam_71, Dashcam_81, Dashcam_91
# GUNAKAN :
rect = [[480,240], [480, 150], [0, 150],[0,240]]

# UNTUK Dashcam_61
# GUNAKAN :
#rect = [[480,240], [480, 100], [0, 100],[0,240]]

# Jalankan Video dengan looping
framecounter = 0

while(1):
    framecounter += 1
    if cap.get(cv.CAP_PROP_FRAME_COUNT) == framecounter:
        cap.set(cv.CAP_PROP_POS_FRAMES, 0)
        framecounter = 0
    
    # STEP 1
    # Capture video's images frame by frame
    ret, frame = cap.read()
    frame = cv.resize(frame, (480, 240))
    
    # STEP 2
    # Perform doubled intensity on the frame
    # Besar intensitas dan thresh berdasarkan tabel
    # param = doubled_intensity(frame, intensitas, threshold)
    DInt = doubled_intensity(frame, 150, 210)
    
    # STEP 3
    # Perform edge detection on the doubled intensity frame
    EdgeDet = sobel_edge_detection(DInt, 5)
    #EdgeDet = canny_edge_detection(DInt)
    
    # STEP 4
    # Threshold the frame after edge detection
    Thresh = thresholding(EdgeDet)
    
    # STEP 5
    # Make a 'region of interest' wich only focus on the road from the 
    # frame after threshold
    ROI = region_of_interest(rect, Thresh) #ERROR NIH ukurannya beda karena resizenya beda ganti ganti lagi ntar ygy

    #6. Copy the original or doubled intensity frame to make a red contours
    frame_copy = frame.copy()
    
    #7. Find contours of a roi frame and draw the contour wich only choose max area index (the road must be max area index)
    RC = recoloring(frame_copy, ROI)
    
    # STEP 8
    # Show output using concanate func
    # This cannot be used for different chanel
    output = stackImages(0.5,([frame,DInt,EdgeDet],[Thresh,ROI,RC]))
    
    # Show the output
    cv.imshow('Output', output)
    key_pressed = cv.waitKey(25) & 0xFF
    if key_pressed == ord('q'):
        break

cv.destroyAllWindows()
cap.release()
