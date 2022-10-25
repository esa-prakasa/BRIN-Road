#Menentukan ROI (Region of Interest)
def roi(img):
    x = int(img.shape[1])
    y = int(img.shape[0])
    shape = np.array([[int(0.045*x), int(0.85*y)], [int(0.73*x), int(0.85*y)], [int(0.49*x), int(0.58*y)], [int(0.375*x), int(0.58*y)]])
    #define a numpy array with the dimensions of img, but comprised of zeros
    mask = np.zeros_like(img)
    #Uses 3 channels or 1 channel for color depending on input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
    #creates a polygon with the mask color
    cv2.fillPoly(mask, np.int32([shape]), ignore_mask_color)
    #returns the image only where the mask pixels are not zero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def color_filter(image):
    #convert RGB to HLS to mask based on HLS
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    lower = np.array([56,120,0])
    upper = np.array([255,255,255])
    yellower = np.array([91,91,46])
    yelupper = np.array([255,150,255])
    yellowmask = cv2.inRange(hls, yellower, yelupper)    
    whitemask = cv2.inRange(hls, lower, upper)
    mask = cv2.bitwise_or(yellowmask, whitemask)  
    masked = cv2.bitwise_and(image, image, mask = mask)    
    return masked

def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

rightSlope, leftSlope, rightIntercept, leftIntercept = [],[],[],[]
def draw_lines(img, lines, thickness=5):
    global rightSlope, leftSlope, rightIntercept, leftIntercept
    rightColor=[0,255,0]
    leftColor=[255,0,0]
    
    #Menghitung slope dan  y-intercept untuk b di persamaan y=mx+b
    for line in lines:
        for x1,y1,x2,y2 in line:
            slope = (y1-y2)/(x1-x2)
            if slope > 0.3:
                if x1 > (640-64):#903 : #500 640 320
                    yintercept = y2 - (slope*x2)                    
                    rightSlope.append(slope)
                    rightIntercept.append(yintercept)
                else: None                
            elif slope < -0.3:
                if x1 < (640-64) : #1018: #600 640 320
                    yintercept = y2 - (slope*x2)                    
                    leftSlope.append(slope)
                    leftIntercept.append(yintercept)    
                 
    #We use slicing operators and np.mean() to find the averages of the 30 previous frames
    #This makes the lines more stable, and less likely to shift rapidly
    leftavgSlope = np.mean(leftSlope[-30:])
    leftavgIntercept = np.mean(leftIntercept[-30:])
    
    rightavgSlope = np.mean(rightSlope[-30:])
    rightavgIntercept = np.mean(rightIntercept[-30:])
    
    #Here we plot the lines and the shape of the lane using the average slope and intercepts
    try:
        left_line_x1 = int((0.58*img.shape[0] - leftavgIntercept)/leftavgSlope)
        left_line_x2 = int((img.shape[0] - leftavgIntercept)/leftavgSlope)
        right_line_x1 = int((0.58*img.shape[0] - rightavgIntercept)/rightavgSlope)
        right_line_x2 = int((img.shape[0] - rightavgIntercept)/rightavgSlope)
        pts = np.array([[left_line_x1, int(0.58*img.shape[0])],[left_line_x2, int(img.shape[0])],[right_line_x2, int(img.shape[0])],[right_line_x1, int(0.58*img.shape[0])]], np.int32)
        pts = pts.reshape((-1,1,2))
        cv2.fillPoly(img,[pts],(0,0,255))      
        cv2.line(img, (left_line_x1, int(0.58*img.shape[0])), (left_line_x2, int(img.shape[0])), leftColor, 3)
        cv2.line(img, (right_line_x1, int(0.58*img.shape[0])), (right_line_x2, int(img.shape[0])), rightColor, 3)
    except ValueError:
        pass

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
   lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
        if lines is not None:
            line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
            draw_lines(line_img, lines)
            #print ('Ada hasil HL')
        else: 
            print ('TIDAK ADA HASIL HL')
            line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
            pass
        return line_img
#Overlaying the Image and the Lines
def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
   return cv2.addWeighted(initial_img, α, img, β, λ)

#Applying it to Video
def processImage(image):
    	interest = roi(image)
	filtering = color_filter(interest)
	canny = cv2.Canny(grayscale(filtering), 50, 120)
	myline = hough_lines(canny, 1, np.pi/180, 10, 37, 6)
	weighted_img = cv2.addWeighted(myline, 1, image, 0.8, 0)
    	return weighted_img

cap = cv2.VideoCapture(videoFile)
while cap.isOpened():
	ret, frame = cap.read()
	frame = processImage(frame)
	x = int(frame.shape[1])
	y = int(frame.shape[0])
	
	cv2.imshow('out video '+videoFile,frame)
    	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()	
