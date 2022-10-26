Tahap Persiapan (Inisialisasi)

import cv2
import glob
import os
import matplotlib.pyplot as plt
from helpers.helper_xml import *

dataset_list = glob.glob('dataset/front/annotation/*.xml')
path_pos = os.path.join('./dataset/front/pos')
path_neg = os.path.join('./dataset/front/neg')


train_images,train_labels = get_images(dataset_list, (64,64))

for x in range(len(train_images)):
	if (train_labels[x]== '1'):
 cv2.imwrite(path_pos+'\img'+str(x)+'.png',train_images[x])   	 
	elif (train_labels[x]=='0'):
    	cv2.imwrite(path_neg+'\img'+str(x)+'.png',train_images[x])


Tahap Training

# deklarasi parameter untuk HOG feature extraction
orientations = 9
pixels_per_cell = (8, 8)
cells_per_block = (2, 2)

# positif dataset:
pos_im_listing = glob.glob('./dataset/front/pos/*')

# negatif dataset:
neg_im_listing= glob.glob('./dataset/front/neg/*')

num_pos_samples = len(pos_im_listing)
num_neg_samples = len(neg_im_listing)
print("Gambar positif:",num_pos_samples)
print("Gambar negatif:",num_neg_samples)

data =[]
labels =[]

img = cv2.imread(pos_im_listing[1])
plt.imshow(img)

for file in pos_im_listing:
	img = cv2.imread(file)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
	# HOG positive fitur
	fd = hog(gray, orientations, pixels_per_cell, cells_per_block, block_norm='L2', feature_vector=True)
	data.append(fd)
	labels.append(1)

for file in neg_im_listing:
	img = cv2.imread(file)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
	# HOG negative fitur
	fd = hog(gray, orientations, pixels_per_cell, cells_per_block, block_norm='L2', feature_vector=True)
	data.append(fd)
	labels.append(0)

# konversi label ke integer
le = LabelEncoder()
labels = le.fit_transform(labels)

# pembagian data training dan testing, untuk training 80%, testing 20%
print(" Proses splitting data...")

(trainData, testData, trainLabels, testLabels) = train_test_split(np.array(data), labels, test_size=0.20, random_state=42)

print(" Proses Training Linear SVM classifier...")
model = LinearSVC()
model.fit(trainData, trainLabels)

# Simpan model:
joblib.dump(model, 'models/model_svm_brin_frontx.mdl')

Tahap Testing

# deklarasi parameter untuk HOG feature extraction
orientations = 9
pixels_per_cell = (8, 8)
cells_per_block = (2, 2)

ystart=0
ystop=0

# panggil model:
model = joblib.load(os.path.join('./models/model_svm_brin_front.mdl'))

# Tes model
scale = 0
detections = []

img = cv2.imread(os.path.join('./images/','img_12.jpg'))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)

sc0 = int(img.shape[0]/2)
sc1 = int(img.shape[1]/2)

# Ubah ukuran gambar jika terlalu besar
img= cv2.resize(img,(sc1, sc0))
img2 =img

# Menentukan ukuran sliding window
(winW, winH)= (64,64)
windowSize=(winW,winH)
downscale=1.5

for resized in pyramid(img, scale=1.5):
    
	# perulangan di atas jendela geser untuk setiap lapisan piramida
	for (x,y,window) in sliding_window(resized, ystart, ystop, stepSize=8, windowSize=(winW,winH)):
    	# jika jendela tidak memenuhi ukuran jendela yang kita inginkan, abaikan saja!
    	if window.shape[0] != winH or window.shape[1] !=winW:
        	continue
       	 
    	window=color.rgb2gray(window)

    	fds = hog(window, orientations, pixels_per_cell, cells_per_block, block_norm='L2')  # ekstraksi fitur HOG
    	fds = fds.reshape(1, -1)
    	pred = model.predict(fds) # gunakan model SVM untuk membuat prediksi pada fitur HOG yang diekstraksi dari jendela
   	 
    	if pred == 1:
        	# tetapkan nilai ambang batas untuk prediksi SVM yaitu hanya tegaskan prediksi di atas probabilitas 0,6
        	if model.decision_function(fds) > 0.6:
            	# print("Lokasi terdeteksi -> ({}, {})".format(x, y))
            	# print("Skala ->  {} | Skor kepercayaan {} \n".format(scale,model.decision_function(fds)))
            	detections.append((int(x * (downscale**scale)), int(y * (downscale**scale)), model.decision_function(fds),
                               	int(windowSize[0]*(downscale**scale)), # buat list semua prediksi yang ditemukan
                                  	int(windowSize[1]*(downscale**scale))))
	scale+=1
    
clone = resized.copy()

rects = np.array([[x, y, x + w, y + h] for (x, y, _, w, h) in detections]) # lakukan nms pada kotak pembatas yang terdeteksi

sc = [score[0] for (x, y, score, w, h) in detections]

# print("skor kepercayaan deteksi: ", sc)
sc = np.array(sc)
pick = non_max_suppression(rects, probs = sc, overlapThresh = 0.3)

for (xA, yA, xB, yB) in pick:
	cv2.rectangle(img, (xA, yA), (xB, yB), (0,255,0), 1)
    
plt.imshow(img)


Tahap Video Processing

def process_image(img):
return pipeline(img, ystart, ystop, model)

ystart=30
ystop=50
    
# memanggil model yang sudah ditraining:
model = joblib.load(os.path.join('./models/model_svm_brin_front.mdl'))

HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(video_output))
