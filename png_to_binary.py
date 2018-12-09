import cv2
import os 

data_path = "../data/png"

for root, dirs, files  in os.walk(data_path):
	cv2.namedWindow("ori")
	cv2.namedWindow("bin")
	
	if (len(dirs)==0):
		for f in files:
			
			im = cv2.imread(os.path.join(root, f), cv2.CV_LOAD_IMAGE_GRAYSCALE)
			#print (im.shape)
			#cv2.imshow("ori",im)
			ret, im_bin = cv2.threshold(im, 127, 255, cv2.THRESH_BINARY)
			#print (im_bin.shape)
			#cv2.imshow("bin", im_bin)
			#cv2.waitKey(0)

			dirname = root.replace("png", "png_binary")
			if (not os.path.exists(dirname)):
				os.makedirs(dirname)
			cv2.imwrite(os.path.join(dirname, f), im_bin)