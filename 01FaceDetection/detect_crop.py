import dlib
import cv2
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

root_path = 'faces'
files= os.listdir(root_path)

global_counter = 0

for i in tqdm(range(len(files))):
	try:
		file = files[i]
		img_pic = cv2.imread(root_path + os.sep + file)
		detector = dlib.get_frontal_face_detector()
		b, g, r = cv2.split(img_pic)
		img_rgb = cv2.merge([r, g, b])
		dets =detector(img_rgb, 1)
		if (len(dets) != 1):
			continue

		for index, face in enumerate(dets):
		# print('face {}; left {}; top {}; right {}; bottom {}'.format(index, face.left(), face.top(), face.right(), face.bottom()))
	#     cv2.rectangle(img_pic,(face.left()-40,face.top()-70),(face.right()+50,face.bottom()+60),(0,255,0),6)
	#     cropImg = img_pic[int(face.top()-70):int(face.bottom()+60),int(face.left()-40):int(face.right()+50)] 
			square_side_size = int(face.bottom()) - int(face.top())
			additional = 0.1 * square_side_size
			cropImg = img_pic[int(face.top()-additional):int(face.bottom()+additional),int(face.left()-additional):int(face.right()+additional)] 
			resized_image = cv2.resize(cropImg, (512, 512)) 
			plt.close()
			cv2.imwrite('result/' + str(global_counter) + '_original.jpg', resized_image)

		global_counter = global_counter + 1
	except:
		continue
	
	
