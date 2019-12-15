import face_recognition as f
from PIL import Image as image

img = f.load_image_file('./test/group.jpg')
loc = f.face_locations(img)

for loc1 in loc:
	top,right,bottom,left = loc1
	fimg = img[top:bottom, left:right]
	pimg = image.fromarray(fimg)
	pimg.show()