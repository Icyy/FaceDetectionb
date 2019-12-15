import face_recognition as f
from PIL import Image as img
image = f.load_image_file('./test/known/Logic.png')
enco = f.face_encodings(image)[0]

image2 = f.load_image_file('./test/known/eminem.jpg')
enco2 = f.face_encodings(image2)[0]

#print(enco)

name = input("enter file name with extension")
uimage = f.load_image_file('./test/unknown/'+name)
if name.endswith(".jpg") or name.endswith(".png"):
	enco3 = f.face_encodings(uimage)[0]
else:
	print("file is not in correct format")

knownFaces = [enco, enco2]

pimage  = img.fromarray(uimage)

r = f.compare_faces(knownFaces, enco3)

if r[0]:
	print("yes this is logic")
	pimage.show()
elif r[1]:
	print("this person is eminem")
	pimage.show()
else:
	print("im sorry i dont know who this is")
	pimage.show()

