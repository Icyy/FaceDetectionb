import face_recognition as f
from PIL import Image as img 
from PIL import ImageDraw as imgd 
import numpy as np 

f2 = f.load_image_file('./test/known/eminem.jpg')
enco2 = f.face_encodings(f2)[0]

f1 = f.load_image_file('./test/known/Logic.png')
enco = f.face_encodings(f1)[0]

me = f.load_image_file('./test/known/me.jpg')
genco = f.face_encodings(me)
name = input("enter file name with the extension")
uimage = f.load_image_file('./test/unknown/'+name)
if name.endswith(".jpg") or name.endswith(".png") or name.endswith(".jpeg"):
    f1 = f.face_locations(uimage)
    # print(f1) dimensions of the face found on the image
    enco3 = f.face_encodings(uimage,f1)
    pimg = img.fromarray(uimage)
    draw = imgd.Draw(pimg)
else:
    print("file is not in correct format")

knownFaces = [enco, enco2, genco]
knownNames = ["Logic", "eminem", "Om"]
#looping through the locations of faces and the encoding to check the match

for(top,right,bottom,left), fecno in zip(f1, enco3):
    match = f.compare_faces(knownFaces,fecno)
    # print(match) returns an array comparing known faces with unknown encoding and it has boolean values whether it matches the image or not eg [true, false]
    name1 = "unknown person"

    # if True in match:
	   # firstIndex = match.index(True)
	   # name1 = knownNames[firstIndex]
    face_distances = f.face_distance(knownFaces,fecno)
    #it compares the distance of lines in the encoding from knowFaces with the fenco one by one. so first logic's enco will be compared then em's
    # print(face_distances) if you do this you will get - [0.80937696 0.54649573], logic's is the first, em's distance is second. clearly the lower difference between the original and the unkown the better
    best_match_index = np.argmin(face_distances)
    #argmin returns the index value of the member which is the least in value column wise in two arrays. here the two arrays will be one of logic's and one with em's
    # print(best_match_index) if you do this you will get 1 as the output cus, the element at 1st postition in the above array is the min and we know the lower the distance the better the match so it checks out eminem as the answer for em.jpg image
    # best_match_index has only the index value of the element with the lowest distance
    if match[best_match_index]:
        name1 = knownNames[best_match_index]

    draw.rectangle(((left, top) ,(right, bottom)), outline=(0,0,0))
    tw, th = draw.textsize(name1)
    draw.rectangle(((left, bottom -th-10), (right, bottom)), fill=(0,0,0), outline=(0,0,0))	


    draw.text((left + 6, bottom - th - 5), name1, fill=(255, 255, 255, 255))



del imgd
pimg.show()


