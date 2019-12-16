# FaceDetectionb
ok you'll need the following modules in python 
- openCV-python
- Face_Recognition
- PIllOW
- numpy



use box.py for the image recognition script. 
vreco.py is an on going project where the script will recognize poeple from direct video feed, I will complete it soon.

instructions to use box.py

first you'll have to add one sample image of the person you are testing this script for. 
to do that place that sample image in /test/known folder. 


then open box.py and add the following code

image_name = f.load_image_file("/test/known/file_name.jpg")


known_encoding = f.face_encodings(image_name)



then just add the known_encoding to the knownFaces array and add the name of the person to knownNames array.
you are good to go now.

note - place the images to be check(the unknown) in /test/unknown folder.
