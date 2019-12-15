import face_recognition as f
face = f.load_image_file('./test/group.jpg')
loc = f.face_locations(face)

number = len(loc)
# print(loc)
print("number of people -", number)
