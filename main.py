import cv2

# Load Haar Cascade
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Load model
ageNet = cv2.dnn.readNet("age_net.caffemodel", "age_deploy.prototxt")
genderNet = cv2.dnn.readNet("gender_net.caffemodel", "gender_deploy.prototxt")

print("Age model:", not ageNet.empty())
print("Gender model:", not genderNet.empty())

# Label
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(21-24)',
           '(25-32)', '(38-43)', '(48-53)', '(60-100)']

genderList = ['Male', 'Female']

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray, 1.1, 5
    )

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]

        if face.shape[0] < 100 or face.shape[1] < 100:
            continue

        blob = cv2.dnn.blobFromImage(
            face, 1.0, (227,227),
            (78.4263377603, 87.7689143744, 114.895847746),
            swapRB=False
        )

        # Gender
        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        gender = genderList[genderPreds[0].argmax()]

        # Age
        ageNet.setInput(blob)
        agePreds = ageNet.forward()
        age = ageList[agePreds[0].argmax()]

        label = f"{gender}, {age}"
        print(label)

        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(frame, label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

    cv2.imshow("Age & Gender Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()