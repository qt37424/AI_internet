import cv2

img = cv2.imread('images/khe.jpg', 1)

fruit_classifier = 'xmlutils.py-master/samples/fruits.xml'

gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

fruit_tracker = cv2.CascadeClassifier(fruit_classifier)

fruit = fruit_tracker.detectMultiScale(gray_img)
print(fruit)

for (x, y, w, h) in fruit:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.putText(img, 'fruit', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

# Finally display the image with the markings
cv2.imshow('my detection',img)

# wait for the keystroke to exit
cv2.waitKey()


print("I'm done")