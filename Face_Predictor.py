from keras.preprocessing import image
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
import matplotlib.image as img
import random, os

# load model
model = load_model('model-cnn15.h5')

input_image=img.imread('Final-test/s1.jpg')

test_image=image.load_img('Final-test/s1.jpg', target_size=(64,64))
test_image=image.img_to_array(test_image)
test_image=np.expand_dims(test_image, axis=0)

#predict model
result= model.predict(test_image)

if result[0][0]>=0.5:
    prediction='Prashant'
    res=result[0][0]
    path2 = "/home/darth-anishman/Face_recogn/dataset/test_set/Ghims"
    
elif result[0][1]>=0.5:
    prediction='Salman Khan'
    res=result[0][1]
    path2 = "/home/darth-anishman/Face_recogn/dataset/test_set/Salman"
elif result[0][2]>=0.5:
    prediction='Shahrukh Khan'
    res=result[0][2]
    path2 = "/home/darth-anishman/Face_recogn/dataset/test_set/Shahrukh"
else:
    prediction='Not recognized'

print('THE RECOGNIZED FACE IN THE IMAGE IS ', prediction)
print('PROBABILITY= ',res)

path=path2
random_filename = random.choice([
    x for x in os.listdir(path)
    if os.path.isfile(os.path.join(path, x))
])

predicted_image=img.imread(path+'/'+random_filename)


fig = plt.figure()
a = fig.add_subplot(1, 2, 1)
plt.imshow(predicted_image)
a.set_title('Predicted Face')

b= fig.add_subplot(1,2,2)
b.set_title('Input Face')
plt.imshow(input_image)






