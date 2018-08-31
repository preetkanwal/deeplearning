
import csv
import numpy as np
import string
import matplotlib.pyplot as plt



input_file = 'letter.data'  



#returns one hot vector corresponding to each letter
def str_vectorizer(strng, alphabet=string.ascii_lowercase):
    vector = [[0 if char != letter else 1 for char in alphabet] 
                  for letter in strng]
    return vector[0]


#extracts the labels and pixel values
def extract_data(x):
    return {
        'label': str_vectorizer(x[1]),
        'image': x[6:134]
    } 



images = []
labels = []

with open('letter.data') as file:
    reader = csv.reader(file, delimiter='\t')
    for row in reader:
        data = extract_data(row)
        images.append(data['image'])
        labels.append(data['label'])

#Converting to numpy arrays
images = np.asarray(images)
images = images.astype(np.float)
labels = np.asarray(labels)



# Define visualization parameters   
h, w = 16, 8 


label = labels[16,:]



index = 45
img = images[index,:]
label = labels[index,:]
img = np.reshape(img, (h,w)) 
plt.title(string.ascii_lowercase[label.argmax()])
plt.imshow(img, cmap='Greys')

np.savez("letters", images=images, labels=labels)