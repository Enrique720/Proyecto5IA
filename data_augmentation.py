# example of zoom image augmentation
from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
import glob

# load the image

dir = './db/'
images = glob.glob(dir+'*')

for imgs in images: 
    #print(image)
    img = load_img(imgs)

    data = img_to_array(img)
    samples = expand_dims(data, 0)
    datagen = ImageDataGenerator(zoom_range=[0.5, 1.0])
    it = datagen.flow(samples, batch_size=1)

    #for i in range(3):
    #	pyplot.subplot(330 + 1 + i)
    #	batch = it.next()
    #	image = batch[0].astype('uint8')
    #
    #	pyplot.imshow(image)
    #
        
    for i in range(3): 
        #pyplot.subplot(300+1+i)
        batch = it.next()
        image = batch[0].astype('uint8')
        pyplot.imshow(image)
        pyplot.axis('off')
        
        pyplot.savefig("db_aumentation/" + imgs[5:] + "_" + str(i)+ ".png")
        pyplot.clf()
    
#pyplot.savefig('db_aumentation/a.png')

#pyplot.show()
