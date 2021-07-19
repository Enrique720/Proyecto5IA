# example of zoom image augmentation
from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
import glob
from PIL import Image
from PIL import ImageOps

# load the image

dir = './db/'
images = glob.glob(dir+'*')


#from skimage import io
def corp_margin(img):
        img2=img.sum(axis=2)
        (row,col)=img2.shape
        row_top=0
        raw_down=0
        col_top=0
        col_down=0
        for r in range(0,row):
                if img2.sum(axis=1)[r]<700*col:
                        row_top=r
                        break
 
        for r in range(row-1,0,-1):
                if img2.sum(axis=1)[r]<700*col:
                        raw_down=r
                        break
 
        for c in range(0,col):
                if img2.sum(axis=0)[c]<700*row:
                        col_top=c
                        break
 
        for c in range(col-1,0,-1):
                if img2.sum(axis=0)[c]<700*row:
                        col_down=c
                        break
 
        new_img=img[row_top:raw_down+1,col_top:col_down+1,0:3]
        return new_img

for imgs in images: 
    #print(image)
    img = load_img(imgs)

    data = img_to_array(img)
    samples = expand_dims(data, 0)
    datagen = ImageDataGenerator(zoom_range=[0.5, 1.0])
    it = datagen.flow(samples, batch_size=1)

    for i in range(3): 
        #pyplot.subplot(300+1+i)
        batch = it.next()
        image = batch[0].astype('uint8')
        #image = corp_margin(image)
        print(image.shape)
        pyplot.imshow(image)
        pyplot.axis('off')
        #pyplot.show()

        
        pyplot.savefig("db_aumentation/" + imgs[5:] + "_" + str(i)+ ".png",bbox_inches='tight')
        pyplot.clf()
    
#pyplot.savefig('db_aumentation/a.png')

#pyplot.show()

print("starting margin reduction ")
dir = './db_aumentation/'
images = glob.glob(dir+'*')

for filePath in images:
    image=Image.open(filePath)
    image.load()
    imageSize = image.size
    
    # remove alpha channel
    invert_im = image.convert("RGB") 
    
    # invert image (so that white is 0)
    invert_im = ImageOps.invert(invert_im)
    imageBox = invert_im.getbbox()
    
    cropped=image.crop(imageBox)
   # print filePath, "Size:", imageSize, "New Size:", imageBox
    cropped.save(filePath)