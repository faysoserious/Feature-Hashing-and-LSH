import skimage
resize_width = 9
resize_height = 8
image = skimage.io.imread("hash1.PNG")
grayscale_image = skimage.color.rgb2gray(image)
smaller_image = skimage.transform.resize(grayscale_image,(resize_height,resize_width),mode='reflect')
pixels = smaller_image
difference = [[] for _ in range(resize_height)]
for i in range(resize_height):
    for j in range(resize_width-1):
        if(smaller_image[i][j]>smaller_image[i][j+1]):
            difference[i].append(True)
        else:
            difference[i].append(False)# every difference is 8 bit, 8 of 8 bit derive a hex

