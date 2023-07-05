import cv2
import numpy as np
# import matplotlib.pyplot as plt
# %matplotlib inline
from IPython.display import Image


IMAGE_NAME = "./images/Fig81a.tif"
FILENAME = "lzw_encodecode.txt"
DECODED_IMAGE_NAME = "Decoded_Image.tif"

def encodeWrite(image_shape, block_size, arr):
    arr = [[*image_shape, block_size]] + arr
    f = open(FILENAME, "w")
    for row in arr:
        print(*row, file=f)
    f.close()

def decodeRead():
    f = open(FILENAME, "r")
    arr = []
    for line in f:
        arr.append([int(x) for x in line.split()])
    f.close()
    image_shape = tuple(arr[0][:2])
    block_size = arr[0][2]
    return image_shape, block_size, arr[1:]

BLOCK_SIZE = -1

CODES_SIZE = 9


def encode(BLOCK_SIZE, CODES_SIZE):
    input_img = cv2.imread(IMAGE_NAME,0)

    codes_sent = 0
    max_code = 0

    entropy = 0
    frequency = {}
    for i in range(256):
        frequency[i] = 0
    for i in range(input_img.shape[0]):
        for j in range(input_img.shape[1]):
            frequency[input_img[i][j]]+=1
    # print(float(input_img.shape[0]))
    for i in range(255):
        if(frequency[i]!=0):
            entropy += frequency[i]/(input_img.shape[0]*input_img.shape[1])*np.log2(frequency[i]/(input_img.shape[0]*input_img.shape[1]))

    if(BLOCK_SIZE>1):
        new_block = np.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype='int32')

    block_list = []
    if(BLOCK_SIZE == -1):
        block_list.append(input_img)
    else:
        for i in range(input_img.shape[0]//BLOCK_SIZE):
            for j in range(input_img.shape[1]//BLOCK_SIZE):
                block_list.append(input_img[i*BLOCK_SIZE:(i+1)*BLOCK_SIZE,j*BLOCK_SIZE:(j+1)*BLOCK_SIZE])



    final_data = []

    for i in block_list:
        block_array = []
        for j in range(i.shape[0]):
            for k in range(i.shape[1]):
                block_array.append(i[j][k])

        lzw = []
        count = 256
        dictonary = {}
        for j in range(256):
            dictonary[(j,)] = j
        prev = ()
        for j in block_array:
            current = prev + (j,)
            if current in dictonary:
                prev = current
            else:
                if(dictonary[current[:-1]]>max_code):
                    max_code = dictonary[current[:-1]]
                codes_sent+=1
                lzw.append(dictonary[current[:-1]])
                if(len(dictonary)<2**CODES_SIZE):
                    dictonary[current] = count 
                    count+=1
                prev = (j,)
        if prev in dictonary:
            if(dictonary[current[:-1]]>max_code):
                max_code = dictonary[current[:-1]]
            codes_sent+=1
            lzw.append(dictonary[prev])

        final_data.append(lzw)
    
    encodeWrite(input_img.shape, BLOCK_SIZE, final_data)
    print(f'Maximum code sent is {max_code}.')
    print(f'Number of codes sent are {codes_sent}')
    # print(input_img.shape)
    print(f'Entropy is {-entropy}.')
    print(f'Number of bits for each value is {codes_sent*CODES_SIZE/(input_img.shape[0]*input_img.shape[1])}')
    print(f'Comression ratio achieved is {(input_img.shape[0]*input_img.shape[1]*8)/(codes_sent*CODES_SIZE)}')

def decode(BLOCK_SIZE, CODES_SIZE):
        
    # lzw_encoded = []

    # image_shape = ()

    # BLOCK_SIZE = 8
    image_shape, block_size, lzw_encoded = decodeRead()


    # image = np.zeros(image_shape)

    blocks = []
    for i in lzw_encoded:
        block = []
        dictionary = {}
        for j in range(256):
            dictionary[j] = [j]
        count = 256
        current = []
        for j in i:
            if not(j in dictionary):
                dictionary[j] = current + [current[0]]
            block.append(dictionary[j])
            if (len(current) != 0) :
                dictionary[count] = current + [dictionary[j][0]]
                count+=1
            current = dictionary[j]
        ans = []
        for j in block: ans.extend(j)
        blocks.append(ans)
    
    image = np.zeros(image_shape, dtype='int32')
    
    

    if(block_size>1):
        for i in range(image_shape[0]//block_size):
            for j in range(image_shape[1]//block_size):
                current_block = blocks.pop(0)
                reconstructed_block = np.zeros((block_size,block_size))
                for k in range(block_size):
                    for l in range(block_size):
                        reconstructed_block[k][l] = current_block.pop(0)

                image[i*block_size:(i+1)*block_size,j*block_size:(j+1)*block_size] = reconstructed_block
    else:
        
        for i in range(image_shape[0]):
            for j in range(image_shape[1]):
                image[i][j] = blocks[0].pop(0)
    cv2.imwrite(DECODED_IMAGE_NAME, image.astype('uint8'))
    
            




def main():
    BLOCK_SIZE = int(input('Enter block size: '))
    CODES_SIZE = int(input('Enter codes size: '))
    print("Options: ")
    print("1. LZW Encode")
    print("2. LZW Decode")
    ch = input("\nEnter your choice: ")
    if ch == '1':
        encode(BLOCK_SIZE, CODES_SIZE)
    elif ch == '2':
        decode(BLOCK_SIZE, CODES_SIZE)
    else:
        print("Invalid choice!\nPlease try again.")


if __name__ == "__main__":
    main()