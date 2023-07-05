import cv2
import numpy as np
# from IPython.display import Image


IMAGE_NAME = "./kodim01.png"
OUTPUT_NAME = "Decoded_Image.tif"

# BLOCK_SIZE = 8

# COEFFICIENT_SENT = 1

def encodeWrite(FILENAME, image_shape, block_size, arr):
    arr = [[*image_shape, block_size]] + arr
    f = open(FILENAME, "w")
    for row in arr:
        print(*row, file=f)
    f.close()

def decodeRead(FILENAME):
    f = open(FILENAME, "r")
    arr = []
    for line in f:
        try:
            arr.append([int(x) for x in line.split()])
        except:
            arr.append([x for x in line.split()])
    f.close()
    image_shape = tuple(arr[0][:2])
    block_size = arr[0][2]
    return image_shape, block_size, arr[1:]

QUANTIZATION_MATRIX_UV = np.array([
    [17,  18,  24,  47,  99,  99,  99,  99],
    [18,  21,  26,  66,  99,  99,  99,  99],
    [24,  26,  56,  99,  99,  99,  99,  99],
    [47,  66,  99,  99,  99,  99,  99,  99],
    [99,  99,  99,  99,  99,  99,  99,  99],
    [99,  99,  99,  99,  99,  99,  99,  99],
    [99,  99,  99,  99,  99,  99,  99,  99],
    [99,  99,  99,  99,  99,  99,  99,  99]])
    
def encode(BLOCK_SIZE, COEFFICIENT_SENT, BW):
    FILENAME = "compressed_jpeg.txt"
    

    # plt.imshow(input_img, cmap="gray")
    def encodeCore(input_img, QUANTIZATION_MATRIX, BLOCK_SIZE, COEFFICIENT_SENT):
        padded_img = np.zeros((np.array(input_img.shape)+BLOCK_SIZE-1)//BLOCK_SIZE*BLOCK_SIZE, dtype='int32')

        # print(input_img.shape)
        # print(padded_img.shape)

        padded_img[0:input_img.shape[0],0:input_img.shape[1]] = input_img

        # print(padded_img)

        # plt.imshow(padded_img, cmap="gray")

        new_block = np.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype='int32')

        block_list = []
        for i in range(padded_img.shape[0]//BLOCK_SIZE):
            for j in range(padded_img.shape[1]//BLOCK_SIZE):
                block_list.append(padded_img[i*BLOCK_SIZE:(i+1)*BLOCK_SIZE,j*BLOCK_SIZE:(j+1)*BLOCK_SIZE])

        # len(block_list)

        level_shifted_blocks = []
        for i in block_list:
            level_shifted_blocks.append(i-128)

        dct_blocks = []
        for i in level_shifted_blocks:
            dct_blocks.append(cv2.dct(i.astype('float64')))

        quantized_blocks = []
        for i in dct_blocks:
            quantized_blocks.append(np.rint(np.divide(i,cv2.resize(QUANTIZATION_MATRIX, dsize=(BLOCK_SIZE, BLOCK_SIZE), interpolation=cv2.INTER_NEAREST))))

        def zigzag(arr):
            dct_coefficients = [arr[0][0]]
            x,y=0,1
            direction = 1
            while(len(dct_coefficients)<BLOCK_SIZE*BLOCK_SIZE):
                dct_coefficients.append(arr[x][y])
                if(direction == 1):
                    x+=1
                    y-=1
                elif(direction == 0):
                    x-=1
                    y+=1
                if(x==BLOCK_SIZE-1):
                    dct_coefficients.append(arr[x][y])
                    y+=1
                    direction = 1 - direction
                elif(y==BLOCK_SIZE-1):
                    dct_coefficients.append(arr[x][y])
                    x+=1
                    direction = 1 - direction
                if(x==0):
                    dct_coefficients.append(arr[x][y])
                    y+=1
                    direction = 1 - direction
                elif(y==0):
                    dct_coefficients.append(arr[x][y])
                    x+=1
                    direction = 1 - direction
            return dct_coefficients


        def remove_trail(list):
            if(list[-1]==0):
                while(len(list)>0 and list[-1]==0):
                    list.pop()
                list.append('EOB')
            return list

        def limitCoefficients(list):
            data = [0]*BLOCK_SIZE*BLOCK_SIZE
            for i in range(COEFFICIENT_SENT):
                data[i] = list[i]
            return data

        final_block_array = []
        for i in quantized_blocks:
            final_block_array.append(remove_trail(limitCoefficients(zigzag(i))))
        return final_block_array
    
    QUANTIZATION_MATRIX = np.array([
        [16,11,10,16,24,40,51,61],
        [12,12,14,19,26,58,60,55],
        [14,13,16,24,40,57,69,56],
        [14,17,22,29,51,87,80,62],
        [18,22,37,56,68,109,103,77],
        [24,35,55,64,81,104,113,92],
        [49,64,78,87,103,121,120,101],
        [72,92,95,98,112,100,103,99]
    ])

    if(BW == 0):
        input_img = cv2.imread(IMAGE_NAME,0)
        final = encodeCore(input_img, QUANTIZATION_MATRIX, BLOCK_SIZE, COEFFICIENT_SENT)
        print(final)
        encodeWrite(FILENAME, (np.array(input_img.shape)+BLOCK_SIZE-1)//BLOCK_SIZE*BLOCK_SIZE , BLOCK_SIZE, final)
    else:
        input_img = cv2.imread(IMAGE_NAME, cv2.IMREAD_COLOR)
        yuv_image = cv2.cvtColor(input_img, cv2.COLOR_BGR2YCrCb)
        y, u, v = cv2.split(yuv_image)
        y_code = encodeCore(y, QUANTIZATION_MATRIX, BLOCK_SIZE, COEFFICIENT_SENT)
        u_code = encodeCore(u, QUANTIZATION_MATRIX_UV, BLOCK_SIZE, COEFFICIENT_SENT)
        v_code = encodeCore(v, QUANTIZATION_MATRIX_UV, BLOCK_SIZE, COEFFICIENT_SENT)
        print(f'y code = {y_code}\nu code = {u_code}\nv code = {v_code}')
        encodeWrite('y_'+FILENAME, (np.array(input_img.shape)+BLOCK_SIZE-1)//BLOCK_SIZE*BLOCK_SIZE, BLOCK_SIZE, y_code)
        encodeWrite('u_'+FILENAME, (np.array(input_img.shape)+BLOCK_SIZE-1)//BLOCK_SIZE*BLOCK_SIZE, BLOCK_SIZE, u_code)
        encodeWrite('v_'+FILENAME, (np.array(input_img.shape)+BLOCK_SIZE-1)//BLOCK_SIZE*BLOCK_SIZE, BLOCK_SIZE, v_code)
        


def decode(BLOCK_SIZE, COEFFICIENT_SENT, BW):
    

    FILENAME = "compressed_jpeg.txt"
    # img_shape = (256,256)

    # final_block_array = decodeRead()
    

    def dencodeCore(img_shape, BLOCK_SIZE, final_block_array, QUANTIZATION_MATRIX, COEFFICIENT_SENT):
        def addTrail(list):
            if(list[-1]=='EOB'):
                list.pop()
                # print(list)
                while(len(list)<BLOCK_SIZE*BLOCK_SIZE):
                    list.append(0)
            return list

        def unZigZag(coefficient_list):
            block = np.zeros((BLOCK_SIZE, BLOCK_SIZE))
            block[0][0] = coefficient_list.pop(0)
            x,y=0,1
            direction = 1
            while(len(coefficient_list)>0):
                block[x][y] = coefficient_list.pop(0)
                if(direction == 1):
                    x+=1
                    y-=1
                elif(direction == 0):
                    x-=1
                    y+=1
                if(x==BLOCK_SIZE-1):
                    block[x][y] = coefficient_list.pop(0)
                    y+=1
                    direction = 1 - direction
                elif(y==BLOCK_SIZE-1):
                    block[x][y] = coefficient_list.pop(0)
                    x+=1
                    direction = 1 - direction
                if(x==0):
                    block[x][y] = coefficient_list.pop(0)
                    y+=1
                    direction = 1 - direction
                elif(y==0):
                    block[x][y] = coefficient_list.pop(0)
                    x+=1
                    direction = 1 - direction
            return block

        de_quantized_blocks = []
        for i in final_block_array:
            de_quantized_blocks.append(unZigZag(addTrail(i)))

        quantized_blocks = []
        for i in de_quantized_blocks:
            quantized_blocks.append(np.multiply(i, cv2.resize(QUANTIZATION_MATRIX, dsize=(BLOCK_SIZE, BLOCK_SIZE), interpolation=cv2.INTER_NEAREST)))

        un_dct_blocks = []
        for i in quantized_blocks:
            un_dct_blocks.append(cv2.idct(i))

        levelized_blocks = []
        for i in un_dct_blocks:
            levelized_blocks.append(i+128)
        # print(len(levelized_blocks))
        image = np.zeros(img_shape, dtype='int32')

        # print(img_shape, BLOCK_SIZE)
        for i in range(img_shape[0]//BLOCK_SIZE):
            for j in range(img_shape[1]//BLOCK_SIZE):
                image[i*BLOCK_SIZE:(i+1)*BLOCK_SIZE,j*BLOCK_SIZE:(j+1)*BLOCK_SIZE] = levelized_blocks.pop(0)
        # print(image.shape)

        return image

    QUANTIZATION_MATRIX = np.array([
        [16,11,10,16,24,40,51,61],
        [12,12,14,19,26,58,60,55],
        [14,13,16,24,40,57,69,56],
        [14,17,22,29,51,87,80,62],
        [18,22,37,56,68,109,103,77],
        [24,35,55,64,81,104,113,92],
        [49,64,78,87,103,121,120,101],
        [72,92,95,98,112,100,103,99]
    ])

    if(BW == 0):
        img_shape, BLOCK_SIZE, final_block_array = decodeRead(FILENAME)
        data_sent = 0
        for i in final_block_array:
            data_sent+=len(i)
        image = dencodeCore(img_shape, BLOCK_SIZE, final_block_array, QUANTIZATION_MATRIX, COEFFICIENT_SENT)
        cv2.imwrite("Decoded_Image.tif", image.astype('uint8'))
        origional_image = cv2.imread(IMAGE_NAME,0)
        padded_origional_img = np.zeros((np.array(origional_image.shape)+BLOCK_SIZE-1)//BLOCK_SIZE*BLOCK_SIZE, dtype='int32')
        padded_origional_img[0:origional_image.shape[0],0:origional_image.shape[1]] = origional_image
        mse = np.square(np.subtract(image,padded_origional_img)).mean() 
        rmse = np.sqrt(mse)

        print(f"RMSE obatined is {rmse}.")

        psnr = 20*np.log10(255/rmse)

        print(f"Obtained PSNR is {psnr}.")

        print(f"Number of coefficients sent are {COEFFICIENT_SENT}")

        print(f"Comression ratio obtained is {(img_shape[0]*img_shape[1])/data_sent}")
    else:
        img_shape, BLOCK_SIZE, y_arr = decodeRead('y_'+FILENAME)
        img_shape, BLOCK_SIZE, u_arr = decodeRead('u_'+FILENAME)
        img_shape, BLOCK_SIZE, v_arr = decodeRead('v_'+FILENAME)
        data_sent = 0
        for i in y_arr:
            data_sent+=len(i)
        for i in u_arr:
            data_sent+=len(i)
        for i in v_arr:
            data_sent+=len(i)
        y_img = dencodeCore(img_shape, BLOCK_SIZE, y_arr, QUANTIZATION_MATRIX, COEFFICIENT_SENT)
        u_img = dencodeCore(img_shape, BLOCK_SIZE, u_arr, QUANTIZATION_MATRIX_UV, COEFFICIENT_SENT)
        v_img = dencodeCore(img_shape, BLOCK_SIZE, v_arr, QUANTIZATION_MATRIX_UV, COEFFICIENT_SENT)
        merged = cv2.merge((y_img.astype('uint8'), u_img.astype('uint8'), v_img.astype('uint8')))
        bgr_image = cv2.cvtColor(merged, cv2.COLOR_YCrCb2BGR)
        final_b, final_g, final_r = cv2.split(bgr_image)
        cv2.imwrite(OUTPUT_NAME, bgr_image)

        origional_image = cv2.imread(IMAGE_NAME,cv2.IMREAD_COLOR)
        origional_b, origional_g, origional_r = cv2.split(origional_image)
        padded_b = np.zeros((np.array(origional_b.shape)+BLOCK_SIZE-1)//BLOCK_SIZE*BLOCK_SIZE, dtype='int32')
        padded_b[0:origional_b.shape[0],0:origional_b.shape[1]] = origional_b
        padded_g = np.zeros((np.array(origional_g.shape)+BLOCK_SIZE-1)//BLOCK_SIZE*BLOCK_SIZE, dtype='int32')
        padded_g[0:origional_g.shape[0],0:origional_g.shape[1]] = origional_g
        padded_r = np.zeros((np.array(origional_r.shape)+BLOCK_SIZE-1)//BLOCK_SIZE*BLOCK_SIZE, dtype='int32')
        padded_r[0:origional_r.shape[0],0:origional_r.shape[1]] = origional_r
        
        error_b = np.square(np.subtract(padded_b,final_b))
        error_g = np.square(np.subtract(padded_g,final_g))
        error_r = np.square(np.subtract(padded_r,final_r))

        mse = (error_b + error_g + error_r).mean()
        rmse = np.sqrt(mse)

        print(f"RMSE obatined is {rmse}.")

        psnr = 20*np.log10(255/rmse)

        print(f"Obtained PSNR is {psnr}.")

        print(f"Number of coefficients sent are {COEFFICIENT_SENT}")

        print(f"Comression ratio obtained is {(img_shape[0]*img_shape[1])/data_sent}")



def main():
    BLOCK_SIZE = int(input('Enter block size: '))
    # print(BLOCK_SIZE)
    COEFFICIENT_SENT = int(input('Enter numebr of coefficients to be sent for each block: '))
    BW = int(input('Enter 0 for B&W and 1 for colored:'))
    print("Options: ")
    print("1. Encode")
    print("2. Decode")
    ch = input("\nEnter your choice: ")
    if ch == '1':
        encode(BLOCK_SIZE, COEFFICIENT_SENT, BW)
    elif ch == '2':
        decode(BLOCK_SIZE, COEFFICIENT_SENT, BW)
    else:
        print("Invalid choice!\nPlease try again.")



if __name__ == "__main__":
    main()
