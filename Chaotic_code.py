import cv2
import numpy as np
import random
from PIL import Image
from numpy import *

# ENCRYPTION

def duffing(d_img_path, d_key, alpha=2.75, beta=0.2):
    d_img = np.array(Image.open(d_img_path), dtype=np.uint8)
    d_img_size = d_img.shape[:2]
    indices = np.arange(d_img_size[0]*d_img_size[1])
    np.random.seed(int(d_key))
    np.random.shuffle(indices)
    x =  0.1
    y =  0.1
    for i in range(d_img_size[0]):
        for j in range(d_img_size[1]):
            _x = y
            _y = -beta * x + alpha * y - y ** 3
            x = _x
            y = _y
            index = indices[i*d_img_size[1]+j]
            px, py = index//d_img_size[1], index%d_img_size[1]
            d_img[i, j] = d_img[px, py]
    return Image.fromarray(d_img)

def arnolds_cat_map(rows, cols):
    key_map = np.zeros((rows, cols, 2), dtype=int)
    # Initial permutation parameters
    a, b = 2, 3
    for i in range(rows):
        for j in range(cols):
            # Arnold's cat map transformation
            x = (i + j * a) % rows
            y = (i * b + j * (a * b + 1)) % cols
            key_map[i][j] = [x, y]
    return key_map

def generateKeyMatrix(N):
    key_matrix = np.zeros((N, N), dtype=np.uint8)
    key_map = arnolds_cat_map(N, N)  # Generating the Arnold's cat map
    for i in range(N):
        for j in range(N):
            # Assigning Arnold's cat map coordinates as key matrix values
            key_matrix[i][j] = key_map[i][j][0] ^ key_map[i][j][1]  # XOR operation of coordinates
    return key_matrix

def arnold(new_key_img, key):
    new_key_img = cv2.imread(new_key_img)
    # key = cv2.imread(key)   
    N = new_key_img.shape[0]
    for i in range(N):
        for j in range(N):
            for  k in range(3):
                # new_key_img[i][j][k] = new_key_img[i][j][k] ^ key[i][j][k]      #XOR
                new_key_img[i][j][k] = new_key_img[i][j][k] ^ key[i][j]
    array = np.array(new_key_img, dtype=np.uint8)
    x,y = meshgrid(range(N),range(N))
    xmap = (x + y) % N
    ymap = (x + 2*y) % N
    for i in range(int(N/16)):
        result = Image.fromarray(array)
        array = array[xmap,ymap]
    result.save("arnold_encrypted.png")
    return result

def xor(img1, img2):
    # Load two color images
    img1 = cv2.imread("duffing_encrypted.png")
    img2 = cv2.imread("arnold_encrypted.png")

    # Split the color channels of the images
    b1, g1, r1 = cv2.split(img1)
    b2, g2, r2 = cv2.split(img2)

    # XOR the corresponding color channels of the two images
    xor_b = cv2.bitwise_xor(b1, b2)
    xor_g = cv2.bitwise_xor(g1, g2)
    xor_r = cv2.bitwise_xor(r1, r2)

    # Merge the XORed color channels into a single color image
    xor_img = cv2.merge((xor_b, xor_g, xor_r))

    # Save the XOR image to a file
    cv2.imwrite("xor_img.png", xor_img)
    return xor_img

def shuffle(shuffle_input):
    # SHUFFLE
    for x in range(30):
        # Row shuffle
        for i in range(shuffle_input.shape[0]):    # traversing through each row 
            if i%2 == 0 and i<shuffle_input.shape[0]-2:
                shuffle_input[[i,i+2]] = shuffle_input[[i+2,i]]  #replacing the (i+2) row with the i row 
                # a,b = b,a

        # Col shuffle
        for i in range(shuffle_input.shape[1]):    # traversing through each column
            if i%2 == 0 and i<shuffle_input.shape[1]-2:
                shuffle_input[:,[i,i+2]] = shuffle_input[:,[i+2,i]] 
    cv2.imwrite("shuffled_img.png", shuffle_input)
    return shuffle_input

# DECRYPTION

def unshuffle(shuffled_img):
    # UNSHUFFLE
    for x in range(30):
        # Col unshuffle
        for i in list(reversed(range(shuffled_img.shape[1]))):
            if i%2 == 0 and i>1:
                shuffled_img[:,[i,i-2]] = shuffled_img[:,[i-2,i]] 

        # Row unshuffle
        for i in list(reversed(range(shuffled_img.shape[0]))):
            if i%2 == 0 and i>1:
                shuffled_img[[i,i-2]] = shuffled_img[[i-2,i]]  
    cv2.imwrite("unshuffled_img.png", shuffled_img)
    return shuffled_img

def xor_rev(unshuffled_img, img2):
    img2 = cv2.imread(img2)
    
    # Split the images into their color channels
    b1, g1, r1 = cv2.split(unshuffled_img)
    b2, g2, r2 = cv2.split(img2)

     # Perform XOR operation on the color channels
    b = cv2.bitwise_xor(b1, b2)
    g = cv2.bitwise_xor(g1, g2)
    r = cv2.bitwise_xor(r1, r2)

    # Merge the color channels back into a single image
    result = cv2.merge((b, g, r))

    # Save the recovered image to a file
    cv2.imwrite("recovered_img.png", result)
    return result 

def duffing_rev(recovered_img_path, d_key, alpha=2.75, beta=0.2):
    d_img = np.array(Image.open(recovered_img_path), dtype=np.uint8)
    d_img_size = d_img.shape[:2]
    indices = np.arange(d_img_size[0]*d_img_size[1])
    np.random.seed(int(d_key))
    np.random.shuffle(indices)
    x =  0.1
    y =  0.1
    for i in range(d_img_size[0]-1, -1, -1):
        for j in range(d_img_size[1]-1, -1, -1):
            index = indices[i*d_img_size[1]+j]
            px, py = index//d_img_size[1], index%d_img_size[1]
            d_img[px, py] = d_img[i, j]
    return Image.fromarray(d_img)

def arnold_rev(img, key):
    img = cv2.imread(img)
    new_image = array(img)
    N = new_image.shape[0]
    x,y = meshgrid(range(N),range(N))
    xmap = (x + y) % N
    ymap = (x + 2*y) % N
    for i in range(1 + N - int(N/16)):
        new_image = new_image[xmap,ymap]
    for i in range(N):
        for j in range(N):
            for k in range(3):
                new_image[i][j][2-k] = new_image[i][j][2-k] ^ key[i][j]
    im = Image.fromarray(new_image)
    arr = np.array(im, dtype=np.uint8)
    result = Image.fromarray(arr)
    result.save('arnold_decrypted.png')
    decipher = cv2.imread("arnold_decrypted.png")
    # decipher = deshuffleMagicSquare(N, cipher)
    cv2.imwrite('arnold_decrypted.png', decipher)
    return decipher


# MAIN FUNCTION

if __name__ == '__main__':

    # KEY IMAGE
    # for duffing
    d_key = 123
    d_img_path = 'new_key_img.png'

    # for arnold
    N = 512  # Adjust the size of the key matrix as needed
    new_key_img = 'new_key_img.png'
    key = generateKeyMatrix(N)
    np.savetxt("key_matrix.txt", key, fmt='%d')

    # Encryption - Duffing and Arnold
    encrypted_img_path = 'duffing_encrypted.png'
    encrypted_img = duffing(d_img_path, d_key)
    encrypted_img.save(encrypted_img_path)
    cipher = arnold(new_key_img, key)

    # XOR
    img1 = 'duffing_encrypted.png'
    img2 = 'arnold_encrypted.png' # key
    xor_img = xor(img1, img2)

    # Decryption

    img_path = 'arnold_encrypted.png'
    img = cv2.imread(img_path)
    N = img.shape[0]
    decipher = arnold_rev(img_path, key)

    # Shuffling
    xor_img_path = 'xor_img.png'
    shuffle_input = cv2.imread(xor_img_path)
    shuffled_img = shuffle(shuffle_input)

    # Unshuffling
    unshuffled_img = unshuffle(shuffled_img)
    cv2.imshow('Unshuffled Image', unshuffled_img)

    # # XOR_rev (Recovered)
    recovered_img = xor_rev(unshuffled_img, img2) 
    recovered_img_path = 'recovered_img.png'
    cv2.imwrite(recovered_img_path, recovered_img)

    # # Duffing_rev
    decrypted_img_path = 'duffing_decrypted.png'
    decrypted_img = duffing_rev(recovered_img_path, d_key)
    decrypted_img.save(decrypted_img_path)


