# Pixel-Cube-Algorithm
Minor project for Bachelor of Technology in Computer Science Engineering at Manipal University Jaipur.

## Introduction
Due to the accelerating use of multimedia technologies in IoT applications, images have become a vital means of transmitting large amounts of information. However, ensuring the confidentiality of this data is a significant challenge. Typically, these data are transmitted via public channels by sensor devices, leaving them vulnerable to potential security risks at every stage of communication, from the perception layer to the network layer. In the case of cloud-enabled IoT applications, the sensors collect information from the physical/perception layer and transmit it to the network layer via the Internet. Yet, by interconnecting various cyber-physical devices, these communications become vulnerable to numerous security risks and assaults/attacks. Sensitive data exchanges that happen rapidly are particularly vulnerable to exploitation by adversaries. As a result, there is a pressing need to develop lightweight security measures for IoT applications.

## Proposed Shuffling Technique
The proposed shuffling technique works in the following way: 

We start by traversing through each row of pixels in the image. At every even-numbered row, the pixels of the entire row are shifted to the next alternate row below it. For example, the 8th row will be shifted to the place of the 10th row, the 10th row to the place of the 12th row and so on. One important thing to note here is that the last even row at the end of the image, when shifted, is wrapped around the pixels of the image and moved to the 0th row or the first even row. 

After all the rows have been shifted, the columns are also shifted similarly; every even alternate column like the 0th column, 2nd column, 4th column and so on is shifted to the next alternate column to its right. We also take into account the wrapping of the last column to be shifted to the first column. 

When combining both these steps, we get the complete shuffling algorithm. As we can see from the image given below that row wise and column-wise pixels are shuffled alternatively to produce this jumbled square of pixels. We repeat this process a total of 30 times. 

Padding:
We first check if the number of pixels in the image in the form (m x n), m represents the no. of rows and n represents the no. of columns, where m and n could be:  
            - (\(m = n\)) : no need to add padding  
            - (\(m > n\)): we add padding to the image, m – n no. of pixel rows  
            - (\(m < n\)): we add padding to the image, n – m no. of pixel columns  
    To remove the padding:  
            - If padding was added to rows (\(m > n\)), crop [m - (m - n)] rows from the bottom of the image.  
            - If padding was added to columns (\(m < n\)), crop [n - (n - m)] columns from the right of the image.  

## Detailed Encryption Methodology
The encryption technique we have proposed has a simple manner of flow of data. The reason for that is so that the entire encryption and shuffling technique to add up to be lightweight for the IoT device. The whole encryption process is done in four phases after which we get the cipher image.  
    - The plain image that is to be transferred via an IoT device gets divided into three channels of red, green and blue values of pixels. The three channels are then passed through the Duffing chaotic map with an arbitrary key – “123”, which is responsible for inducing the first round of chaos into the image.  
    - After this, an image of Baboon is chosen, which is divided into three channels of red, green and blue values of pixels. To generate a key, two steps are followed. First, a key matrix is generated based on the size of the baboon image. Then this key matrix along with the RGB pixel values generated from the image are passed through the Arnold’s Cat chaotic map to induce some more chaos.  
    - After the first two phases are executed, an XOR operation is performed between the image from the Duffing map and the key from the Arnold’s Cat chaotic map. This highlights the reason for using a hybrid transform where hybrid refers to the two: Duffing and Arnold’s Cat chaotic maps in the encryption methodology.  
    - The output of the XOR will further be shuffled by the proposed shuffling technique to rearrange and mix the pixels of the coloured image of each channel. This shuffling induces confusion in the final image and also increases the strength of encryption. A detailed explanation of the shuffling technique is given in the previous section.  

![flowchart](https://github.com/user-attachments/assets/bdb28592-3518-4497-9e26-bb27d340ce24)
