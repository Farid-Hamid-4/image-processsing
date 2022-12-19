import sys
import numpy as np
import warp as wp
from PIL import Image

@wp.kernel
def meanGreyscale(input: wp.array(dtype=wp.float32, ndim=2), output: wp.array(dtype=wp.float32, ndim=2), dimensions: wp.array(dtype=int, ndim=1), kernSize: int):
    i, j = wp.tid() # thread index
    image_x = dimensions[0]
    image_y = dimensions[1]
    border = kernSize/2
    pixelValue = float(0.0)

    # check if the index is within valid boundaries, i.e., ensure they aren't edges
    if ((i >= border) and (j >= border) and (i < image_x - border) and (j < image_y - border)):
        for x in range (-border, border+1):
            for y in range(-border, border+1):
                pixelValue += input[i+x][j+y]
        pixelValue = pixelValue/float(kernSize * kernSize)
    else: # the index is at an edge
        left = border
        right = border
        top = border
        bottom = border

        if (i - border < 0): left = i
        if (j - border < 0): top = j
        if (i + border < image_x): right += 1
        elif (i + border > image_x): right = image_x - i
        if (j + border < image_y): bottom += 1
        elif (j + border > image_y): bottom = image_y - j
        
        for x in range(-left, right):
            for y in range(-top, bottom):
                pixelValue += input[i+x][j+y]
        pixelValue = pixelValue/(float(left + right) * float(bottom + top))

    output[i][j] = wp.clamp(pixelValue, 0.0, 255.0)

@wp.kernel
def meanColor(input: wp.array(dtype=wp.float32, ndim=3), output: wp.array(dtype=wp.float32, ndim=3), dimensions: wp.array(dtype=int), kernSize: int):
    i, j, k = wp.tid() # thread index
    image_x = dimensions[0]
    image_y = dimensions[1]
    border = kernSize/2
    pixelValue = float(0.0)

    # check if the index is within valid boundaries, i.e., ensure they aren't edges
    if ((i >= border) and (j >= border) and (i < image_x - border) and (j < image_y - border)):
        for x in range(-border, border+1):
            for y in range(-border, border+1):
                pixelValue += input[i+x][j+y][k]
        pixelValue = pixelValue/float(kernSize * kernSize)
    else: # the index is at an edge
        left = border
        right = border
        top = border
        bottom = border

        if (i - border < 0): left = i
        if (j - border < 0): top = j
        if (i + border < image_x): right += 1
        elif (i + border > image_x): right = image_x - i
        if (j + border < image_y): bottom += 1
        elif (j + border > image_y): bottom = image_y - j

        for x in range(-left, right):
            for y in range(-top, bottom):
                pixelValue += input[i+x][j+y][k]
        pixelValue = pixelValue/(float(left + right) * float(bottom + top))

    output[i][j][k] = wp.clamp(pixelValue, 0.0, 255.0)

@wp.kernel
def unsharpMaskingGreyscale(input: wp.array(dtype=wp.float32, ndim=2), output: wp.array(dtype=wp.float32, ndim=2), dimensions: wp.array(dtype=int), kernSize: int, kConstant: float):
    i, j = wp.tid() # thread index
    image_x = dimensions[0]
    image_y = dimensions[1]
    border = kernSize/2
    pixelValue = float(0.0)

    # check if the index is within valid boundaries, i.e., ensure they aren't edges
    if ((i >= border) and (j >= border) and (i < image_x - border) and (j < image_y - border)):
        for x in range(-border, border+1):
            for y in range(-border, border+1):
                pixelValue += input[i+x][j+y]

        pixelValue = pixelValue/float(kernSize*kernSize)
        pixelValue = input[i][j] + kConstant*(input[i][j] - pixelValue)
    else: # the index is at an edge
        left = border
        right = border
        top = border
        bottom = border

        if (i - border < 0): left = i
        if (j - border < 0): top = j
        if (i + border < image_x): right += 1
        elif (i + border > image_x): right = image_x - i
        if (j + border < image_y): bottom += 1
        elif (j + border > image_y): bottom = image_y - j

        for x in range(-left, right):
            for y in range(-top, bottom):
                pixelValue += input[i+x][j+y]

        pixelValue = pixelValue/(float(left + right) * float(bottom + top))
        pixelValue = input[i][j] + kConstant*(input[i][j] - pixelValue)

    output[i][j] = wp.clamp(pixelValue, 0.0, 255.0)

@wp.kernel
def unsharpMaskingColor(input: wp.array(dtype=wp.float32, ndim=3), output: wp.array(dtype=wp.float32, ndim=3), dimensions: wp.array(dtype=int), kernSize: int, kConstant: float):
    i, j, k = wp.tid() # thread index
    image_x = dimensions[0]
    image_y = dimensions[1]
    border = kernSize/2
    pixelValue = float(0.0)

    # check if the index is within valid boundaries, i.e., ensure they aren't edges
    if ((i >= border) and (j >= border) and (i < image_x - border) and (j < image_y - border)):
        for x in range(-border, border+1):
            for y in range(-border, border+1):
                pixelValue += input[i+x][j+y][k]

        pixelValue = pixelValue/float(kernSize*kernSize)
        pixelValue = input[i][j][k] + kConstant*(input[i][j][k] - pixelValue)
    else: # the index is at an edge
        left = border
        right = border
        top = border
        bottom = border

        if (i - border < 0): left = i
        if (j - border < 0): top = j
        if (i + border < image_x): right += 1
        elif (i + border > image_x): right = image_x - i
        if (j + border < image_y): bottom += 1
        elif (j + border > image_y): bottom = image_y - j

        for x in range(-left, right):
            for y in range(-top, bottom):
                pixelValue += input[i+x][j+y][k]

        pixelValue = pixelValue/(float(left + right) * float(bottom + top))
        pixelValue = input[i][j][k] + kConstant*(input[i][j][k] - pixelValue)

    output[i][j][k] = wp.clamp(pixelValue, 0.0, 255.0)

def main():
    # command line arguments verifier
    if(len(sys.argv) > 6):
        print("Too many command line argument. Try again")
        exit(1)
    if(sys.argv[1] == "-s"): flag = "-s"
    elif (sys.argv[1] == "-n"): flag = "-n"
    else: 
        print("Invalid algType. Try again")
        exit(1)
    if(int(sys.argv[2]) > 0 and int(sys.argv[2]) % 2 != 0): kernSize = int(sys.argv[2])
    param = float(sys.argv[3])
    inFileName = sys.argv[4]
    outFileName = sys.argv[5]

    # setup
    wp.init()
    device = "cpu"
    image = Image.open(inFileName)
    imageFormat = image.format
    imageMode = image.mode
    if(imageMode == 'RGBA'): image = image.convert('RGB')
    numpyArr = np.asarray(image, dtype=wp.float32)
    imageShape = numpyArr.shape
    if(imageMode == 'L'):
        dim = (imageShape[0], imageShape[1])
    else:
        dim = (imageShape[0], imageShape[1], imageShape[2])
    inWarpData = wp.array(numpyArr, dtype=wp.float32, device=device)
    outWarpData = wp.zeros(shape=imageShape, dtype=wp.float32, device=device)
    dimensions = wp.array(imageShape, dtype=wp.int32, device=device)

    # noise removal
    if(flag == '-n'):
        # greyscale mode
        if(imageMode == 'L'):
            wp.launch(kernel=meanGreyscale, dim=(dim[0], dim[1]), inputs=[inWarpData, outWarpData, dimensions, kernSize], device=device)
        # colored mode
        elif(imageMode == 'RGB' or imageMode == 'RGBA'):
            wp.launch(kernel=meanColor, dim=(dim[0], dim[1], dim[2]), inputs=[inWarpData, outWarpData, dimensions, kernSize], device=device)
    # sharpening
    elif(flag == '-s'):
        # greyscale mode
        if(imageMode == 'L'):
            wp.launch(kernel=unsharpMaskingGreyscale, dim=(dim[0], dim[1]), inputs=[inWarpData, outWarpData, dimensions, kernSize, param], device=device)
        # colored mode
        elif(imageMode == 'RGB' or imageMode == 'RGBA'):
            wp.launch(kernel=unsharpMaskingColor, dim=(dim[0], dim[1], dim[2]), inputs=[inWarpData, outWarpData, dimensions, kernSize, param], device=device)

    numpyOutArr = outWarpData.numpy()
    imageOut = Image.fromarray(np.uint8(numpyOutArr))
    imageOut.save(outFileName)

if __name__ == '__main__':
    main()