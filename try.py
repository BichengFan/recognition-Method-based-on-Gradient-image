import skimage
import numpy as np
from PIL import Image
from skimage import data
import matplotlib.pyplot as plt
import matplotlib.image as mping
from collections import deque

# improved RGA
def region_growing_alo(gradientMatrix, coordinate_Matrix, present_seed):
    queue_seed = deque([present_seed])
    temp_segmentation_ele = [present_seed]
    while queue_seed:
    # print(queue_seed)
        present_seed_centre = queue_seed.popleft()
        print("present_seed_centre", present_seed_centre)
        if present_seed_centre[0] + 1 < len(gradientMatrix):
            if [present_seed_centre[0] + 1, present_seed_centre[1]] in coordinate_Matrix and gradientMatrix[present_seed_centre[0] + 1][present_seed_centre[1]] == 255:  # and possessed_Matrix[present_seed_centre[0] + 1][present_seed_centre[1]] == 0
                queue_seed.append([present_seed_centre[0] + 1, present_seed_centre[1]])
                temp_segmentation_ele.append([present_seed_centre[0] + 1, present_seed_centre[1]])
                coordinate_Matrix.remove([present_seed_centre[0] + 1, present_seed_centre[1]])

        if present_seed_centre[0] + 1 < len(gradientMatrix) and present_seed_centre[1] + 1 < len(gradientMatrix[present_seed_centre[0] + 1]):
            if [present_seed_centre[0] + 1, present_seed_centre[1] + 1] in coordinate_Matrix and gradientMatrix[present_seed_centre[0] + 1][present_seed_centre[1] + 1] == 255:  # and possessed_Matrix[present_seed_centre[0] + 1][present_seed_centre[1] + 1] == 0
                queue_seed.append([present_seed_centre[0] + 1, present_seed_centre[1] + 1])
                temp_segmentation_ele.append([present_seed_centre[0] + 1, present_seed_centre[1] + 1])
                coordinate_Matrix.remove([present_seed_centre[0] + 1, present_seed_centre[1] + 1])

        if present_seed_centre[1] + 1 < len(gradientMatrix[present_seed_centre[0]]):
            if [present_seed_centre[0], present_seed_centre[1] + 1] in coordinate_Matrix and gradientMatrix[present_seed_centre[0]][present_seed_centre[1] + 1] == 255:  # and possessed_Matrix[present_seed_centre[0]][present_seed_centre[1] + 1] == 0
                queue_seed.append([present_seed_centre[0], present_seed_centre[1] + 1])
                temp_segmentation_ele.append([present_seed_centre[0], present_seed_centre[1] + 1])
                coordinate_Matrix.remove([present_seed_centre[0], present_seed_centre[1] + 1])

        if present_seed_centre[0] - 1 > 0 and present_seed_centre[1] + 1 < len(gradientMatrix[present_seed_centre[0] - 1]):
            if [present_seed_centre[0] - 1, present_seed_centre[1] + 1] in coordinate_Matrix and gradientMatrix[present_seed_centre[0] - 1][present_seed_centre[1] + 1] == 255:  # and possessed_Matrix[present_seed_centre[0] - 1][present_seed_centre[1] + 1] == 0
                queue_seed.append([present_seed_centre[0] - 1, present_seed_centre[1] + 1])
                temp_segmentation_ele.append([present_seed_centre[0] - 1, present_seed_centre[1] + 1])
                coordinate_Matrix.remove([present_seed_centre[0] - 1, present_seed_centre[1] + 1])

        if present_seed_centre[0] - 1 > 0:
            if [present_seed_centre[0] - 1, present_seed_centre[1]] in coordinate_Matrix and gradientMatrix[present_seed_centre[0] - 1][present_seed_centre[1]] == 255:  # and possessed_Matrix[present_seed_centre[0] - 1][present_seed_centre[1]] == 0
                queue_seed.append([present_seed_centre[0] - 1, present_seed_centre[1]])
                temp_segmentation_ele.append([present_seed_centre[0] - 1, present_seed_centre[1]])
                coordinate_Matrix.remove([present_seed_centre[0] - 1, present_seed_centre[1]])

        if present_seed_centre[0] - 1 > 0 and present_seed_centre[1] - 1 > 0:
            if [present_seed_centre[0] - 1, present_seed_centre[1] - 1] in coordinate_Matrix and gradientMatrix[present_seed_centre[0] - 1][present_seed_centre[1] - 1] == 255:  # possessed_Matrix[present_seed_centre[0] - 1][present_seed_centre[1] -1] == 0
                queue_seed.append([present_seed_centre[0] - 1, present_seed_centre[1] - 1])
                temp_segmentation_ele.append([present_seed_centre[0] - 1, present_seed_centre[1] - 1])
                coordinate_Matrix.remove([present_seed_centre[0] - 1, present_seed_centre[1] - 1])

        if present_seed_centre[1] - 1 > 0:
            if [present_seed_centre[0], present_seed_centre[1] - 1] in coordinate_Matrix and gradientMatrix[present_seed_centre[0]][present_seed_centre[1] - 1] == 255:  # and possessed_Matrix[present_seed_centre[0]][present_seed_centre[1] -1] == 0
                queue_seed.append([present_seed_centre[0], present_seed_centre[1] - 1])
                temp_segmentation_ele.append([present_seed_centre[0], present_seed_centre[1] - 1])
                coordinate_Matrix.remove([present_seed_centre[0], present_seed_centre[1] - 1])

        if present_seed_centre[0] + 1 < len(gradientMatrix) and present_seed_centre[1] - 1 > 0:
            if [present_seed_centre[0] + 1, present_seed_centre[1] - 1] in coordinate_Matrix and gradientMatrix[present_seed_centre[0] + 1][present_seed_centre[1] - 1] == 255:  # and possessed_Matrix[present_seed_centre[0] + 1][present_seed_centre[1] -1] == 0
                queue_seed.append([present_seed_centre[0] + 1, present_seed_centre[1] - 1])
                temp_segmentation_ele.append([present_seed_centre[0] + 1, present_seed_centre[1] - 1])
                coordinate_Matrix.remove([present_seed_centre[0] + 1, present_seed_centre[1] - 1])
    return temp_segmentation_ele

# generate structuring element
def generate_element():
    str_element = [[a, b] for a in [-1, 0, 1] for b in [-1, 0, 1]]
    return str_element

# dilate function according to its definition
def erode(matrix, coord_list, str_element, str_coord):
      erode_list = []
      for x in coord_list:
         temp_value = []
         for y in str_coord:
             if x[0] + y[0] >= 0 and x[0] + y[0] < len(matrix) and x[1] + y[1] >= 0 and x[1] + y [1] < len(matrix[x[0] + y[0]]):
                 temp_value.append(matrix[x[0] + y[0]][x[1] + y[1]] - str_element[y[0] + 1][y[1] + 1])
         erode_list.append(min(temp_value))
      return erode_list

# erode fuction according to its definition
def dilate(matrix, coord_list, str_element, str_coord):
    dilate_list = []
    for x in coord_list:
        temp_value = []
        for y in str_coord:
            if x[0] - y[0] >= 0 and x[0] - y[0] < len(matrix) and x[1] - y[1] >= 0 and x[1] - y[1] < len(matrix[x[0] - y[0]]):
                temp_value.append(matrix[x[0] - y[0]][x[1] - y[1]] + str_element[y[0] + 1][y[1] + 1])
        dilate_list.append(max(temp_value))
    return dilate_list

# list to matrix
def list_matrix(list, height, width):
  matrix = []
  for x in range(height):
      line = []
      for y in range(width):
         line.append(list[x * width + y])
      matrix.append(line)
  return matrix

# first step get 8 connected area
def getadjacentcells(xCoord, yCoord, grayMatrix):
    connectedarea = []
    for x, y in [(xCoord + i, yCoord + j) for i in (-1, 0, 1) for j in (-1, 0, 1)]:
        if x >= 0 and y >= 0 and x < len(grayMatrix) and y < len(grayMatrix[0]):
            connectedarea.append([x, y, grayMatrix[x][y]])
    return connectedarea

# open file
im1 = data.imread(r'G:\pythoncode\python3.5\\recongnitionMethodbasedonGradientImg\\11.png')

# i = 63
# image2 = []
# while i < 326:
#     image2.append(im1[i])
#     i += 1
#
# img_data = np.array(image2)
# img_show = Image.fromarray(img_data)
# img_show.show()
# img_show.close()

# grey image
im2 = skimage.color.rgb2grey(im1)
grayMatrix = 256 * im2
grayMatrix_cp = grayMatrix.copy()
# ready for morphology
coord_list = [[x, y] for x in range(len(grayMatrix)) for y in range(len(grayMatrix[0]))]
str_element = [[0, 2, 0], [2, 3, 2], [0, 2, 0]]
str_coord = generate_element()

# obtain dilate matrix and erode matrix
dilateList = dilate(grayMatrix, coord_list, str_element, str_coord)
erodeList = erode(grayMatrix, coord_list, str_element, str_coord)

# obtain dilate matrix and erode matrix
dilateMatrix = list_matrix(dilateList, len(grayMatrix), len(grayMatrix[0]))
erodeMatrix = list_matrix(erodeList, len(grayMatrix), len(grayMatrix[0]))

# enhance edge
generateMatrix = [[0 for col in range(0,  len(grayMatrix[0]))] for raw in range(0,len(grayMatrix))]
for x in range(0, len(grayMatrix)):
    for y in range(0, len(grayMatrix[0])):
        if dilateMatrix[x][y] - erodeMatrix[x][y] > 5:
            if grayMatrix[x][y] >= (dilateMatrix[x][y] + erodeMatrix[x][y]) / 2:
                generateMatrix[x][y] = dilateMatrix[x][y]
            else:
                generateMatrix[x][y] = erodeMatrix[x][y]
        else:
            generateMatrix[x][y] = np.float64(0)

# display enhanced pic
generateMatrix_cpy = generateMatrix.copy()
generateMatrix_cpy1 = np.array(generateMatrix_cpy, dtype = 'uint8')
img_cpy = Image.fromarray(generateMatrix_cpy1)
img_cpy.save('22_enhanced.png')
img_cpy.close()
im3 = mping.imread('22_enhanced.png')
implot3 = plt.imshow(im3)
plt.show()
plt.close()
# calculate central pixel's third central moment
gradientLists = []
coordLists = [[y, x, 0] for y in range(len(grayMatrix)) for x in range(len(grayMatrix[0]))]
for x in coordLists:
    connectedarea = getadjacentcells(x[0], x[1], grayMatrix)
    sum = 0
    for x in connectedarea:
        sum += x[2]
    temp_mean = sum / len(connectedarea)
    # 3rd central moment
    sum = 0
    for y in connectedarea:
        sum += (y[2] - temp_mean) ** 3
    nthmoment = sum / len(connectedarea)
    if nthmoment < 0:
        gradientLists.append(np.float64(0))
        x[2] = 1
    elif nthmoment > 255:
        gradientLists.append(np.float64(255))
        x[2] = 2
    else:
        gradientLists.append(round(nthmoment, 1))

print('-----------------------------------------------------------------------------')
gradientMatrix = list_matrix(gradientLists, len(grayMatrix), len(grayMatrix[0]))

sortedgradientcount = []
temp = np.array(list(set(gradientLists)))
singal_ele_gradientlist = np.sort(temp, kind='heapsort')
count = 0
for x in singal_ele_gradientlist:
    print(count, [x, gradientLists.count(x), gradientLists.count(x) * 1.0 / len(gradientLists)])
    sortedgradientcount.append([x, gradientLists.count(x), gradientLists.count(x) * 1.0 / len(gradientLists)])
    count += 1

import math
jt = []
i = 1
while i < len(sortedgradientcount) - 2:
    p0t = 0
    u0t = 0
    c0t = 0
    p1t = 0
    u1t = 0
    c1t = 0

    j = 0
    while j <= i:
        p0t += sortedgradientcount[j][2]
        u0t += sortedgradientcount[j][0] * sortedgradientcount[j][2]
        j += 1
    u0t = u0t / p0t
    j = 0
    while j <= i:
        c0t = (sortedgradientcount[j][0] - u0t) ** 2 * sortedgradientcount[j][2]
        j += 1
    c0t = c0t / p0t

    j = i + 1
    while j <= len(sortedgradientcount)-1:
        p1t += sortedgradientcount[j][2]
        u1t += sortedgradientcount[j][0] * sortedgradientcount[j][2]
        j += 1
    u1t = u1t / p1t
    j = i + 1
    while j <= len(sortedgradientcount)-1:
        c1t = (sortedgradientcount[j][0] - u1t) ** 2 * sortedgradientcount[j][2]
        j += 1
    c1t = c1t / p1t

    jt.append(1 + 2 * (p0t*math.log(c0t) + p1t*math.log(c1t)) - 2 * (p0t * math.log(p0t) + p1t * math.log(p1t)))
    i += 1

print(jt)
thresholdjt = min(jt)

threshold = sortedgradientcount[jt.index(thresholdjt) + 1]
print(threshold)

for x in coordLists:
    if gradientMatrix[x[0]][x[1]] > threshold[0]:
        gradientMatrix[x[0]][x[1]] = 255
    else:
        gradientMatrix[x[0]][x[1]] = 0

file3 = open('G:\pythoncode\python3.5\\recongnitionMethodbasedonGradientImg\gradientresult111.txt', 'w')
i = 0
while i < len(gradientMatrix):
    j = 0
    file3.write(str(i) + '---> ')
    while j < len(gradientMatrix[0]):
        if j == len(gradientMatrix[0]) - 1:
             file3.write(str(gradientMatrix[i][j]) + '\n' )
        else:
             file3.write(str(gradientMatrix[i][j]) + ' ')
        j += 1
    i += 1

file3.close()

present_seed = [[239,464], [320, 471],[274, 239]]
coordinate_Matrix = [[i, j] for i in range(len(gradientMatrix)) for j in range(len(gradientMatrix[0]))]
segmentation_ele_list = []
for x in present_seed:
    segmentation_ele_list.append(region_growing_alo(gradientMatrix, coordinate_Matrix, x))

y_max = 0
y_min = len(gradientMatrix)
x_max = 0
x_min = len(gradientMatrix[0])
for x in segmentation_ele_list:
    y = np.array(x)
    if y_max < np.max(y[:, 0]):
        y_max = np.max(y[:, 0])
    if y_min > np.min(y[:, 0]):
        y_min = np.min(y[:, 0])
    if x_max < np.max(y[:, 1]):
        x_max = np.max(y[:, 1])
    if x_min > np.min(y[:, 1]):
        x_min = np.min(y[:, 1])

target = []
i = y_min
while i <= y_max:
    target.append(im1[i][x_min: x_max + 1])
    i += 1

target_img = Image.fromarray(np.array(target))
target_img.show()
target_img.save('target.png')
target_img.close()