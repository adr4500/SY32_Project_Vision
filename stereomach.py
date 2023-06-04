import numpy as np
from skimage import data
from skimage.color import rgb2gray
from skimage import io 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


'''
-----------------------------------------------------
Fonctions optimisées de block matching classique
-----------------------------------------------------
'''

def get_pixel_block(img, xPx : int, yPx : int, N) :
    # On considère que N est impair
    n = int(np.floor(N/2))
    yInf = yPx - n #if yPx - n >= 0 else 0
    xInf = xPx - n #if xPx - n >= 0 else 0
    ySup = yPx + n #if yPx + n <= img.shape[0] else img.shape[0]
    xSup = xPx + n #if xPx + n <= img.shape[1] else img.shape[1]
    block = img[yInf:ySup+1, xInf:xSup+1] 
    return block

def findDispFromSAD(block, blocksOfLine) :
    diffs = np.abs(np.subtract(blocksOfLine, block, dtype=int))
    sads = np.sum(diffs, axis=(1,2))
    if sads.shape[0] == 0 :
        return -1
    return np.argmin(sads)

def block_matching(imgGrayLeft, imgGrayRight, maxdisp, N, refIsLeft : bool) :
    # N doit être impair
    n = int(np.floor(N/2))
    if refIsLeft :
        borneSupX = imgGrayLeft.shape[1] - n 
        borneSupY = imgGrayLeft.shape[0] - n 
        dispMap = np.full_like(imgGrayLeft, -1, int)
        for y in range(n+1, borneSupY) :
            lineBlocks = []
            for x in range(n+1, borneSupX) :
                block = get_pixel_block(imgGrayRight, x, y, N)
                lineBlocks.append(block)
            lineBlocks = np.array(lineBlocks)

            for x in range(n+1, borneSupX) :
                blockRef = get_pixel_block(imgGrayLeft, x, y, N)
                maxBlockInd = max(x-maxdisp, 0)
                dispMap[y,x] = np.abs(findDispFromSAD(blockRef, lineBlocks[maxBlockInd:x]) - maxdisp)
    else :
        borneSupX = imgGrayRight.shape[1] - n 
        borneSupY = imgGrayRight.shape[0] - n 
        dispMap = np.full_like(imgGrayRight, -1, int)
        for y in range(n+1, borneSupY) :
            lineBlocks = []
            for x in range(n+1, borneSupX) :
                block = get_pixel_block(imgGrayLeft, x, y, N)
                lineBlocks.append(block)
            lineBlocks = np.array(lineBlocks)

            for x in range(n+1, borneSupX) :
                blockRef = get_pixel_block(imgGrayRight, x, y, N)
                maxBlockInd = min(x+maxdisp, imgGrayRight.shape[1])
                dispMap[y,x] = findDispFromSAD(blockRef, lineBlocks[x:maxBlockInd])
    return dispMap



'''
--------------------------------------------------------------------------------
Block Matching avec valeur maximale de SAD pour valider un appariement de pixels
-------------------------------------------------------------------------------- 
'''

def findDispFromSADWithTreshold(block, blocksOfLine, treshold) :
    diffs = np.abs(np.subtract(blocksOfLine, block, dtype=int))
    sads = np.sum(diffs, axis=(1,2))
    if sads.shape[0] == 0 :
        return -1
    if np.min(sads) > treshold :
        return -1
    # Plusieurs appariements possibles, on ignore l'ambiguïté
    if np.count_nonzero(sads == np.min(sads)) > 1 :
        return -1
    return np.argmin(sads)

def blockMatchingTreshold(imgGrayLeft, imgGrayRight, maxdisp, N, refIsLeft : bool, treshold) :
    # N doit être impair
    n = int(np.floor(N/2))
    if refIsLeft :
        borneSupX = imgGrayLeft.shape[1] - n 
        borneSupY = imgGrayLeft.shape[0] - n 
        dispMap = np.full_like(imgGrayLeft, -1, int)
        for y in range(n+1, borneSupY) :
            lineBlocks = []
            for x in range(n+1, borneSupX) :
                block = get_pixel_block(imgGrayRight, x, y, N)
                lineBlocks.append(block)
            lineBlocks = np.array(lineBlocks)

            for x in range(n+1, borneSupX) :
                blockRef = get_pixel_block(imgGrayLeft, x, y, N)
                maxBlockInd = max(x-maxdisp, 0)
                disp = findDispFromSADWithTreshold(blockRef, lineBlocks[maxBlockInd:x], treshold)
                if disp != -1 :
                    dispMap[y,x] = np.abs(disp - maxdisp)
    else :
        borneSupX = imgGrayRight.shape[1] - n 
        borneSupY = imgGrayRight.shape[0] - n 
        dispMap = np.full_like(imgGrayRight, -1, int)
        for y in range(n+1, borneSupY) :
            lineBlocks = []
            for x in range(n+1, borneSupX) :
                block = get_pixel_block(imgGrayLeft, x, y, N)
                lineBlocks.append(block)
            lineBlocks = np.array(lineBlocks)

            for x in range(n+1, borneSupX) :
                blockRef = get_pixel_block(imgGrayRight, x, y, N)
                maxBlockInd = min(x+maxdisp, imgGrayRight.shape[1])
                disp = findDispFromSADWithTreshold(blockRef, lineBlocks[x:maxBlockInd], treshold)
                dispMap[y,x] = disp
    return dispMap



'''
------------------------------
Post-traitement des disparités
------------------------------
'''

def filtre_mode(dispMap, sizeFiltre) :
    newDispMap = np.full_like(dispMap, -1)
    n = int(np.floor(sizeFiltre/2))
    for y in range(n+1, dispMap.shape[0] - n + 1) :
        for x in range(n+1, dispMap.shape[1] - n + 1) :
            block = get_pixel_block(dispMap, x, y, sizeFiltre)
            values, counts = np.unique(block, return_counts=True)
            maxInd = np.argmax(counts)
            newDispMap[y,x] = values[maxInd]
    
    return newDispMap





# Main à faire