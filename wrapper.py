# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 23:18:00 2021

@author: TSERING JORGAIS
"""
from pathlib import Path



import skimage.io
from matplotlib import pyplot as plt


from Functions import compute_hist, otsu_binarization, foreground_extraction, connected_component, binary_morphology


def solution1():
    image_path=Path('../coins.png')
    bins=10
    my_freq, my_centre, inbuilt_freq, inbuilt_centre=compute_hist(image_path, bins)
    
    return

def solution2():
    image_path=Path('../coins.png')
    time_w, time_b,t_w,t_b,image,inbuilt_t=otsu_binarization(image_path)
    print('time for sigme w: ', time_w)
    print('time for sigme_b:', time_b)
    print('threshold_w: ',t_w)
    print('threshold_b: ',t_b)
    print('inbuilt threshold: ',inbuilt_t)
    
    plt.figure()
    plt.imshow(image, cmap='gray')
    
    return

def solution3():
    text_path=Path('../SingleColorText_Gray.png')
    background_path=Path('../GrassBackground.png')
    final_image=foreground_extraction(text_path,background_path)
    plt.figure()
    skimage.io.imshow(final_image)
    return
    
def solution4():
    image_path=Path('../PiNumbers.png')
    digit, count_1= connected_component(image_path)
    print(digit)
    print(count_1)

    return

def solution5():
    image_path=Path('../NoisyImage.png')
    cleared=binary_morphology(image_path)
    plt.figure()
    plt.imshow(cleared,cmap='gray')
    
    return
    
    
    
    
    

def main():
    #solution1()
    #solution2()
    solution3()
    #solution4()
    #solution5()
    return


if __name__ == '__main__':
    main()