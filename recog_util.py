# -*- coding: utf-8 -*-
# Created on 2018/8/24
__author__ = 'panyx'

import cv2
import numpy as np
import os
import re
from xml.dom.minidom import Document
#return: img_src 图片数组数据，字典格式
#        img_src.keys 图片名字，数组格式
def read_imgsrc(rootdir):
    img_src = {}
    try:
        for parent,dirnames,filenames in os.walk(rootdir):
            for filename in filenames:
                img_src[filename] = cv2.imread(rootdir + "//" + filename)
        return img_src,img_src.keys()
    except Exception as e:
        print(e)


#给原图标记label（加边框）
#return: img_labeled:标记后的图片（原图上），图片数组数据
#        boxes:边框数组数据(x,y,w,h)
def draw_label(img_rgb):
    lower_yellow = np.array([11, 55, 111])
    upper_yellow = np.array([33, 255, 255])

    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # mask = cv2.GaussianBlur(mask, (3, 3), 0)
    # mask = cv2.medianBlur(mask, 3)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    mask = cv2.erode(mask, kernel, iterations=1)
    
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 4))
    mask = cv2.erode(mask, kernel2, iterations=1)
    
    kernel3=  cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
    mask = cv2.dilate(mask, kernel3, iterations=1)
    
    #mask在用于原图提取
    masked = cv2.bitwise_and(img_rgb,img_rgb,mask=mask)
    cv2.imshow("masked", masked)
    cv2.waitKey(0)
    
    #创建边框
    mser = cv2.MSER_create(_delta=10, _min_area=50, _max_area=600)
    
    #提取mask的边框
    # th = cv2.adaptiveThreshold(mask, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 5, 3)
    regions, boxes = mser.detectRegions(mask)

    #原图标记label
    for box in boxes:
        x, y, w, h = box
        cv2.rectangle(img_rgb, (x, y), (x + w, y + h), (255, 0, 0), 1)
    img_labeled = img_rgb
    return img_labeled,boxes

    #二值化
    #ret,mask_binary=cv2.threshold(mask,10,255,cv2.THRESH_BINARY)
    #image=cv2.add(img_rgb, np.zeros(np.shape(img_rgb), dtype=np.uint8), mask=mask_for_show)


#将mask中的边框数据转换成labelImg指定的格式存储到xml文件
#parame: boxes:mask的边框数据，数组(x,y,w,h)
#        img_name:图片名字
#        img_dir:图片保存路径
#        img_size:图片大下(height,width,depth)
def mask2xml(boxes, img_name, img_dir, img_size):
    #创建dom文档
    doc = Document()
    #创建根节点
    annotation = doc.createElement('annotation')
    #根节点插入dom树
    doc.appendChild(annotation)
    #插入folder
    folder = doc.createElement('folder')
    folder_text = doc.createTextNode('imgs')
    folder.appendChild(folder_text)
    annotation.appendChild(folder)
    #插入filename
    filename = doc.createElement('filename')
    filename_text = doc.createTextNode(img_name)
    filename.appendChild(filename_text)
    annotation.appendChild(filename)
    #插入文件路径
    path = doc.createElement('path')
    path_text = doc.createTextNode(img_dir + "\\" + img_name)
    path.appendChild(path_text)
    annotation.appendChild(path)
    #插入sorce
    source = doc.createElement('source')
    database = doc.createElement('database')
    database_text = doc.createTextNode('Unknown')
    database.appendChild(database_text)
    source.appendChild(database)
    annotation.appendChild(source)
    #插入图片大小
    size = doc.createElement('size')
    width = doc.createElement('width')
    width_text = doc.createTextNode(str(img_size[1]))
    width.appendChild(width_text)
    size.appendChild(width)
    height = doc.createElement('height')
    height_text = doc.createTextNode(str(img_size[0]))
    height.appendChild(height_text)
    size.appendChild(height)
    depth = doc.createElement('depth')
    depth_text = doc.createTextNode(str(img_size[2]))
    depth.appendChild(depth_text)
    size.appendChild(depth)
    annotation.appendChild(size)
    #插入segmented
    segmented = doc.createElement('segmented')
    segmented_text = doc.createTextNode('0')
    segmented.appendChild(segmented_text)
    annotation.appendChild(segmented)
    #插入object
    for box in boxes:
        object = doc.createElement('object')
        name = doc.createElement('name')
        name_text = doc.createTextNode('G0')
        name.appendChild(name_text)
        object.appendChild(name)
        pose = doc.createElement('pose')
        pose_text = doc.createTextNode('Unspecified')
        pose.appendChild(pose_text)
        object.appendChild(pose)
        truncated = doc.createElement('truncated')
        truncated_text = doc.createTextNode('0')
        truncated.appendChild(truncated_text)
        object.appendChild(truncated)
        difficult = doc.createElement('difficult')
        difficult_text = doc.createTextNode('0')
        difficult.appendChild(difficult_text)
        object.appendChild(difficult)
        bndbox = doc.createElement('bndbox')
        x, y, w, h = box
        xmin = doc.createElement('xmin')
        xmin_text = doc.createTextNode(str(x))
        xmin.appendChild(xmin_text)
        bndbox.appendChild(xmin)
        ymin = doc.createElement('ymin')
        ymin_text = doc.createTextNode(str(y))
        ymin.appendChild(ymin_text)
        bndbox.appendChild(ymin)
        xmax = doc.createElement('xmax')
        xmax_text = doc.createTextNode(str(x+w))
        xmax.appendChild(xmax_text)
        bndbox.appendChild(xmax)
        ymax = doc.createElement('ymax')
        ymax_text = doc.createTextNode(str(y+h))
        ymax.appendChild(ymax_text)
        bndbox.appendChild(ymax)
        object.appendChild(bndbox)
        annotation.appendChild(object)

    with open(img_dir+"\\"+img_name.replace('jpg','xml'), "wb+") as f:
        #print(img_dir+"\\"+img_name.replace('jpg','xml'))
        f.write(doc.toprettyxml(indent='\t', encoding='utf-8'))

rootdir = r"E:\document\kingpoint\41\src"
labeldir = r"C:\Users\JockJo\Desktop\labelTool\labelTool\imgs"

img_datas, img_names= read_imgsrc(rootdir)
for img_name in img_names:
    img_rgb = img_datas[img_name]
    img_labeled, boxes = draw_label(img_rgb)
    #边框数据转换成label格式，输出为xml文件
    #mask2xml(boxes, img_name.replace("src", "label"), labeldir, np.shape(img_rgb))
    #cv2.imwrite(labeldir + "\\" + img_name.replace("src","label"), img_labeled)
    cv2.imshow(img_name, img_labeled)
    cv2.waitKey(0)
