# 评估训练模型的mAP前需要将测试图片和对应的xml文件放到指定文件夹中
import shutil
import os

test_images_path = "C:/Users/User/Desktop/mAP-keras-yolo3/mAP-master/input/images-optional/"  # 存放测试集图片
test_xmls_path = "C:/Users/User/Desktop/mAP-keras-yolo3/mAP-master/input/ground-truth/"  # 存放测试集标注文件
dir_xml_path = "C:/Users/User/Desktop/yolov3-firedetection/VOCdevkit/VOC2007/Annotations/"  # xml标注文件夹


def map_preprocess():
    with open("yolo_annotation.txt", 'r') as f:
        test_annotation_list = f.read().split("\n")  # test_annotation.txt最后一行末尾不能带有换行符
        test_images_list = list(map(lambda x: x.split(" ")[0], test_annotation_list))  # 测试集图片的绝对路径
    for test_image in test_images_list:
        shutil.copy(test_image, test_images_path)  # 将测试图片复制到mAP指定输入文件夹images-optional下
        if test_image.endswith(".jpg"):
            xml_name = test_image.split("/")[-1].replace(".jpg", ".xml")  # 测试xml名称
        if test_image.endswith(".png"):
            xml_name = test_image.split("/")[-1].replace(".png", ".xml")  # 测试xml名称
        shutil.copy(os.path.join(dir_xml_path, xml_name), test_xmls_path)  # 将测试xml复制到mAP指定输入文件夹ground-truth下


if __name__ == '__main__':
    map_preprocess()
