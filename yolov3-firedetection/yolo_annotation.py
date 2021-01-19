from PIL import Image
import xml.etree.cElementTree as ET
import os
import random
import sys

class_ids = {"smoke": 0,"fire":1}
dir_xml_path = "C:/Users/User/Desktop/yolov3-firedetection/VOCdevkit/VOC2007/Annotations/"  # xml标注文件夹
dir_image_path = "C:/Users/User/Desktop/yolov3-firedetection/VOCdevkit/VOC2007/JPEGImages/"  # 原始图像文件夹


# 将VOC-xml标注格式转换成yolo-txt标注格式
def convert_annotation():
    yolo_annotation_list = []
    class_num = [0] * len(list(class_ids.keys()))
    # 检查labelImg标注文件格式是否为.xml
    xml_path_list = [xml for xml in os.listdir(dir_xml_path) if xml.endswith(".xml")]
    for xml_path in xml_path_list:
        xml_path = os.path.join(dir_xml_path, xml_path)  # xml标注文件的绝对路径
        tree = ET.parse(xml_path)  # 载入xml数据
        root = tree.getroot()  # 获取根节点<annotation>
        image_name = root.find('filename').text  # 获取图片名称
        image_path = os.path.join(dir_image_path, image_name)
        # 判断图片格式是否为jpg或png
        if image_name.endswith('.jpg') == False and image_name.endswith('.png') == False:
            print("Warning：format of %s is wrong!"%(image_name))
            continue
        # 判断图片是否存在
        try:
            img = Image.open(image_path)
        except FileNotFoundError:
            print("Warning：%s is not found!"%(image_name))
            continue
        # 判断图片是否损坏
        try:
            img.load()
        except OSError:
            print("Warning：%s is broken!"%(image_name))
            continue
        obj_list = root.findall('object')
        yolo_annotation = ""
        for obj in obj_list:
            class_name = obj.find('name').text
            # 判断类别框是否误标
            if class_name not in list(class_ids.keys()):
                print("Warning：the %s in %s is not the target!" % (class_name, image_name))
                continue
            class_id = class_ids[class_name]  # 获取标注框类别代号
            class_num[class_id] += 1  # 样本标签数量统计
            # yolo数据集的标注格式为“.../raccoon/images/raccoon-131.jpg 1,1,199,184,0 139,77,202,145,0”
            xmlbox = obj.find("bndbox")
            xmin = int(xmlbox.find("xmin").text)
            ymin = int(xmlbox.find("ymin").text)
            xmax = int(xmlbox.find("xmax").text)
            ymax = int(xmlbox.find("ymax").text)
            yolo_annotation += f'{xmin},{ymin},{xmax},{ymax},{class_id} '
        if len(yolo_annotation) == 0:  # 负样本空标注末尾不能有空格
            yolo_annotation_list.append(image_path)
        else:
            yolo_annotation_list.append(image_path + " " + yolo_annotation[:-1])  # 去掉每行末尾空格
    for i in range(0, len(class_ids), 1):
        print(f"the total amount of %s is %i" % (list(class_ids.keys())[i], class_num[i]))
    return yolo_annotation_list


# 按指定比例8：1：1随机划分训练集、验证集和测试集
def create_yolo_annotation(yolo_annotation_path, yolo_train_annotation_path, yolo_test_annotation_path):
    yolo_annotation_list = convert_annotation()
    if os.path.exists(yolo_annotation_path):
        os.remove(yolo_annotation_path)
    # os.mknod(yolo_annotation_path)  # win10下没有os.mknod属性
    with open(yolo_annotation_path, 'w') as f:  # 打开指定文件，没有则新建
        for yolo_annotation in yolo_annotation_list[0:len(yolo_annotation_list)-1]:
            f.write(yolo_annotation + '\n')  # 默认最后一行末尾不能带有换行符，否则报错cannot load image""
        f.write(yolo_annotation_list[len(yolo_annotation_list)-1])
    test_portion = 0.1  # 测试集比例
    test_num = int(test_portion * len(yolo_annotation_list))  # 测试集数量
    random.shuffle(yolo_annotation_list)  # 随机打乱yolo_annotation.txt中标注信息的顺序
    test_annotation_list = random.sample(yolo_annotation_list, test_num)  # 随机选取测试集
    for k in test_annotation_list:
        yolo_annotation_list.remove(k)
    train_annotation_list = yolo_annotation_list  # 训练集和验证集
    # 创建训练集和验证集
    if os.path.exists(yolo_train_annotation_path):
        os.remove(yolo_train_annotation_path)
    with open(yolo_train_annotation_path, 'w') as f:  # 打开指定文件，没有则新建
        for train_annotation in train_annotation_list[0:len(train_annotation_list)-1]:
            f.write(train_annotation + '\n')  # 默认最后一行末尾不能带有换行符，否则报错cannot load image""
        f.write(train_annotation_list[len(train_annotation_list)-1])
    # 创建测试集
    if os.path.exists(yolo_test_annotation_path):
        os.remove(yolo_test_annotation_path)
    with open(yolo_test_annotation_path, 'w') as f:  # 打开指定文件，没有则新建
        for test_annotation in test_annotation_list[0:len(test_annotation_list)-1]:
            f.write(test_annotation + '\n')  # 默认最后一行末尾不能带有换行符，否则报错cannot load image""
        f.write(test_annotation_list[len(test_annotation_list)-1])
    return train_annotation_list, test_annotation_list


# 统计训练集、验证集和测试集中各检测类别的标签数
def calculate_labels():
    train_annotation_list, test_annotation_list = create_yolo_annotation("yolo_annotation.txt", "yolo_train_annotation.txt", "yolo_test_annotation.txt")
    class_names = list(class_ids.keys())  # 标签类别名称
    class_nums = list(class_ids.values())  # 标签类别编号
    train_class_num_list = []
    test_class_num_list = []
    # 统计训练集、验证集标签数
    for train_annotation in train_annotation_list:
        labels_list = train_annotation.split(" ")[1:]
        train_class_num_list.extend(list(map(lambda x: x.split(",")[-1], labels_list)))
    for i in range(0, len(class_names), 1):
        # list.count()统计列表中某个元素出现的次数
        print(f"训练集和验证集标签数目统计为：the total amount of {class_names[i]} is {train_class_num_list.count(str(class_nums[i]))}")
    # 统计测试集标签数
    for test_annotation in test_annotation_list:
        labels_list = test_annotation.split(" ")[1:]
        test_class_num_list.extend(list(map(lambda x: x.split(",")[-1], labels_list)))
    for i in range(0, len(class_names), 1):
        # list.count()统计列表中某个元素出现的次数
        print(f"测试集标签数目统计为：the total amount of {class_names[i]} is {test_class_num_list.count(str(class_nums[i]))}")


if __name__ == '__main__':
    convert_annotation()
    create_yolo_annotation("yolo_annotation.txt", "yolo_train_annotation.txt", "yolo_test_annotation.txt")
    calculate_labels()
