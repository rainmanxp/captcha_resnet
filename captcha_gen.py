# -*- coding: UTF-8 -*-
from captcha.image import ImageCaptcha  # pip install captcha
from PIL import Image
import random
import time
import captcha_setting
import os,sys
import random
import multiprocessing

def random_captcha():
    captcha_text = []
    for i in range(captcha_setting.MAX_CAPTCHA):
        c = random.choice(captcha_setting.ALL_CHAR_SET)
        captcha_text.append(c)
    return ''.join(captcha_text)

# 生成字符对应的验证码
def gen_captcha_text_and_image():
    image = ImageCaptcha()
    captcha_text = random_captcha()
    captcha_image = Image.open(image.generate(captcha_text))
    return captcha_text, captcha_image


def gen(idx):
    text, image = gen_captcha_text_and_image()
    fn = text + '_' + str(idx) + '.png'
    image.save(path  + os.path.sep +  fn)
    
    if idx%2000 == 0:
        print(idx)

if __name__ == '__main__':
    prefix = 'dataset' + os.path.sep
    if len(sys.argv) == 1:
        count = 1000
        path = captcha_setting.TRAIN_DATASET_PATH
    elif len(sys.argv) == 2:
        count = int(sys.argv[1])
        path = captcha_setting.TRAIN_DATASET_PATH
    else:
        count = int(sys.argv[1])
        path = prefix + sys.argv[2]

    #count = 10000
    #path = captcha_setting.TRAIN_DATASET_PATH    #通过改变此处目录，以生成 训练、测试和预测用的验证码集
    import shutil
    shutil.rmtree(path)
    if not os.path.exists(path):
        os.makedirs(path)
        
    fn_list = range(count)
    print(len(fn_list),fn_list[0])
    with multiprocessing.Pool(processes = 500) as pool:
        pool.map(gen,fn_list)
        pool.close()
        pool.join()
    
