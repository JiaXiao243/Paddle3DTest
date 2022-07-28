import pytest
import numpy as np
import subprocess
import re
import sys
import yaml
import platform
import os.path
import allure

from ModelsTestFramework import RepoInit
from ModelsTestFramework import RepoDataset
from ModelsTestFramework import TestOcrModelFunction


def get_model_list():
    import sys
    result = []
    with open('models_list.yaml') as f:
      lines = f.readlines()
      for line in lines:
         r = re.search('/(.*)/', line)
         result.append(line.strip('\n'))
    return result

def setup_module():
    """
    """
    RepoInit(repo='PaddleOCR')
    RepoDataset()


@allure.story('get_pretrained_model')
@pytest.mark.parametrize('yml_name', get_model_list())
def test_ocr_accuracy_get_pretrained_model(yml_name):
    model_name=os.path.splitext(os.path.basename(yml_name))[0]
    allure.dynamic.title(model_name+'_get_pretrained_model')
    allure.dynamic.description('获取预训练模型')
    r = re.search('/(.*)/', yml_name)
    category=r.group(1)
    print(category)
    model = TestOcrModelFunction(model=model_name, yml=yml_name, category=category)
    model.test_ocr_get_pretrained_model()

@allure.story('eval')
@pytest.mark.parametrize('yml_name', get_model_list())
@pytest.mark.parametrize("use_gpu", [True])
def test_ocr_accuracy_eval(yml_name, use_gpu):
    if sys.platform == 'darwin' and use_gpu==True:
        pytest.skip("mac skip GPU")

    model_name=os.path.splitext(os.path.basename(yml_name))[0]
    if use_gpu==True:
       hardware='_GPU'
    else:
       hardware='_CPU'
    allure.dynamic.title(model_name+hardware+'_eval')
    allure.dynamic.description('模型评估')
    r = re.search('/(.*)/', yml_name)
    r = re.search('/(.*)/', yml_name)
    category=r.group(1)
    print(category)
    model = TestOcrModelFunction(model=model_name, yml=yml_name, category=category)
    model.test_ocr_eval(use_gpu)

@allure.story('infer')
@pytest.mark.parametrize('yml_name', get_model_list())
@pytest.mark.parametrize("use_gpu", [True,False])
def test_ocr_accuracy_infer(yml_name, use_gpu):
    if sys.platform == 'darwin' and use_gpu==True:
        pytest.skip("mac skip GPU")

    model_name=os.path.splitext(os.path.basename(yml_name))[0]
    if use_gpu==True:
       hardware='_GPU'
    else:
       hardware='_CPU'
    allure.dynamic.title(model_name+hardware+'_infer')
    allure.dynamic.description('模型预测')
    r = re.search('/(.*)/', yml_name)
    category=r.group(1)
    print(category)
    model = TestOcrModelFunction(model=model_name, yml=yml_name, category=category)
    model.test_ocr_rec_infer(use_gpu)

@allure.story('export_model')
@pytest.mark.parametrize('yml_name', get_model_list())
@pytest.mark.parametrize("use_gpu", [True,False])
def test_ocr_accuracy_export_model(yml_name, use_gpu):
    if sys.platform == 'darwin' and use_gpu==True:
        pytest.skip("mac skip GPU")
    model_name=os.path.splitext(os.path.basename(yml_name))[0]
    if use_gpu==True:
       hardware='_GPU'
    else:
       hardware='_CPU'
    allure.dynamic.title(model_name+hardware+'_export_model')
    allure.dynamic.description('模型动转静')
    r = re.search('/(.*)/', yml_name)
    category=r.group(1)
    print(category)
    model = TestOcrModelFunction(model=model_name, yml=yml_name, category=category)
    model.test_ocr_export_model(use_gpu)

@allure.story('predict')
@pytest.mark.parametrize('yml_name', get_model_list())
@pytest.mark.parametrize("enable_mkldnn", [True, False])
def test_ocr_accuracy_predict_mkl(yml_name, enable_mkldnn):
    if sys.platform == 'darwin' and enable_mkldnn==True:
        pytest.skip("mac skip mkldnn predict")
    model_name=os.path.splitext(os.path.basename(yml_name))[0]
    if enable_mkldnn==True:
       hardware='_mkldnn'
    else:
       hardware='_CPU'
    allure.dynamic.title(model_name+hardware+'_predict')
    allure.dynamic.description('预测库预测')
    r = re.search('/(.*)/', yml_name)
    category=r.group(1)
    print(category)
    model = TestOcrModelFunction(model=model_name, yml=yml_name, category=category)
    model.test_ocr_rec_predict(False, 0, enable_mkldnn)

@allure.story('predict')
@pytest.mark.parametrize('yml_name', get_model_list())
@pytest.mark.parametrize("use_tensorrt", [True, False])
def test_ocr_accuracy_predict_trt(yml_name, use_tensorrt):
    if sys.platform == 'darwin':
        pytest.skip("mac skip tensorRT predict")
    model_name=os.path.splitext(os.path.basename(yml_name))[0]
    if use_tensorrt==True:
       hardware='_tensorRT'
    else:
       hardware='_GPU'
    allure.dynamic.title(model_name+hardware+'_predict')
    allure.dynamic.description('预测库预测')
    r = re.search('/(.*)/', yml_name)
    category=r.group(1)
    print(category)
    model = TestOcrModelFunction(model=model_name, yml=yml_name, category=category)
    model.test_ocr_rec_predict(True, use_tensorrt, 0)

@allure.story('train')
@pytest.mark.parametrize('yml_name', get_model_list())
@pytest.mark.parametrize("use_gpu", [True])
def test_ocr_funtion_train(yml_name, use_gpu):
    if sys.platform == 'darwin' and use_gpu==True:
        pytest.skip("mac skip GPU")
    model_name=os.path.splitext(os.path.basename(yml_name))[0]
    if use_gpu==True:
       hardware='_GPU'
    else:
       hardware='_CPU'
    allure.dynamic.title(model_name+hardware+'_train')
    allure.dynamic.description('训练')
    r = re.search('/(.*)/', yml_name)
    category=r.group(1)
    print(category)
    model = TestOcrModelFunction(model=model_name, yml=yml_name, category=category)
    model.test_ocr_train(use_gpu)

@pytest.mark.skipif(sys.platform == 'win32', reason="does not run on windows")
@allure.story('train')
@pytest.mark.parametrize('yml_name', get_model_list())
def test_ocr_accuracy_train(yml_name):
    if sys.platform == 'darwin' or sys.platform == 'win32':
        pytest.skip("mac/windows skip convergence plot")

    model_name=os.path.splitext(os.path.basename(yml_name))[0]
    allure.dynamic.title(model_name+'_train_convergence')
    allure.dynamic.description('训练收敛性')
    r = re.search('/(.*)/', yml_name)
    category=r.group(1)
    print(category)
    model = TestOcrModelFunction(model=model_name, yml=yml_name, category=category)
    model.test_ocr_train_acc()
