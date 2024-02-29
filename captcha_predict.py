# -*- coding: UTF-8 -*-
from captcha_setting import *
from captcha_test import *


def predict_model(resnet_model,resnet_model_init_weight,model_name,data_loader):
    torch.manual_seed(RANDOM_SEED)

    model = resnet_model(weights=resnet_model_init_weight)
    model.fc = nn.Sequential(nn.Linear(model.fc.in_features, CLASS_NUM))
    
    model = model.to(DEVICE)
        
    model.load_state_dict(torch.load(model_name),strict=False)
    model.eval()

    print(f"load {resnet_model.__name__} succ.")
    
    output_predicted_labels_list, output_true_labels_list = predict(model, data_loader, device=DEVICE)
    
    print('true    label:\n',output_true_labels_list)
    print('predict label:\n',output_predicted_labels_list)

    
    
if __name__ == '__main__':
    resnet_model = RESNET_DICT['net']['resnet34']
    resnet_model_init_weight = RESNET_DICT['init_weight']['resnet34']

    if len(sys.argv) > 1:
        model_name = sys.argv[1]
        resnet_model = RESNET_DICT['net'][model_name]
        resnet_model_init_weight = RESNET_DICT['init_weight'][model_name]

    data_loader = my_dataset.get_predict_data_loader()
    predict_model(resnet_model,resnet_model_init_weight,SAVE_MODEL_NAME,data_loader)


