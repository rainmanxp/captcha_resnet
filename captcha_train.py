# -*- coding: UTF-8 -*-
from captcha_setting import *

def train_model(resnet_model,resnet_model_init_weight):
    torch.manual_seed(RANDOM_SEED)

    model = resnet_model(weights=resnet_model_init_weight)
    model.fc = nn.Sequential(nn.Linear(model.fc.in_features, CLASS_NUM))
    
    model = model.to(DEVICE)
            

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE) 
    

    start_time = time.time()
    train_loader = my_dataset.get_train_data_loader() 
    for epoch in range(NUM_EPOCH):
        model.train()
        for batch_idx, (features, targets) in enumerate(train_loader):
            features = features.to(DEVICE)
            targets = targets.to(DEVICE)

            ### FORWARD AND BACK PROP
            optimizer.zero_grad()
            predicts = model(features)
            loss_fuc = nn.MultiLabelSoftMarginLoss()
            loss = loss_fuc(predicts, targets)

            loss.backward()
            ### UPDATE MODEL PARAMETERS
            optimizer.step()

            ### LOGGING
            if not batch_idx % 50:
                print ('Epoch: %03d/%03d | Batch %04d/%04d | Loss: %.6f' 
                       %(epoch+1, NUM_EPOCH, batch_idx, len(train_loader), loss))




    print('Total Training Time: %.2f min' % ((time.time() - start_time)/60))

    torch.save(model.state_dict(),SAVE_MODEL_NAME)
    

if __name__ == '__main__':
    resnet_model = RESNET_DICT['net']['resnet34']
    resnet_model_init_weight = RESNET_DICT['init_weight']['resnet34']

    if len(sys.argv) > 1:
        model_name = sys.argv[1]
        resnet_model = RESNET_DICT['net'][model_name]
        resnet_model_init_weight = RESNET_DICT['init_weight'][model_name]

        
    train_model(resnet_model,resnet_model_init_weight)


