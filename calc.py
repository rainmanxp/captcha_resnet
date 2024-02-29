from captcha_setting import *

def compute_accuracy(model, data_loader, device):
    correct_pred, num_examples = 0, 0
    for i, (features, targets) in enumerate(data_loader):
            
        features = features.to(device)
        targets = targets.to(device)

        
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        optimizer.zero_grad()
        preds = model(features)
        new_preds = preds.reshape(preds.size(0),4,int(preds.size(1)/4))
        _,predicted_labels = torch.max(new_preds,2)
        
        new_targets = targets.reshape(targets.size(0),4,int(targets.size(1)/4))
        _,true_labels = torch.max(new_targets,2)        
            
        num_examples += targets.size(0)
        correct_pred += predicted_labels.eq(true_labels).all(dim=1).sum().item()
        
    return correct_pred/num_examples * 100
        
       
def predict(model, data_loader, device):
    get_char = np.vectorize(lambda x:ALL_CHAR_SET[x])
    concat_char = np.vectorize(lambda x:ALL_CHAR_SET[x])
    output_predicted_labels_list, output_true_labels_list = None, None
    for i, (features, targets) in enumerate(data_loader):
        features = features.to(device)
        targets = targets.to(device)

        
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        optimizer.zero_grad()
        preds = model(features)
        new_preds = preds.reshape(preds.size(0),4,int(preds.size(1)/4))
        _,predicted_labels = torch.max(new_preds,2)
        if output_predicted_labels_list is not None:
            output_predicted_labels_list = np.append(output_predicted_labels_list,get_char(predicted_labels.cpu().numpy()),axis=0)
        else:
            output_predicted_labels_list = get_char(predicted_labels.cpu().numpy()) 
                
        new_targets = targets.reshape(targets.size(0),4,int(targets.size(1)/4))
        _,true_labels = torch.max(new_targets,2)     

        if output_true_labels_list is not None:
            output_true_labels_list = np.append(output_true_labels_list,get_char(true_labels.cpu().numpy()),axis=0)
        else:
            output_true_labels_list = get_char(true_labels.cpu().numpy()) 
        
        
        
    return output_predicted_labels_list, output_true_labels_list

