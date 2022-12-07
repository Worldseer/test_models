import os
import math
import torch
import torch.nn as nn
import click as ck
import numpy as np
import pandas as pd
import pickle
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import sys
from script import generate_data_loader_all
from script.utils import Ontology
from script import create_model

@ck.command()
@ck.option(
    '--data-root', '-dr', default='./data_2016',
    help='Prediction model')
@ck.option(
    '--batch-size', '-bs', default=16,
    help='Batch size for training')
@ck.option(
    '--epochs', '-ep', default=30,
    help='Training epochs')
@ck.option(
    '--emb-dim', '-ed', default=16,
    help='Embedding Dim')
@ck.option(
    '--winding-size', '-ms', default=40,
    help='Winding matrix size')


def main(data_root,batch_size,epochs,emb_dim,winding_size):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    go = Ontology(f'{data_root}/go.obo', with_rels=True)
    go_list = pd.read_pickle(f'{data_root}/terms.pkl') 
    train_df = pd.read_pickle(f'{data_root}/train_data_train.pkl')
    valid_df = pd.read_pickle(f'{data_root}/train_data_valid.pkl')
    test_df = pd.read_pickle(f'{data_root}/test_data.pkl') 
    out_file = os.path.join(f'./predict/prediction_axialgo.pkl') #Output path for prediction.pkl
    trainloader = generate_data_loader_all.trainloader(train_df,go_list,winding_size,batch_size)
    validloader = generate_data_loader_all.trainloader(valid_df,go_list,winding_size,batch_size)
    testloader = generate_data_loader_all.trainloader(test_df,go_list,winding_size,batch_size,shuffle=False) #cecause the prediction.pkl is generated without disruption
    model = create_model.AxialGO(emb_dim,winding_size,len(go_list)) #Generate new AxialGO
    model.to(device) 
    loss_fn = nn.BCELoss()
    optimizer =  torch.optim.SGD(model.parameters(),lr=0.3,weight_decay=1e-5,momentum=0.9) 
    # optimizer = torch.optim.Adam(model.parameters(),lr=lr)
    terms = go_list['terms'].values.flatten()
    terms_dict = {v: i for i, v in enumerate(terms)}
    n_terms = len(terms_dict)
    best_loss = 1000000.0
    eval_Fmax = 0
    save_path = None
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for idx,(X,y) in enumerate(trainloader):
            model.zero_grad()
            y = y.to(device)
            y_pre = model(X[0].to(device)) #winding style a-f corresponds to X[0]-X[5] respectively
            loss = loss_fn(y_pre,y)
            loss.backward()
            print(loss)
            optimizer.step()
            train_loss += loss.detach().item()
        model.eval()
        with torch.no_grad(): #Evaluation of the validation dataset, with no update of the gradient
            Loss = 0
            valid_loss = 0
            preds = []
            valid_labels = []
            for idx,(X,y) in enumerate(validloader):
                y = y.to(device)
                y_pre = model(X[0].to(device))
                loss = loss_fn(y_pre,y)
                valid_loss += loss.detach().item()
                preds = np.append(preds, y_pre.detach().cpu().numpy())
                valid_labels = np.append(valid_labels, y.detach().cpu().numpy())
            print(f'Epoch {epoch}: Loss - {train_loss}, Valid loss - {valid_loss}')
            
            if valid_loss < best_loss: #Save the model with the minimum loss in the validation set
                if save_path!=None: #Delete the old model
                    os.remove(save_path)
                best_loss = valid_loss
                save_path = os.path.join(f"./model/" +"AxialGO_valid_loss_"+ str("%.4f"%float(valid_loss)) +".param") #Save the best model
                torch.save(model.state_dict(), save_path)
     

    print("------------Load BEST Model----------------")
    print(save_path)
    model_test = create_model.AxialGO(emb_dim,winding_size,len(go_list)) #Generate new parameters
    model_test.load_state_dict(torch.load(save_path)) #Load the best model 
    model_test.to(device)
    model_test.eval()
    with torch.no_grad():
        test_loss = 0
        preds = []
        test_labels = []
        for idx,(X,y) in enumerate(testloader):
            y = y.to(device)
            y_pre = model_test(X[0].to(device))
            batch_loss = loss_fn(y_pre,y)
            test_loss += batch_loss.detach().cpu().item()
            preds = np.append(preds, y_pre.detach().cpu().numpy())
            test_labels = np.append(test_labels, y.detach().cpu().numpy())
        test_loss /= idx
        preds = preds.reshape(-1, n_terms)
        print(f'Test Loss - {test_loss}')  
        preds = list(preds)
        
    # Propagate scores using ontology structure
    print("Starting Propagating")
    for i in range(len(preds)):
        prop_annots = {}
        for go_id, j in terms_dict.items():
            score = preds[i][j]
            for sup_go in go.get_anchestors(go_id):
                if sup_go in prop_annots:
                    prop_annots[sup_go] = max(prop_annots[sup_go], score)
                else:
                    prop_annots[sup_go] = score
        for go_id, score in prop_annots.items():
            if go_id in terms_dict:
                preds[i][terms_dict[go_id]] = score

    test_df['preds'] = preds
    print(f"prediction.pkl is stored in {out_file}")
    test_df.to_pickle(out_file)    
            
            
def predict_df(data_df,labels,preds):
    if labels.is_cuda:
        labels = labels.cpu().detach().numpy()
        preds = preds.cpu().detach().numpy()
    pre_df = data_df
    labs = []
    pres = []
    for label,pred in zip(labels,preds):
        labs.append(label)
        pres.append(pred)
    pre_df["labels"] = labs
    pre_df["preds"] = pres
    return pre_df

           
if __name__ == '__main__':
    main()
