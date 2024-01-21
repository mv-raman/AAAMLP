import os
import numpy as np
import pandas as pd
from sklearn import ensemble
from sklearn import metrics
from sklearn import model_selection
from tqdm import tqdm
from PIL import Image
    
def create_dataset(training_df,image_dir):
    
    images = []
    
    targets = []
    
    for index,row in tqdm(training_df.iterrows(),
                         total = len(training_df),
                         desc = 'processing_images'):
        
        image_id = row['ImageId']
        
        image_path = os.path.join(image_dir,image_id)
    
        image = Image.open(image_path + ".png")
        
        image = image.resize((256,256),resample=Image.BILINEAR)
        
        image = np.array(image)
        
        image = image.ravel()
        
        images.append(image)
        
        targets.append(int(row['target']))
        
        
    images = np.array(images)
    
    print(images.shape)
    
    return images,targets


if __name__ == '__main__':
    
    img_path = '/Users/v0m01sk/Documents/Code/AAAMLP/code/data/siim_png/train_png/'
    csv_path = '/Users/v0m01sk/Documents/Code/AAAMLP/code/data/train-rle.csv'
    
    df = pd.read_csv(csv_path)
    
    df['target'] = np.where(df[' EncodedPixels']==' -1',0,1)
    
    df['kfold'] = -1
    
    df = df.sample(frac=1).reset_index(drop=True)
    
    y = df.target.values
    
    kf = model_selection.StratifiedKFold(n_splits=5)
    
    for f, (t_,v_) in enumerate(kf.split(X=df, y=y)):
        df.loc[v_,'kfold']=f
        
        
    for fold_ in range(5):
        train_df = df[df.kfold != fold_].reset_index(drop=True)
        test_df = df[df.kfold == fold_].reset_index(drop=True)
        
        xtrain, ytrain = create_dataset(train_df,img_path)
        xtest, ytest = create_dataset(test_df,img_path)
        
        clf = ensemble.RandomForestClassifier(n_jobs=-1)
        clf.fit(xtrain, ytrain)
        
        preds = clf.predict_proba(xtest)[:,1]
        
        print(f"Fold:{fold_}")
        print(f"AUC:{metrics.roc_auc_score(ytest,preds)}")
        print("")
    