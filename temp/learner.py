import torch
import torch.nn as nn
import torch.nn.functional as F

#Backbone CNN
'''This 3-block custom CNN is used as the backbone for my protoNET. Takes in 32x32 res input and produces as 
    256b 1D embedding tensor. Youcan play with adding additional conv layers to increase learnt features, or anything
    else'''

class BackBone_CNN(nn.Module):
    def __init__(self,output_size):
        super(BackBone_CNN,self).__init__()
        def conv_block(in_channels,out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1),
                nn.BatchNorm2d(out_channels),           
                nn.ReLU(),
                nn.Dropout(p=0.1),
                nn.MaxPool2d(kernel_size=2)
            )
        self.encoder=nn.Sequential( 
            conv_block(3,64),       #32x32 => 16x16
            conv_block(64,64),      #16x16 => 8x8
            conv_block(64,64),      #8*8 => 4*4
            conv_block(64,64)       #4*4 => 2*2
        )
        self.classifier=nn.Linear(64*2*2,output_size)
    def forward(self,x):
        x=self.encoder(x)
        x=x.view(x.size(0),-1)
        x=self.classifier(x)
        return x
    


#loss fn
class MetaIO_params:
    '''This function takes in {support:embeddings,labels ; query:embeddings,labels} as input, processes
        it, and returns the loss. 'n' such losses are calculated, summed and then optimized as a
        bunch, which forms a task. This entire fn is an episodic, stateless function'''
    
    def __init__(self,support_embeddings,support_labels,query_embeddings,query_labels):
        self.support_embeddings=support_embeddings
        self.support_labels=support_labels
        self.query_embeddings=query_embeddings
        self.query_labels=query_labels

    #1. Mean Embeddings
    def Ck(support_embeddings,support_labels):      #type:ignore
        classes=torch.unique(support_labels)
        mean_embed=[] 
        for c in classes:
            mask=support_labels==c                  #boolean tensor which is true for values with matching class numbers
            embed=support_embeddings[mask]          #Filtered values according to the bool tensor #type:ignore
            prototype=embed.mean(dim=0)             
            mean_embed.append(prototype)
        return torch.stack(mean_embed)              #returns a tensor containing mean dimensions for each classes, like for class 1-3, the o/p looks like [[1,0.5,3],[4,2,1],[0.6,9,4]]
    
    #2. Loss+predictive probablity:
    def lossFn(ck_embed, query_embeddings,query_labels,support_labels): #type:ignore
        dist=torch.cdist(query_embeddings,ck_embed,p=2)        #calculates euclidean dist (Cosine sim not used as it isn't a Bergman dist)#type:ignore
        p_phi=F.log_softmax(-dist,dim=1)                       #Softmaxx over negatives

        # Map query labels to index positions of unique support labels. This is done to avoid confusing the loss fn, as mismatch in labels results in bogus values
        unique_classes=torch.unique(support_labels)
        label_map={label.item(): idx for idx,label in enumerate(unique_classes)}
        mapped_labels=torch.tensor(
            [label_map[label.item()] for label in query_labels if label.item() in label_map],
            dtype=torch.long,device=query_labels.device
        )

        # Filter query embeddings and logits to match mapped labels
        valid_indices=[i for i,label in enumerate(query_labels) if label.item() in label_map]
        query_embeddings=query_embeddings[valid_indices]
        p_phi=p_phi[valid_indices]

        ce_loss=F.nll_loss(p_phi,mapped_labels)
        prediction=torch.argmax(p_phi,dim=1)
        accuracy=(prediction==mapped_labels).float().mean()
        return ce_loss,accuracy
    
    #3. the return function for loss and accuracy: -
    def get_params(self):
        prototype=MetaIO_params.Ck(self.support_embeddings,self.support_labels)
        ce_loss,accuracy=MetaIO_params.lossFn(prototype,self.query_embeddings,self.query_labels,self.support_labels) #type:ignore
        return ce_loss,accuracy