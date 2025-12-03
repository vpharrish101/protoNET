import torch 
import matplotlib.pyplot as plt
import gc
'''In learn2learn==0.2.0, sometimes the .utils package won't be exported properly, even though it exists in the root folder(atleast in my case). This
   was resolved by editing the __init__.py on the main folder, and editing the "from utils import *" =>"from . import utils" and restarting the kernel'''
import learn2learn.utils        

from IPython.display import clear_output
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from global_var import TuningParameters
from episode_sample import EpisodeSampler
from learner import BackBone_CNN,MetaIO_params
import test


def main(device=torch.device('cuda')):
    global model,p,e
    p=TuningParameters()
    e=EpisodeSampler(p.train_ways,p.train_samples,p.test_samples)

    train_classes=p.train_ways
    train_samples=p.train_samples
    test_samples=p.test_samples
    epoch=p.epoch
    tasks_per_epoch=p.tasks_per_epoch
    
    
    model=BackBone_CNN(p.output_size).to(device)
    optimizer=optim.Adam(model.parameters(),lr=0.04)
    scheduler=CosineAnnealingLR(optimizer,T_max=epoch,eta_min=0.0001)

    losses=[]
    accuracies=[]
    epochs=[]

    for i in range(epoch):
        summed_loss,total_accuracy=0.0,0.0
        optimizer.zero_grad()

        for j in range(tasks_per_epoch):

            #Episodes from training data are extracted and shifted to the GPU(or cpu)
            (sdata,slabels),(qdata,qlabels)=e.sample_train_episode()
            support_data=sdata.to(device)           
            query_data=qdata.to(device)
            support_labels=slabels.to(device)
            query_labels=qlabels.to(device)

            #embedding the support/query data using CNN(backbone) and call the MetaIO_params class
            support_embeddings=model(support_data)
            query_embeddings=model(query_data)
            m=MetaIO_params(support_embeddings,support_labels,query_embeddings,query_labels)

            #normally, we chain the losses using computational graphs, then differentiate it at the end of this loop
            #by calling the loss.backward() fn. But if we're calling the backward() fn inside the loop, it reduces the
            #space complexity from O(n) to O(1) as only the previous graph exists in the memory. 
            #This also falls well by the associative rule of differentiation 
            ce_loss,accuracy=m.get_params()
            (ce_loss/tasks_per_epoch).backward()

            summed_loss+=ce_loss.item()
            total_accuracy+=accuracy.item()
            print(f"Inner: Epoch: {j}")
            
        optimizer.step()
        scheduler.step()

        e._train_episode=None
        e._val_episode=None
        e._test_episode=None
        
        # Averaged metrics
        avg_loss=summed_loss/tasks_per_epoch
        avg_acc=total_accuracy/tasks_per_epoch
        losses.append(avg_loss)
        accuracies.append(avg_acc)
        epochs.append(i)

        #Accuracy and Loss / Epoch plots 
        clear_output(wait=True)
        plt.figure(figsize=(10,4))
        plt.subplot(1,2,1)
        plt.plot(epochs,losses,label='Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Meta-Training Loss')
        plt.legend()

        plt.subplot(1,2,2)
        plt.plot(epochs,accuracies,label='Accuracy',color='orange')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Meta-Training Accuracy')
        plt.legend()
        plt.tight_layout()
        plt.show()

        if i:
            print(f"Outer Epoch {i+1}: Loss={avg_loss:.4f}, Acc={avg_acc:.4f}")
        torch.cuda.empty_cache()
        gc.collect()              
    print(f"Final Loss: {losses[-1]:.4f}")
    print(f"Final Accuracy: {accuracies[-1]*100:.2f}%")
    test.evaluate_on_test_set(model,e,p,device=torch.device('cuda'),num_tasks=100)  #type:ignore

if __name__=="__main__":
    main()
    #garbage_disposal()                #Use it if you run trials and need the memory to be cleared.