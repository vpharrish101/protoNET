import torch

from learner import MetaIO_params

#Model performance validation fn
'''This fn is used mainly to test the protoNET in an unseen dataset. Uses testing episodes, which are not in the
    original training episodes, to better gauge the performance of the NN as a whole.'''
    
def evaluate_on_test_set(model,episode_sampler,tuning_params,device=torch.device('cuda'),num_tasks=100):
    model.eval()
    total_loss=0.0
    total_accuracy=0.0

    with torch.no_grad():               #prevents updation of CNN params
        for i in range(num_tasks):
            
            #data is split and shifted to device
            (sdata,slabels),(qdata,qlabels)=episode_sampler.sample_test_episode()
            support_data=sdata.to(device)
            query_data=qdata.to(device)
            support_labels=slabels.to(device)
            query_labels=qlabels.to(device)

            #CNN embddings are extracted, and loss/accuracy are calculatd
            support_embeddings=model(support_data)
            query_embeddings=model(query_data)
            m=MetaIO_params(support_embeddings,support_labels,query_embeddings,query_labels)
            ce_loss,accuracy=m.get_params()
            
            total_loss+=ce_loss.item()
            total_accuracy+=accuracy.item()

    #Final report
    avg_loss=total_loss/num_tasks
    avg_accuracy=total_accuracy/num_tasks

    print(f"Test Set Evaluation over {num_tasks} tasks")
    print(f"Average Loss: {avg_loss:.4f}")
    print(f"Average Accuracy: {avg_accuracy * 100:.2f}%")

    return avg_loss, avg_accuracy