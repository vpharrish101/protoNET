import learn2learn as l2l
import torchvision.transforms as tf

from learn2learn.data import MetaDataset,TaskDataset,partition_task
from learn2learn.data.transforms import LoadData,RemapLabels,FusedNWaysKShots


#Data Division and Sampler
'''This is a helper class designed to return one sampled task containing {support,query} datasets and labels, while
    also transoforming the data and indexing in O(1) [look about MetaDataSet for more info]. Do change the folder 
    directory.'''
class EpisodeSampler:
    def __init__(self,train_ways,train_samples,test_samples):
        self.train_ways=train_ways
        self.train_samples=train_samples
        self.test_samples=test_samples
        self.k=train_samples+test_samples

        self.transforms=tf.Compose([
            tf.Resize(92),                     
            tf.RandomResizedCrop(84,scale=(0.8,1.0)),
            tf.RandomHorizontalFlip(),
            tf.ColorJitter(brightness=0.2,contrast=0.2),
            tf.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225]),])

        #One MetaDataset per split
        self._train_meta=MetaDataset(
            l2l.vision.datasets.MiniImagenet(
                root='E:/DL/Samples/MAML/data',
                download=True,
                mode='train',
                transform=self.transforms
            )
        )
        self._val_meta=MetaDataset(
            l2l.vision.datasets.MiniImagenet(
                root='E:/DL/Samples/MAML/data',
                download=True,
                mode='validation',
                transform=self.transforms
            )
        )
        self._test_meta=MetaDataset(
            l2l.vision.datasets.MiniImagenet(
                root='E:/DL/Samples/MAML/data',
                download=True,
                mode='test',
                transform=self.transforms
            )
        )

        # Holders for dynamic TaskDataset wrappers
        self._train_episode=None
        self._val_episode=None
        self._test_episode=None

    #This fn defines the blueprint for task generation module.
    def build_episode(self,meta_ds):
        return TaskDataset(
            meta_ds,
            task_transforms=[
                FusedNWaysKShots(meta_ds, n=self.train_ways, k=self.k),
                LoadData(meta_ds),
                RemapLabels(meta_ds)
            ],
            num_tasks=1000   # Only 1000 tasks to sample dynamically
        )

    '''These are the variables through which datasets are psuedo-dynamically generated on-demand. Whenever these fn's are called, they
        return the blueprint, then .sample() fn is used to get the data and labels. Once done, the value is set to None is the main 
        loop to clear the variables. To reduce time complexity of repeatedly building new episodes, we build ~1000 episodes for every
        inner loop called, sample from them and then rebuild new datasets by setting the self._*_episode to None.'''
    @property
    def train_episode(self):
        if self._train_episode is None:
            self._train_episode=self.build_episode(self._train_meta)
        return self._train_episode

    @property
    def val_episode(self):
        if self._val_episode is None:
            self._val_episode=self.build_episode(self._val_meta)
        return self._val_episode

    @property
    def test_episode(self):
        if self._test_episode is None:
            self._test_episode=self.build_episode(self._test_meta)
        return self._test_episode

    def sample_train_episode(self):                 #Returns 1 random training episode containing support and query seperately
        data,labels=self.train_episode.sample()
        return partition_task(data,labels,shots=self.train_samples)

    def sample_validation_episode(self):            #Returns 1 random validation episode containing support and query seperately
        data,labels=self.val_episode.sample()
        return partition_task(data,labels,shots=self.train_samples)

    def sample_test_episode(self):                  #Returns 1 random testing episode containing support and query seperately
        data,labels=self.test_episode.sample()
        return partition_task(data,labels,shots=self.train_samples)


