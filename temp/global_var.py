#Tuning parameters:
'''This is just a class used for looking and editing at all the various parameters used throught the code'''

class TuningParameters():
    #data params
    _train_ways=7
    _train_samples=5
    _test_samples=15
    
    #meta params
    _epoch=300                      #how many times the collective losses are summed and updated
    _tasks_per_epoch=150            #how many distinct losses are calculated  

    #misc params
    _output_size=64                 #CNN output size

    
    @property
    def train_ways(self):
        return self._train_ways
    @property
    def train_samples(self):
        return self._train_samples
    @property
    def test_samples(self):
        return self._test_samples
    @property
    def epoch(self):
        return self._epoch
    @property
    def tasks_per_epoch(self):
        return self._tasks_per_epoch
    @property
    def output_size(self):
        return self._output_size
