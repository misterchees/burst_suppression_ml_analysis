from MachineLearning.Utils.epochs import Epochs
from MachineLearning.Core.ml_object import MLObject

ml_object = MLObject(False, True)
print("Updating epochs 1")
ml_object.update_current_epochs(1)
print("Updating epochs 2")
ml_object.update_current_epochs(1)
print("Updating epochs 3")
ml_object.set_attributes(channel=2)
ml_object.update_current_epochs(1)




