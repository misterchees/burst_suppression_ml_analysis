"""
Here is the place to test any code.
"""
from MachineLearning.Features.transforms import Transforms
from MachineLearning.IO.save_result import SaveResult


def transform():
    saver = SaveResult()

    transformer = Transforms(("faw",), "welch", {})
    transformer.update_current_epochs(1)

    transformer.calculate_and_save_psd_for_epochs("faw", saver)


if __name__ == '__main__':
    transform()
