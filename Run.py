from doe.DoeMan import DoeMan
import matplotlib.pyplot as plt
from utils.Utils import Utils

dman = DoeMan.generateDoeMan('data/UserDefines.json', 'Sample')

# run mode
def crossValidation():
    global dman
    print(dman.compareModelsErrors())

def previewValidation():
    global dman
    Utils.plot(dman.checkError())
    plt.show()

def previewPredict():
    global dman
    num = 100
    Utils.plot(dman.fitAndPredict(num))
    plt.show()


# Utils.correlation(dman.inputs, dman.outputs)
crossValidation()
# previewValidation()
previewPredict()
