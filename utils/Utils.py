import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
# matplotlib.use('Agg')

class Utils:
    @staticmethod
    def plot(plotlt):
        if not plotlt:
            return
        for k, v in plotlt.items():
            if type(v[1][0]) == np.ndarray:  # predicty
                Utils.plot3D(plotlt)
                return
            else:
                Utils.plot2D(plotlt)
                return

    @staticmethod
    def plot2D(plotlt):
        print("Plot2D")
        size = len(plotlt)
        col = int(math.sqrt(size)) + 1
        row = col
        if col * (row - 1) >= size:
            row -= 1
        index = 1
        for k, v in plotlt.items():
            pid = str(row) + str(col) + str(index)
            ax = plt.subplot(pid)
            plt.title(str(v[2]))
            plt.plot(v[0], color='orange', alpha=1.0, label='Target')
            plt.plot(v[1], color='green', alpha=1.0, label='Prediction')
            plt.legend()
            plt.grid(True)
            plt.subplots_adjust(wspace=0.3, hspace=0.3)
            ax.set_xlabel('Data sample')
            ax.set_ylabel('Value')
            index += 1

    @staticmethod
    def getShortName(name):
        shortname = ''
        strs = name.strip().split(' ')
        for s in strs:
            shortname += s[:2]
        return shortname

    @staticmethod
    def correlation(inputs, outputs):
        data = {}
        cols = []
        opts = []
        for ipt in inputs:
            data[ipt.colName] = ipt.values
            cols.append(ipt.colName)
        for opt in outputs:
            tempname = Utils.getShortName(opt.colName)
            data[tempname] = opt.values
            cols.append(tempname)
            opts.append(tempname)

        df = pd.DataFrame(data, columns=cols)
        corr = df.corr()
        corr_print = corr.loc[:, df.columns.isin(opts)]
        corr_print = corr_print[:-len(opts)]
        pd.set_option('display.max_columns', None)
        print(corr_print)
