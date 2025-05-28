from MachineLearning.Utils.plots import Plots

plots = Plots()
plt1 = plots.plot_butterworth_filtering()
plt2 = plots.plot_butterworth_filtering(order=2)
plt3 = plots.plot_butterworth_filtering(order=8)
plt4 = plots.plot_butterworth_filtering(order=16)
plt5 = plots.plot_butterworth_filtering(order=32)

plt1.show(block=False)
plt2.show(block=False)
plt3.show(block=False)
plt4.show(block=False)
plt5.show()
