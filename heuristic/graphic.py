
__authors__ = ["Melissa", "Ronaldo", "Pedro Pablo"]

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QObject, QTimer
from numpy.random import randn
from numpy import asarray
from matplotlib.figure import Figure

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Qt5Agg")


class Application(QApplication):
	def __init__(self, argv, update_func):
		super().__init__(argv)
		
		self._tick = QTimer(self)
		self._tick.setSingleShot(False)

		self._temp = [0]
		self._x = [0]
		self._y = [0]
		
		self._fig = plt.figure("real-time results...")
		self._rline = self._fig.gca().plot(self._x, self._temp, "-r",
		                                  label="Temp.")
		self._graph = self._fig.gca().plot(self._x, self._y, "-b",
		                                   label="Obj. Func.")
		self._fig.legend()

		self._tick.timeout.connect(update_func)
		self._tick.start()

	def show_graph(self):
		self._fig.show()

	def append(self, temperature, value):
		self._temp.append(temperature)
		self._x.append(len(self._x))
		self._y.append(value)

		self._graph[0].set_xdata(asarray(self._x))
		self._graph[0].set_ydata(asarray(self._y))

		self._rline[0].set_xdata(asarray(self._x))
		self._rline[0].set_ydata(asarray(self._temp))
		
		self._fig.gca().set_xlim(left=len(self._x) - 100, right=max(self._x),
		                         auto=True)
		self._fig.gca().set_ylim(top=max(max(self._temp), max(self._y)) + .5,
		                         bottom=min(min(self._temp), min(self._y)) - .5,
		                         auto=True)
		self._fig.canvas.draw()
