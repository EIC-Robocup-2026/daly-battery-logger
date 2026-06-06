import sys

from PyQt5.QtWidgets import QApplication

from daly_logger.gui_app import MainWindow

app = QApplication(sys.argv)
app.setApplicationName("Daly BMS Monitor")
window = MainWindow()
window.show()
sys.exit(app.exec_())
