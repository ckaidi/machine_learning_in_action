from PyQt5.QtWidgets import*
import sys
from PyQt5.QtCore import *
from PyQt5.QtGui import *

class fileDialog:
    def __init__(self) -> None:
        super(fileDialog,self).__init__()
        self.app=QApplication(sys.argv)
        self.setDirectory()
        self.setFilter('*')

    def setFilter(self,filter):
        self.filter=filter

    def setDirectory(self,path=r''):
        self.path=path

    def openFileSelectDialog(self,filter_=''):
        if(filter_!=''):self.filter=filter_
        return QFileDialog.getOpenFileName(caption="open file dialog",directory= self.path,filter=self.filter)[0]

    def openFileSaveDialog(self,filter_):
        return QFileDialog.getSaveFileName(caption='save file',directory=self.path,filter=filter_)[0]