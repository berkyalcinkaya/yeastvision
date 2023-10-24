# from PyQt5.QtWidgets import QApplication, QComboBox, QStyledItemDelegate, QStyle, QMessageBox, QWidget, QVBoxLayout, QListView
# from PyQt5.QtCore import QRect
# from PyQt5.QtGui import QPainter

# class RemoveItemDelegate(QStyledItemDelegate):
#     def __init__(self, parent=None):
#         super(RemoveItemDelegate, self).__init__(parent)

#     def paint(self, painter, option, index):
#         super().paint(painter, option, index)
#         close_icon = self.parent().style().standardIcon(QStyle.SP_DockWidgetCloseButton)
#         rect = self.get_close_button_rect(option.rect)
#         close_icon.paint(painter, rect)

#     def get_close_button_rect(self, item_rect):
#         icon_size = 16
#         return QRect(item_rect.right() - icon_size - 5, item_rect.center().y() - icon_size // 2, icon_size, icon_size)

# class CustomListView(QListView):
#     def __init__(self, combo, parent=None):
#         super(CustomListView, self).__init__(parent)
#         self._combo = combo

#     def mousePressEvent(self, event):
#         combo = self._combo
#         delegate = combo.itemDelegate()
#         index_under_mouse = self.indexAt(event.pos())
        
#         if isinstance(delegate, RemoveItemDelegate) and delegate.get_close_button_rect(self.visualRect(index_under_mouse)).contains(event.pos()):
#             reply = QMessageBox.question(self, "Confirmation", "Are you sure you want to remove this item?",
#                                          QMessageBox.Yes | QMessageBox.No)
#             if reply == QMessageBox.Yes:
#                 combo.removeItem(index_under_mouse.row())
#         else:
#             super().mousePressEvent(event)

# app = QApplication([])

# window = QWidget()
# layout = QVBoxLayout(window)
# combo = QComboBox(window)
# combo.setView(CustomListView(combo))
# combo.setItemDelegate(RemoveItemDelegate(combo))
# combo.addItem("Item 1")
# combo.addItem("Item 2")
# combo.addItem("Item 3")
# layout.addWidget(combo)
# window.setLayout(layout)

# window.resize(400, 200)
# window.show()

# app.exec_()
