import sys
import numpy as np
from PyQt5.QtWidgets import QMainWindow, QWidget, QGridLayout, QApplication
import pyqtgraph as pg

class CompanionWindow(QMainWindow):
    def __init__(self, num_masks, parent=None):
        super().__init__(parent)
        self.num_masks = num_masks
        self.image_panels = []
        
        # Set up the main widget and layout
        self.central_widget = QWidget()
        self.layout = QGridLayout()
        self.central_widget.setLayout(self.layout)
        self.setCentralWidget(self.central_widget)

        # Determine layout based on num_masks
        if self.num_masks <= 3:
            # Single row with num_masks columns
            for i in range(self.num_masks):
                image_panel = pg.ImageItem()
                self.image_panels.append(image_panel)
                view = pg.GraphicsLayoutWidget()
                vb = view.addViewBox()
                vb.addItem(image_panel)
                self.layout.addWidget(view, 0, i)
        else:
            # Multiple rows, two columns, account for odd numbers
            for i in range(self.num_masks):
                image_panel = pg.ImageItem()
                self.image_panels.append(image_panel)
                view = pg.GraphicsLayoutWidget()
                vb = view.addViewBox()
                vb.addItem(image_panel)
                row = i // 2
                col = i % 2
                self.layout.addWidget(view, row, col)

    def update(self, image, mask_list):
        # Display the image in all panels, and the masks in their respective panels
        for i in range(self.num_masks):
            if i < len(mask_list):
                mask = mask_list[i]
            else:
                # Fill with zeros if not enough masks are provided
                mask = np.zeros(image.shape[:2] + (3,), dtype=np.uint8)  # Assume RGB mask

            # Overlay mask onto the image (if needed)
            # This example just displays the mask, you can modify for actual overlay
            combined_image = image.copy()  # This would be where you handle the overlay logic
            # Update each panel with the combined image
            self.image_panels[i].setImage(combined_image)
            self.image_panels[i].setImage(mask)  # Display the mask for simplicity


if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Create an example window with 5 masks
    window = CompanionWindow(num_masks=5)
    window.show()

    # Example of updating the window
    dummy_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    dummy_masks = [np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8) for _ in range(5)]
    window.update(dummy_image, dummy_masks)

    sys.exit(app.exec_())
