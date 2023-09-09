def build_widgets(self):
        rowspace = self.mainViewRows+1
        cspace = 2
        self.labelstyle = """QLabel{
                            color: white
                            } 
                         QToolTip { 
                           background-color: black; 
                           color: white; 
                           border: black solid 1px
                           }"""
        self.statusbarstyle = ("color: white;" "background-color : black")
        self.boldfont = QtGui.QFont("Arial", 14, QtGui.QFont.Bold)
        self.medfont = QtGui.QFont("Arial", 12)
        self.smallfont = QtGui.QFont("Arial", 10)
        self.headings = ('color: rgb(200,10,10);')
        self.dropdowns = ("color: white;"
                        "background-color: rgb(40,40,40);"
                        "selection-color: white;"
                        "selection-background-color: rgb(50,100,50);")
        self.checkstyle = "color: rgb(190,190,190);"

        self.statusBar = QStatusBar()
        self.statusBar.setFont(self.medfont)
        self.statusBar.setStyleSheet(self.statusbarstyle)
        self.setStatusBar(self.statusBar)

        self.gpuDisplayTorch = ReadOnlyCheckBox("gpu - torch  |  ")
        self.gpuDisplayTF= ReadOnlyCheckBox("gpu - tf")
        self.gpuDisplayTF.setFont(self.smallfont)
        self.gpuDisplayTorch.setFont(self.smallfont)
        self.gpuDisplayTF.setStyleSheet(self.checkstyle)
        self.gpuDisplayTorch.setStyleSheet(self.checkstyle)
        self.gpuDisplayTF.setChecked(False)
        self.gpuDisplayTorch.setChecked(False)
        self.checkGPUs()
        if self.tf:
            self.gpuDisplayTF.setChecked(True)
        if self.torch:
            self.gpuDisplayTorch.setChecked(True)
        self.statusBarLayout = QGridLayout()
        self.statusBarWidget = QWidget()
        self.statusBarWidget.setLayout(self.statusBarLayout)

        self.cpuCoreDisplay = QLabel("")
        self.cpuCoreDisplay.setFont(self.smallfont)
        self.cpuCoreDisplay.setStyleSheet(self.labelstyle)
        self.updateThreadDisplay()

        self.hasLineageBox = ReadOnlyCheckBox("lineage data")
        self.hasCellDataBox = ReadOnlyCheckBox("cell data")
        for display in [self.hasLineageBox,self.hasCellDataBox]:
            display.setFont(self.smallfont)
            display.setStyleSheet(self.checkstyle)
            display.setChecked(False)

        self.statusBarLayout.addWidget(self.cpuCoreDisplay, 0, 1, 1,1, alignment=(QtCore.Qt.AlignCenter))
        self.statusBarLayout.addWidget(self.gpuDisplayTF, 0, 2, 1, 1)
        self.statusBarLayout.addWidget(self.gpuDisplayTorch, 0, 3, 1, 1)
        self.statusBarLayout.addWidget(self.hasLineageBox, 0,4,1,1)
        self.statusBarLayout.addWidget(self.hasCellDataBox,0,5,1,1)
        self.statusBar.addWidget(self.statusBarWidget)
        
        self.dataDisplay = QLabel("")
        self.dataDisplay.setMinimumWidth(300)
        # self.dataDisplay.setMaximumWidth(300)
        self.dataDisplay.setStyleSheet(self.labelstyle)
        self.dataDisplay.setFont(self.smallfont)
        self.dataDisplay.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignBottom)
        self.updateDataDisplay()
        self.l.addWidget(self.dataDisplay, rowspace-1,cspace+1,1,20)


        self.experimentLabel = QLabel("Experiment:")
        self.experimentLabel.setStyleSheet(self.labelstyle)
        self.experimentLabel.setFont(self.smallfont)
        self.l.addWidget(self.experimentLabel, 0,1,1,2)
        self.experimentSelect= QComboBox()
        self.experimentSelect.setStyleSheet(self.dropdowns)
        self.experimentSelect.setFocusPolicy(QtCore.Qt.NoFocus)
        self.experimentSelect.setFont(self.medfont)
        self.experimentSelect.currentIndexChanged.connect(self.experimentChange)
        self.l.addWidget(self.experimentSelect, 0, 3, 1,2)
        

        self.channelSelectLabel = QLabel("Channel: ")
        self.channelSelectLabel.setStyleSheet(self.labelstyle)
        self.channelSelectLabel.setFont(self.smallfont)
        self.l.addWidget(self.channelSelectLabel, 0, self.mainViewCols-7,1,2)
        self.channelSelect = QComboBox()
        self.channelSelect.setStyleSheet(self.dropdowns)
        self.channelSelect.setFont(self.medfont)
        self.channelSelect.setCurrentText("")
        self.channelSelect.currentIndexChanged.connect(self.channelSelectIndexChange)
        self.channelSelect.setEditable(True)
        self.channelSelect.editTextChanged.connect(self.channelSelectEdit)
        self.channelSelect.setEnabled(False)
        self.l.addWidget(self.channelSelect, 0, self.mainViewCols-5,1, 2)
        self.l.setAlignment(self.channelSelect, QtCore.Qt.AlignLeft)
        self.channelSelect.setMinimumWidth(200)
        self.channelSelect.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.channelSelect.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        setattr(self.channelSelect, "items", lambda: [self.channelSelect.itemText(i) for i in range(self.channelSelect.count())])


        self.labelSelectLabel = QLabel("Label: ")
        self.labelSelectLabel.setStyleSheet(self.labelstyle)
        self.labelSelectLabel.setFont(self.smallfont)
        self.l.addWidget(self.labelSelectLabel, 0, self.mainViewCols-3,1,2)
        self.labelSelect = QComboBox()
        self.labelSelect.setStyleSheet(self.dropdowns)
        self.labelSelect.setFont(self.medfont)
        self.labelSelect.setCurrentText("")
        self.labelSelect.currentIndexChanged.connect(self.labelSelectIndexChange)
        self.labelSelect.setEditable(True)
        self.labelSelect.editTextChanged.connect(self.labelSelectEdit)
        self.labelSelect.setMinimumWidth(200)
        self.labelSelect.setEnabled(False)
        self.l.addWidget(self.labelSelect, 0, self.mainViewCols-2,1,2)
        self.l.setAlignment(self.labelSelect, QtCore.Qt.AlignLeft)
        setattr(self.labelSelect, "items", lambda: [self.labelSelect.itemText(i) for i in range(self.labelSelect.count())])
        self.labelSelect.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.labelSelect.setSizeAdjustPolicy(QComboBox.AdjustToContents)

        label = QLabel('Drawing:')#[\u2191 \u2193]')
        label.setStyleSheet(self.headings)
        label.setFont(self.boldfont)
        self.l.addWidget(label, rowspace,1,1,5)

        label = QLabel("Brush Type:")
        label.setStyleSheet(self.labelstyle)
        label.setFont(self.medfont)
        self.l.addWidget(label,rowspace+1,1,1,2)
        self.brushTypeSelect = QComboBox()
        self.brushTypeSelect.addItems(["Eraser", "Brush", "Outline"])
        self.brushTypeSelect.setCurrentIndex(-1)
        self.brushTypeSelect.currentIndexChanged.connect(self.brushTypeChoose)
        self.brushTypeSelect.setStyleSheet(self.dropdowns)
        self.brushTypeSelect.setFont(self.medfont)
        self.brushTypeSelect.setCurrentText("")
        self.brushTypeSelect.setFocusPolicy(QtCore.Qt.NoFocus)
        self.brushTypeSelect.setEnabled(False)
        self.brushTypeSelect.setFixedWidth(90)
        self.brushTypeSelect.setEnabled(False)
        self.l.addWidget(self.brushTypeSelect, rowspace+1,3,1,1)

        
        label = QLabel("Brush Size")
        label.setStyleSheet(self.labelstyle)
        label.setFont(self.medfont)
        self.l.addWidget(label, rowspace+2,1,1,2)
        self.brush_size = 3
        self.brushSelect = QSpinBox()
        self.brushSelect.setMinimum(1)
        self.brushSelect.setValue(self.brush_size)
        self.brushSelect.valueChanged.connect(self.brushSizeChoose)
        self.brushSelect.setFixedWidth(90)
        self.brushSelect.setStyleSheet(self.dropdowns)
        self.brushSelect.setFont(self.medfont)
        edit = self.brushSelect.lineEdit()
        edit.setFocusPolicy(QtCore.Qt.NoFocus)
        self.brushSelect.setEnabled(False)
        self.l.addWidget(self.brushSelect, rowspace+2,3,1,1)

        line = QVLine()
        line.setStyleSheet('color:white;')
        self.l.addWidget(line, rowspace,4,6,1)

        label = QLabel('Tracking')#[\u2191 \u2193]')
        label.setStyleSheet(self.headings)
        label.setFont(self.boldfont)
        self.l.addWidget(label, rowspace,5,1,4)

        self.cellNumButton = QCheckBox("cell nums")
        self.cellNumButton.setStyleSheet(self.checkstyle)
        self.cellNumButton.setFont(self.medfont)
        self.cellNumButton.stateChanged.connect(self.toggleCellNums)
        self.cellNumButton.setEnabled(False)
        self.l.addWidget(self.cellNumButton, rowspace+1, 5, 1,2)

        self.showLineageButton = QCheckBox("lineages")
        self.showLineageButton.setStyleSheet(self.checkstyle)
        self.showLineageButton.setFont(self.medfont)
        self.showLineageButton.stateChanged.connect(self.toggleLineages)
        self.showLineageButton.setEnabled(False)
        self.l.addWidget(self.showLineageButton, rowspace+1, 7, 1,2)

        self.trackButton = QPushButton('track cells')
        #self.trackButton.setFixedWidth(90)
        #self.trackButton.setFixedHeight(20)
        self.trackButton.setStyleSheet(self.styleInactive)
        self.trackButton.setFont(self.medfont)
        self.trackButton.clicked.connect(self.trackButtonClick)
        self.trackButton.setEnabled(False)
        self.trackButton.setToolTip("Track current cell labels")
        self.l.addWidget(self.trackButton, rowspace+2, 5,1,2)

        self.trackObjButton = QPushButton('track to cell')
        #self.trackObjButton.setFixedWidth(90)
        #self.trackObjButton.setFixedHeight(20)
        self.trackObjButton.setFont(self.medfont)
        self.trackObjButton.setStyleSheet(self.styleInactive)
        self.trackObjButton.clicked.connect(self.trackObjButtonClick)
        self.trackObjButton.setEnabled(False)
        self.trackObjButton.setToolTip("Track current non-cytoplasmic label to a cellular label")
        self.l.addWidget(self.trackObjButton, rowspace+2, 7,1,2)

        self.lineageButton = QPushButton("get lineages")
        self.lineageButton.setStyleSheet(self.styleInactive)
        #self.lineageButton.setFixedWidth(90)
        #self.lineageButton.setFixedHeight(18)
        self.lineageButton.setFont(self.medfont)
        self.lineageButton.setToolTip("Use current budNET mask to assign lineages to a cellular label")
        self.lineageButton.setEnabled(False)
        self.showMotherDaughters = False
        self.showLineages = False
        self.lineageButton.clicked.connect(self.getLineages)
        self.l.addWidget(self.lineageButton, rowspace+3, 5,1,2, Qt.AlignBottom)

        self.showMotherDaughtersButton = QCheckBox("mother-daughters")
        self.showMotherDaughtersButton.setStyleSheet(self.checkstyle)
        self.showMotherDaughtersButton.setFont(self.medfont)
        self.showMotherDaughtersButton.stateChanged.connect(self.toggleMotherDaughters)
        self.showMotherDaughtersButton.setEnabled(False)
        self.l.addWidget(self.showMotherDaughtersButton, rowspace+3, 7, 1,2)

        self.interpolateButton = QPushButton("Enhance Tracking Through Frame Interpolation")
        self.interpolateButton.setStyleSheet(self.styleInactive)
        self.interpolateButton.setFont(self.medfont)
        self.interpolateButton.setEnabled(False)
        self.interpolateButton.clicked.connect(self.interpolateButtonClicked)
        self.l.addWidget(self.interpolateButton, rowspace+4, 5,1,4)


        line = QVLine()
        line.setStyleSheet('color:white;')
        self.l.addWidget(line, rowspace,9,6,1)
    
        label = QLabel('Segmentation')#[\u2191 \u2193]')
        label.setStyleSheet(self.headings)
        label.setFont(self.boldfont)
        self.l.addWidget(label, rowspace,10,1,5)
        
        #----------UNETS-----------
        self.GB = QGroupBox("Unets")
        self.GB.setStyleSheet("QGroupBox { border: 1px solid white; color:white; padding: 10px 0px;}")
        self.GBLayout = QGridLayout()
        self.GB.setLayout(self.GBLayout)
        self.GB.setToolTip("Select Unet(s) to be used for segmenting channel")
    
        self.getModels()
        self.modelChoose = QComboBox()
        self.modelChoose.addItems(sorted(self.modelNames, key = lambda x: x[0]))
            #self.modelChoose.setItemChecked(i, False)
        #self.modelChoose.setFixedWidth(180)
        self.modelChoose.setStyleSheet(self.dropdowns)
        self.modelChoose.setFont(self.medfont)
        self.modelChoose.setFocusPolicy(QtCore.Qt.NoFocus)
        self.modelChoose.setCurrentIndex(-1)
        self.GBLayout.addWidget(self.modelChoose, 0,0,1,7)

        self.modelButton = QPushButton(u'run model')
        self.modelButton.clicked.connect(self.computeModels)
        self.GBLayout.addWidget(self.modelButton, 0,7,1,2)
        self.modelButton.setEnabled(False)
        self.modelButton.setStyleSheet(self.styleInactive)
        self.l.addWidget(self.GB, rowspace+1,10,3,5, Qt.AlignTop | Qt.AlignHCenter)

        #------Flourescence Segmentation -----------pp-p-
        self.segButton = QPushButton(u'blob detection')
        self.segButton.setEnabled(False)
        self.segButton.clicked.connect(self.doFlou)
        self.segButton.setStyleSheet(self.styleInactive)
        self.l.addWidget(self.segButton, rowspace+1+2,10,3,5, Qt.AlignTop | Qt.AlignLeft)

        #----------------------------------s-------------

        line = QVLine()
        line.setStyleSheet('color:white;')
        self.l.addWidget(line, rowspace,15,6,1)

        label = QLabel('Display')#[\u2191 \u2193]')
        label.setStyleSheet(self.headings)
        label.setFont(self.boldfont)
        self.l.addWidget(label, rowspace,16,1,5)

        self.contourButton = QCheckBox("Show Contours")
        self.contourButton.setStyleSheet(self.checkstyle)
        self.contourButton.setFont(self.medfont)
        self.contourButton.stateChanged.connect(self.toggleContours)
        self.contourButton.setShortcut(QtCore.Qt.Key_C)
        self.contourButton.setEnabled(False)
        self.l.addWidget(self.contourButton, rowspace+1, 16,1,2)

        self.plotButton = QCheckBox("Show Plot Window")
        self.plotButton.setStyleSheet(self.checkstyle)
        self.plotButton.setFont(self.medfont)
        self.plotButton.stateChanged.connect(self.togglePlotWindow)
        self.plotButton.setShortcut(QtCore.Qt.Key_P)
        self.l.addWidget(self.plotButton, rowspace+1, 18, 1,2)



        self.maskTypeSelect  = MaskTypeButtons(parent=self, row = rowspace+3, col = 16)

        # self.autoSaturationButton = QPushButton("Auto")
        # self.autoSaturationButton.setFixedWidth(45)
        # self.autoSaturationButton.setEnabled(True)
        # self.autoSaturationButton.setStyleSheet(self.styleInactive)
        # self.autoSaturationButton.setEnabled(False)
        # self.autoSaturationButton.clicked.connect(self.resetAutoSaturation)
        # self.l.addWidget(self.autoSaturationButton,rowspace+4, 22,1,1)
        # self.saturationSlider = RangeSlider(self)
        # self.saturationSlider.setMinimum(0)
        # self.saturationSlider.setMaximum(255)
        # self.saturationSlider.setLow(self.saturation[0])
        # self.saturationSlider.setHigh(self.saturation[-1])
        # self.saturationSlider.setTickPosition(QSlider.TicksRight)
        # self.saturationSlider.setFocusPolicy(QtCore.Qt.NoFocus)
        # self.saturationSlider.setEnabled(False)
        # self.l.addWidget(self.saturationSlider, rowspace+4, 22,1,3)
        for i in range(self.mainViewRows):
            self.l.setRowStretch(i, 10)

        self.l.setColumnStretch(20,2)
        self.l.setColumnStretch(0,2)
        self.l.setContentsMargins(0,0,0,0)
        self.l.setSpacing(0)