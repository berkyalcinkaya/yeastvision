
## Overview

### Link to Matlab repository: [Life Cycle Tracking](https://github.com/MirandaLab/Full_Life_Cycle_tracking/tree/main)

Time series images depicting full life cycles were first interpolated using RIFE. Segmentations were performed on the interpolated time series, composed of real and synthetic images, using a cellpose architecture containing the following pre-trained models:
- **ProSeg**: For generalist segmentation.
- **MatSeg**: For segmentation of mating states after cell fusion.
- **SpoSeg**: For spores and early germination stages.
- **BudSeg**: For detecting mother-daughter cell pairs based on their connection at the budneck.

Full life cycle tracking involved the following steps:

### 1. FIEST Tracking
The SpoSeg, MatSeg, and ProSeg segmentations of interpolated time series generated cell masks with unique indexes in each image. For every indexed cell mask at a given time point, the tracking algorithm projected the indexed cell mask from the previous image onto the cell masks in the next segmented image and identified the indexed mask with the highest overlap. Once identified, the index of the cell mask in the next image is replaced by the index of cell masks in the previous image, forcing each cell to acquire a unique index throughout the time series.

### 2. Correction of Incomplete Cell Masks
ProSeg failures to correctly detect mating or sporulation events were identified as discrepancies in the number of cell mask indexes assigned by ProSeg, MatSeg, or SpoSeg. Whenever identified, erroneous ProSeg masks were replaced by the correct MatSeg or SpoSeg mask. After all potential corrections occurred, the interpolated and tracked segmentation time series were downsampled ensuring that the final tracks correspond to the real images. ProSeg cell masks and tracks can be labeled as sporulating or mating by their overlap with SpoSeg or MatSeg tracks.

### 3. Mother-Daughter Cell Pair Identification
The overlap between BudSeg masks and ProSeg masks in tracked segmentations was used to find cell mask index pairs, and the most frequent pairs in a tracked time series were assigned as mother and daughter cells.

### 4. Finding the Indexes of the Proliferative Haploids that Mated to Form a Zygote
Each initial tracked cell mask for mating events, MatSeg, was skeletonized and overlapped with all previous time points in the ProSeg tracks, until finding the first time point where the area covered by the mating event contained two cell mask indices with an overlap ratio greater than 0.25 but less than 4. In case of ascus mating, the operation eventually leads to an overlap of SpoSeg masks, labeling early germination, with MatSeg masks, labeling intra-ascus mating.

### 5. Assign Descendant Cells to an Ancestor Sporulated Cell
The search for descendants was restricted by a watershed algorithm that delineates an area around the centroids of germinating asci where cell masks can be assigned to the different germinating events, including mating events detected by MatSeg.
