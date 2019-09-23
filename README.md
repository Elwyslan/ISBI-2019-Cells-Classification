# ISBI2019_Classification Normal-Malignant Cells WBC Images

**Author:** José Elwyslan Maurício de Oliveira\
**University:** Federal University of Sergipe - Brazil\
**Email:** jose.oliveira@dcomp.ufs.br\
**Codalab User ID:** Elwyslan\

Competition @ <https://competitions.codalab.org/competitions/20395>

**Aim**:
Classification of leukemic B-lymphoblast cells from normal B-lymphoid precursors from blood smear microscopic images. In this challenge, a dataset of cells with labels (normal versus malignant) will be provided to train machine learning based classifier to identify normal cells from leukemic blasts (malignant cells). These cells  have been segmented from the images after those images have been stain normalized. The overall size of the images were 2560x1920, while a single cell image is roughly of the size of 300x300 pixels. The images are representative of images in the real-world because these contain some staining noise and illumination errors, although these errors have largely been fixed by us via our own inhouse method on stain normalization [1,2].

**Usage**:\
1 - Execute "A_rawDataHandler.py" to check train folder data.\
2 - Execute "B_dataAugmentation.py" to create "data" folder.\
3 - Check "classifiers/" folder.\

**References and credits**:\
[1] Anubha Gupta, Rahul Duggal, Ritu Gupta, Lalit Kumar, Nisarg Thakkar, and Devprakash Satpathy, “GCTI-SN: Geometry-Inspired Chemical and Tissue Invariant Stain Normalization of Microscopic Medical Images,” under review.\
[2] Ritu Gupta, Pramit Mallick, Rahul Duggal, Anubha Gupta, and Ojaswa Sharma, "Stain Color Normalization and Segmentation of Plasma Cells in Microscopic Images as a Prelude to Development of Computer Assisted Automated Disease Diagnostic Tool in Multiple Myeloma," 16th International Myeloma Workshop (IMW), India, March 2017.\
