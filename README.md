# xPAND

## Run Pipeline
xPAND can be run on any detector as long as you have the initial detections in COCO format. The Initial Filtering and Class Confirmation steps are in the folder *ClassConfirmation/*. The Box Confirmation and Sample Selection steps are in the folder *BoxConfirmation/*. To run **one iteration** of xPAND you only have to execute:
```
1) The script xpand/ClassConfirmation/run_lc_pipeline.sh, changing the variables BASE_PATH, BASE_NAME and METHOD.
2) The script xpand/BoxConfirmation/run_bc_pipeline.sh, also changing the variables.
```

Make sure that you have previously obtained the initial detections (candidate pseudo-labels in COCO format) and have trained the ClassConfirmation (see *xpand/ClassConfirmation/README.md*) and BoxConfirmation (see *xpand/BoxConfirmation/README.md*) modules. The output will be the final pseudo-labels with which you can perform the final train of your detector.
