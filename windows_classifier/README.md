# Quick description

1. Annotate images using __https://app.cvat.ai/jobs?page=1__ and export the annotations in COCOv1 format;
2. prepare_dataset.py -- convert the resulting dataset to the Huggingface DINOv2 format;
3. train.py -- train the model;
4. visualize.py -- show 5 random images and predictions of the test dataset.