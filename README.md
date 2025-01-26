# Joint_IDSF_Bangla
Dataset and Code for the paper "Leveraging Machine-Generated Data for Joint Intent Detection and Slot Filling in Bangla: A Resource-Efficient Approach"

## Generated Bangla Dataset
The Bangla training and validation datasets have been generated using the Llama 3 70B version.
The test dataset has been manually curated by 4 annotators with a Cohen Kappa score of 0.83.
The dataset can be found here: https://github.com/AHMRezaul/Joint_IDSF_Bangla/tree/main/Dataset-Llama3-70B/Dataset.


The distribution of intents and slots throughout the training, validation, and test sets are provided here: 
![image](https://github.com/user-attachments/assets/b027c490-eb0f-4643-9078-cb70bf78d2a1)

### User Agreement

> ***By downloading [our dataset](https://github.com/AHMRezaul/Joint_IDSF_Bangla/tree/main/Dataset-Llama3-70B/Dataset), USER agrees:***
> * to use the dataset for research or educational purposes only.
> * to **not** distribute the dataset or part of the dataset in any original or modified form.
> * and to cite our paper above whenever the dataset is employed to help produce published results.


## Training and Evaluation
```bash
$ python3 main.py --task {task_name} \
                  --model_type {model_type} \
                  --model_dir {model_dir_name} \
                  --do_train --do_eval \
                  --use_crf

# For Generated dataset
$ python3 main.py --task snips \
                  --model_type bert \
                  --model_dir snips_model \
                  --do_train --do_eval
```

## Prediction
```bash
$ python3 predict.py --input_file {INPUT_FILE_PATH} --output_file {OUTPUT_FILE_PATH} --model_dir {SAVED_CKPT_PATH}
```

## Result
![image](https://github.com/user-attachments/assets/cdab7266-4071-4b55-956a-d1ab8c7c61a8)


## Citation
Please cite the paper "Leveraging Machine-Generated Data for Joint Intent Detection and Slot Filling in Bangla: A Resource-Efficient Approach".

### Acknowledgement
Our code is based on the unofficial implementation of the JointBERT+CRF paper from https://github.com/monologg/JointBERT.
