## multiHead_relevanceWeight_audio
This project implements a front-end for learning 2-D audio time-frequency representations using multi-head relevance weighting. A 1-D Cosine modulated Gaussian filter-bank layer learns the t-f representation and multi-head relevance subnetworks generates weight masks to enhance the representations. 

### ASC:
Follow the order mentioned below: <br />
1. data_augmentation: <br />
Run all the files.<br />
Ex: 
```
cd data_augmentation/
python gen_extr_wavfiles_2020_add.py
```
2. learn filterbank: <br />
```
cd learn_means/
python train_learn_means_torch_dataloader_no_split.py
```
3. save features using learned filter-bank: <br />
```
cd save_learned_feat
python audio_based/train_save_features.py 
python spec_corr/spec_correction_together.py
```
4. multi-head relevance weighting:<br />
```
cd multi_head/
python train_relWt_cntxt_fcnn.py
```
