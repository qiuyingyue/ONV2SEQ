ONV2SEQ: Biomimetic Perception Learning for Sketch Generation 
=============================================================
## codebase
  1 files that doesn't work and can be deleted: nn.py, mnist_classification.py

  2 training sketch-rnn: (model.py, sketch_rnn_train.py)
	(1) '''python sketch_rnn_train.py --log_root=xxx --'''
	(2) Data file path: sketch:[data_dir]
  3 training sketch-pix2seq (model_cnn_encoder.py, sketch_rnn_train_image.py)
        (1) '''python sketch_rnn_train_image.py --log_root=/home/qyy/workspace/models/cnn_encoder_5classes --resume_training=False --hparams='{"img_size":64}' '''
	(2)You can also modify hparam(e.g. img_size, data_set) in model_dnn_encoder.py and other parameters(e.g.data_dir, log_root) in sketch_rnn_train_image.py  
	(3) Data file path: sketch:[data_dir]/sketch, images in numpy: [data_dir]/image_64*64
 	    It can be modified in Line 156-157 in sketch_rnn_train_image.py

  4 training sketch-onv2seq (model_dnn_encoder.py, sketch_rnn_train_onv.py)
       Usage: 
	(1)python sketch_rnn_train_onv.py --log_root=/home/qyy/workspace/models/dnn_encoder_5classes --resume_training=False --pretrain_decoder=True --decoder_root=/home/qyy/workspace/backup_models/rnn_encoder_5classes_bs500 --hparams='{"onv_size":10000}'
	(2)python sketch_rnn_train_onv.py --log_root=/home/qyy/workspace/models/dnn_encoder_5classes --resume_training=True
	(3)You can also modify hparam(e.g. onv_size, data_set) in model_dnn_encoder.py and other parameters(e.g.log_root, decoder_root) in sketch_rnn_train_onv.py
	(4) Data file path: sketch:[data_dir]/sketch, onv from left eye: [data_dir]/onv_9936_thick, onv from right eye: [data_dir]/onv_9936_thick_right
         It can be modified in Line 153-155 in sketch_rnn_train_onv.py 
  
  5 onv_process.py: conovert png file to onv

  6 onv_to_binary.py: convert onv to binary value (setting a threshold for all data)
data: processed data for network input

  7 svg2img.py(need python3): 
    Three usages:
    (1)Data preprocessing on original sketch files to obtain conrresponding png/onv/numpy image
    (2)convert .svg to .png from one directory to another
    (3)place multiple images to a grid for display
display_svg: generated sketch result (.svg)
  /home/qyy/workspace/display_svg/image_sequence_0.05  

  8 temp.py: testing file run in spider

  9 data_interpolate.py: 

## Other directory 
### display_image
png version of display_svg
### jupyter_demo
### backup_models
 good trained models for final result (important)
   e.g. dnn_encoder_5classes_pretrainedrnn_binocular: use dnn encoder; use pretrained decoder of sketch-rnn; binocular means onv for both eyes; training batch size is 500 for 5classes model

### models
 trained models for self testing


