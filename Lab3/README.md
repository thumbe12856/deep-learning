Training:
```
$ python train.py --id sat --caption_model show_attend_tell --input_json data/cocotalk.json --input_fc_dir data/cocotalk_fc --input_att_dir data/cocotalk_att --input_label_h5 data/cocotalk_label.h5 --batch_size 10 --learning_rate 5e-4 --learning_rate_decay_start 0 --scheduled_sampling_start 0 --checkpoint_path log/sat --save_checkpoint_every 6000 --val_images_use 5000 --max_epochs 1
```

Testing:
```
$ python eval.py --save_weight True --model log_st/model-best.pth --infos_path log_st/infos_st-best.pkl --image_folder data/my_test/ --num_images 10 --beam_size 10
```

Visualiaztion:
```
copy output words to visualiztion.py "words"
```

```
$ python visualiztion.py
```
