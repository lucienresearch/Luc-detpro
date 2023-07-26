### detpro.sh
python prompt/run_coco.py train data/ data/ checkpoints/exp fg_bg_5_5_6_end soft 0.5 0.5 0.6 8 end  
python prompt/run_coco.py train data/ data/ checkpoints/exp fg_bg_5_6_7_end soft 0.5 0.6 0.7 8 end  
python prompt/run_coco.py train data/ data/ checkpoints/exp fg_bg_5_7_8_end soft 0.5 0.7 0.8 8 end  
python prompt/run_coco.py train data/ data/ checkpoints/exp fg_bg_5_8_9_end soft 0.5 0.8 0.9 8 end  
python prompt/run_coco.py train data/ data/ checkpoints/exp fg_bg_5_9_10_end soft 0.5 0.9 1.1 8 end  
python prompt/run_coco.py test data/ data/ checkpoints/exp fg_bg_5_10_end soft 0.0 0.5 0.5 checkpoints/exp/fg_bg_5_5_6_endepoch6.pth checkpoints/exp/fg_bg_5_6_7_endepoch6.pth checkpoints/exp/fg_bg_5_7_8_endepoch6.pth checkpoints/exp/fg_bg_5_8_9_endepoch6.pth checkpoints/exp/fg_bg_5_9_10_endepoch6.pth  

data 数据目录  
/home/lzk/research/ovd/Luc-detpro/data  
