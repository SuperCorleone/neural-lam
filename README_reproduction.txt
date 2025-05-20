Dataset can be downloaded on: https://nextcloud.liu.se/s/meps

srun -p gpu \
     --nodes=1 \
     --ntasks=1 \
     --cpus-per-task=20 \
     --mem=0 \
     --gres=gpu:1 \
     --time=2-00:00:00 \
     --nodelist=icsnode05 \
     --pty \
     env PL_DISABLE_MPI=1 \
     bash -i

######Pre-process dataset######
### create graph ###
multi-scale: python create_mesh.py --graph multiscale --data data_07
hierarchical: python create_mesh.py --graph hierarchical --hierarchical 1 --levels 3 --dataset data_07
### create graph ###
python create_mesh.py --data data_07
### create graph ###
python create_parameter_weights.py --dataset data_07 --n_workers 16 --batch_size 128

######Training ######

python train_model.py --model graph_efm --graph hierarchical --n_workers 10 --batch_size 2 --n_example_pred 0 --hidden_dim 64 --processor_layers 4 --precision 16-mixed --val_interval 10 --dataset data_07
python train_model.py --model graph_efm --graph multiscale --n_workers 10 --batch_size 2 --n_example_pred 0 --hidden_dim 64 --processor_layers 4 --precision 16-mixed --val_interval 10 --dataset data_07
python train_model.py --model graph_fm --graph hierarchical --n_workers 10 --batch_size 2 --n_example_pred 0 --hidden_dim 64 --processor_layers 4 --precision 16-mixed --val_interval 10 --dataset data_07
python train_model.py --model graphcast --graph multiscale --n_workers 10 --batch_size 2 --n_example_pred 0 --hidden_dim 64 --processor_layers 4 --precision 16-mixed --val_interval 10 --dataset data_07

######Evaluating ######
python train_model.py --eval val --load saved_models/graph_efm-4x64-05_16_15-0855/min_val_loss.ckpt --graph hierarchical --n_workers 10 --dataset data_07 --hidden_dim 64 --processor_layers 4 --precision 16-mixed  --batch_size 2
python train_model.py --eval val --load saved_models/graph_efm-4x64-05_17_17-1154/min_val_loss.ckpt --graph multiscale --n_workers 10 --dataset data_07 --hidden_dim 64 --processor_layers 4 --precision 16-mixed  --batch_size 2
python train_model.py --eval val --load saved_models/graph_fm-4x64-05_17_12-1744/min_val_loss.ckpt --graph hierarchical --n_workers 10 --dataset data_07 --hidden_dim 64 --processor_layers 4 --precision 16-mixed  --batch_size 2 --model graph_fm
python train_model.py --eval val --load saved_models/graphcast-4x64-05_17_12-9462/min_val_loss.ckpt --graph multiscale --n_workers 10 --dataset data_07 --hidden_dim 64 --processor_layers 4 --precision 16-mixed  --batch_size 2 --model graphcast

######Testing ######
python train_model.py --eval test --load saved_models/graph_efm-4x64-05_16_15-0855/min_val_loss.ckpt --graph hierarchical --n_workers 10 --dataset data_07 --hidden_dim 64 --processor_layers 4 --precision 16-mixed  --batch_size 2
python train_model.py --eval test --load saved_models/graph_efm-4x64-05_17_17-1154/min_val_loss.ckpt --graph multiscale --n_workers 10 --dataset data_07 --hidden_dim 64 --processor_layers 4 --precision 16-mixed  --batch_size 2
python train_model.py --eval test --load saved_models/graph_fm-4x64-05_17_12-1744/min_val_loss.ckpt --graph hierarchical --n_workers 10 --dataset data_07 --hidden_dim 64 --processor_layers 4 --precision 16-mixed  --batch_size 2 --model graph_fm
python train_model.py --eval test --load saved_models/graphcast-4x64-05_17_12-9462/min_val_loss.ckpt --graph multiscale --n_workers 10 --dataset data_07 --hidden_dim 64 --processor_layers 4 --precision 16-mixed  --batch_size 2 --model graphcast