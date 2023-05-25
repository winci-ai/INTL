
# pre-train and linear classification:
python cifar/main.py \
--dataset cifar10 \
--arch resnet18 \
--projection_size 2048 \
--epochs 1000 \
--lr 0.3 \
--wd 1e-4 \
--method intl \
--data_path ./data/ \
