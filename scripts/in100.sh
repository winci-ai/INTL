# pre-train:
python main.py \
--dataset in100 \
--arch resnet18 \
--bs 128 \
--projection_size 4096 \
--epochs 400 \
--lr 0.5 \
--wd 2.5e-5 \
--method intl \
--data_path ./data/IN100/ \

# linear classification:
python evaluate.py \
--epochs 100 \
--lr-classifier 0.2 \
--pretrained INTL_in100_ep400_resnet18.pth.tar \
--data_path ./data/IN100/ \
