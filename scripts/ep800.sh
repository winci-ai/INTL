
# pre-train:
python main.py \
--dataset imagenet \
--epochs 800 \
--lr 0.3 \
--method intl \
--data_path ./data/ImageNet/ \

# linear classification:
python evaluate.py \
--epochs 100 \
--lr-classifier 0.2 \
--pretrained INTL_ep800_resnet50.pth.tar \
--data_path ./data/ImageNet/ \

# semi-supervised classification:
python evaluate.py \
--epochs 20 \
--lr-classifier 0.2 \
--lr-backbone 0.006 \
--weights finetune \
--train-percent 10 \  #--train-percent 1 \
--pretrained INTL_ep800_resnet50.pth.tar \
--data_path ./data/ImageNet/ \