
# pre-train:
python main.py \
--dataset imagenet \
--epochs 100 \
--lr 0.5 \
--method intl_m \
--data_path ./data/ImageNet/ \
--multi-crop \

# linear classification:
python evaluate.py \
--epochs 100 \
--lr-classifier 0.4 \
--pretrained INTL_multi-crop_ep100_resnet50.pth.tar \
--data_path ./data/ImageNet/ \

# semi-supervised classification:
python evaluate.py \
--epochs 20 \
--lr-classifier 0.2 \
--lr-backbone 0.004 \
--weights finetune \
--train-percent 10 \  #--train-percent 1 \
--pretrained INTL_multi-crop_ep100_resnet50.pth.tar \
--data_path ./data/ImageNet/ \