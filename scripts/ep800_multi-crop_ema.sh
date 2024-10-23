
# pre-train:
python main.py \
--dataset imagenet \
--epochs 800 \
--lr 0.3 \
--method intl_m \
--data_path ./data/ImageNet/ \
--multi-crop \

# linear classification:
python evaluate.py \
--epochs 100 \
--lr-classifier 0.4 \
--pretrained INTL_multi-crop_ep800_resnet50.pth.tar \
--data_path ./data/ImageNet/ \

# semi-supervised classification:
python evaluate.py \
--epochs 20 \
--lr-classifier 0.2 \
--lr-backbone 0.004 \
--weights finetune \
--train-percent 10 \  #--train-percent 1 \
--pretrained INTL_multi-crop_ep800_resnet50.pth.tar \
--data_path ./data/ImageNet/ \
