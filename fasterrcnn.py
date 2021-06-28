import torchvision
import torchvision.models.detection.faster_rcnn


# 2 classes+1 background
def get_model():
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        pretrained=False, num_classes=3
    )
    return model
