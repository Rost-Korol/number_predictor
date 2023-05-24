import torch
from torch import nn
from torchvision.transforms import ToTensor
import torchvision.transforms as TF


# Function to recognize number from input image
def predict_number(img, model):
    model.eval()
    with torch.inference_mode():
        img = torch.unsqueeze(img, dim=0).cuda()
        pred_logits = model(img)
        pred_prob = torch.softmax(pred_logits.squeeze(), dim=0)
        answer = pred_prob.argmax()
    return answer.item()

# Pipeline to prepare image from draw
def prepare_img(img):
    convert_tensor = ToTensor()
    img = convert_tensor(img)
    img = TF.Grayscale()(img)
    inverter = TF.RandomInvert(1)
    img = inverter(img)
    img = TF.functional.resize(img, [28, 28], interpolation=TF.InterpolationMode.NEAREST)
    #img = TF.Resize(size=[28, 28])(img)
    return img

# Pipeline to prepare image from photo
def prepare_photo(photo):
    photo = photo.convert('RGB')
    convert_tensor = ToTensor()
    photo = TF.functional.adjust_brightness(photo, 3)
    photo = TF.functional.adjust_contrast(photo, 3)
    photo = convert_tensor(photo)
    photo = TF.Grayscale()(photo)
    inverter = TF.RandomInvert(1)
    photo = inverter(photo)
    photo = TF.Resize(size=[28, 28])(photo)
    return photo


########################################################################################################################


# Class of a CNN model
class TinyVGG_2(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()

        # Convolutional block 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=input_shape,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        # Convolutional block 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        # Classifier layer
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                in_features=hidden_units * 7 * 7,  # image size after conv and pool
                out_features=output_shape
            ),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.classifier(x)

        return x


# model load function
# load pretrained model
def load_model():
    model = torch.load(f='model/number_predictor.pth')
    return model


