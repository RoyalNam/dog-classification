import torch
import torchvision
from PIL import Image


class PredictionPipeline:
    def __init__(self, filename, device='cpu'):
        self.filename = filename
        self.device = device
        self.transforms: torchvision.transforms.Compose = torch.load(r'model\transforms.pth')
        self.model: torch.nn.Module = torch.load(r'model\updated_model.pth')
        self.model.load_state_dict(torch.load(r'model\model.pth'))

    def predict(self):
        self.model.to(self.device)
        target_img = Image.open(self.filename).convert('RGB')
        img = self.transforms(target_img).unsqueeze(0).to(self.device)

        with open(r'classes.txt', 'r') as file:
            classes = [line.strip() for line in file]

        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(img)
            probabilities = torch.softmax(y_pred, dim=1)
            max_prob, pred_label = torch.max(probabilities, dim=1)

            # Convert the Tensors to Python scalars before returning
            return {"Prob": max_prob.item(), "Label": classes[pred_label.item()]}


if __name__ == '__main__':
    try:
        prediction = PredictionPipeline(filename=r'artifacts\data_ingestion\test\Beagle\01.jpg')
        resutl = prediction.predict()
        print(resutl)
    except Exception:
        raise

