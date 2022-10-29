from serving.predictor import Predictor
from PIL import Image
import io


class SeldonAPI:
    def __init__(self):
        self.model = Predictor.default_from_model_registry()

    def predict(self, image: bytes) -> int:
        image = Image.open(io.BytesIO(image))
        output = self.model.predict(image)
        return output
