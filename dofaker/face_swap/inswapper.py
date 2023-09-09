from insightface import model_zoo
from .base_swapper import BaseSwapper

class InSwapper(BaseSwapper):
    def __init__(self, model_file=None):
        providers = model_zoo.model_zoo.get_default_providers()
        session = model_zoo.model_zoo.PickableInferenceSession(model_file, providers=providers)
        self.swapper = model_zoo.inswapper.INSwapper(model_file=model_file, session=session)

    def forward(self, img, latent):
        return self.swapper.forward(img, latent)

    def get(self, img, target_face, source_face, paste_back=True):
        return self.swapper.get(img, target_face, source_face, paste_back)
