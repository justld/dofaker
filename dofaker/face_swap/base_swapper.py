class BaseSwapper:

    def forward(self, img, latent, *args, **kwargs):
        raise NotImplementedError

    def get(self,
            img,
            target_face,
            source_face,
            paste_back=True,
            *args,
            **kwargs):
        raise NotImplementedError
