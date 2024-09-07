from timm.utils import ModelEmaV3


class MyModelEma(ModelEmaV3):
    def forward_meta(self):
        return self.module.forward_meta()

    def get_inter_ce_logits(self):
        return self.module.get_inter_ce_logits()
