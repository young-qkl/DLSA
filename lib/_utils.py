from collections import OrderedDict
import sys
import torch
from torch import nn
from torch.nn import functional as F
from bert.modeling_bert import BertModel


def load_weights(model, load_path):
    dict_trained = torch.load(load_path)['model']
    dict_new = model.state_dict().copy()
    for key in dict_new.keys():
        if key in dict_trained.keys():
            dict_new[key] = dict_trained[key]
    model.load_state_dict(dict_new)
    del dict_new
    del dict_trained
    torch.cuda.empty_cache()
    print('load weights from {}'.format(load_path))
    return model


class _LAVTSimpleDecode(nn.Module):
    def __init__(self, backbone, classifier):
        super(_LAVTSimpleDecode, self).__init__()
        self.backbone = backbone
        self.classifier = classifier

    def forward(self, x, l_feats, l_mask, lang_feat=None, return_feat=False):
        input_shape = x.shape[-2:]
        lang_feat = l_feats[:, :, 0] if lang_feat is None else lang_feat
        features = self.backbone(x, l_feats, l_mask, lang_feat)
        x_c1, x_c2, x_c3, x_c4 = features

        cls_out = self.classifier(
            x_c4, x_c3, x_c2, x_c1,
            lang_feat=lang_feat,
            l_feats=l_feats,
            l_mask=l_mask,
            return_feat=return_feat,
        )

        if return_feat:
            logits, dec_feat = cls_out
            logits = F.interpolate(logits, size=input_shape, mode='bilinear', align_corners=True)
            return {
                'logits': logits,
                'dec_feat': dec_feat,
                'l_feats': l_feats,
                'l_mask': l_mask,
                'lang_feat': lang_feat,
            }

        x = F.interpolate(cls_out, size=input_shape, mode='bilinear', align_corners=True)
        return x


class LAVT(_LAVTSimpleDecode):
    pass


###############################################
# LAVT One: put BERT inside the overall model #
###############################################
class _LAVTOneSimpleDecode(nn.Module):
    def __init__(self, backbone, classifier, args):
        super(_LAVTOneSimpleDecode, self).__init__()

        self.backbone = backbone
        self.classifier = classifier
        self.text_encoder = BertModel.from_pretrained(args.ck_bert)
        self.text_encoder.pooler = None

    def forward(self, x, text, l_mask, lang_feat=None, return_feat=False):
        input_shape = x.shape[-2:]

        ### language inference ###
        l_feats_raw = self.text_encoder(text, attention_mask=l_mask)[0]  # (B, N_l, 768)
        lang_feat = l_feats_raw[:, 0, :] if lang_feat is None else lang_feat  # (B, 768)
        l_feats = l_feats_raw.permute(0, 2, 1)  # (B, 768, N_l)
        l_mask = l_mask.unsqueeze(dim=-1)  # (B, N_l, 1)
        ##########################

        features = self.backbone(x, l_feats, l_mask, lang_feat)
        x_c1, x_c2, x_c3, x_c4 = features

        cls_out = self.classifier(
            x_c4, x_c3, x_c2, x_c1,
            lang_feat=lang_feat,
            l_feats=l_feats,
            l_mask=l_mask,
            return_feat=return_feat,
        )

        if return_feat:
            logits, dec_feat = cls_out
            logits = F.interpolate(logits, size=input_shape, mode='bilinear', align_corners=True)
            return {
                'logits': logits,
                'dec_feat': dec_feat,
                'l_feats': l_feats,
                'l_mask': l_mask,
                'lang_feat': lang_feat,
            }

        x = F.interpolate(cls_out, size=input_shape, mode='bilinear', align_corners=True)
        return x


class LAVTOne(_LAVTOneSimpleDecode):  #change
    pass
