import json
import os
import os.path

import torch
import pytorch2caffe as pc
pc.replacepool()

from models.sm6_retinanet import sm6
from models.sm8_retinanet_humanface import sm8
#from models.rpns.basic_rpn import basic_rpn
#from models.sm6_res import SM6Res

from models.filter_nofc import filter_nofc
from models.detector import Detector
#from models.rfcn import RFCN

def load_config(config_path):
    assert(os.path.exists(config_path))
    cfg = json.load(open(config_path, 'r'))
    for key in cfg.keys():
        if key != 'shared' and type(cfg[key]) is dict:
            cfg[key].update(cfg['shared'])
    return cfg


def main():
    print("Loading config...")
    cfg = load_config('models/SM10X_RetinaNet_HumanFace/20180528/config.json')
    print("OK.")

    print("Loading weights...")
    #model = sm6(pretrained=False, cfg=cfg['shared'])
    #model = sm8(pretrained=False, cfg=cfg['shared'])
    #model = SM6Res(pretrained=False, cfg=cfg['shared'])
    #model = basic_rpn(8, 2, 1)

    #model = Detector(cfg)
    model = filter_nofc()
    #model = RFCN(8, 3, 2)

    print(model)

    checkpoint = torch.load(
        'models/SM10X_RetinaNet_HumanFace/20180528/filter_face.pth',
        map_location=lambda storage, loc: storage.cuda(
            torch.cuda.current_device()
        )
    )

    model.eval()
    print("OK.")

    pretrained_dict = checkpoint['state_dict']
    model_dict = model.state_dict()

    #for k in model_dict.keys():
    #    print(k)

    matched_dict = {}
    for k, v in pretrained_dict.items():
        k = k.replace("module.", "", 1)

        #k = k.replace("rcnn.", "", 1)

        """
        k = k.replace("feature_extractor.", "", 1)
        k = k.replace("fpn.", "", 1)
        k = k.replace("rpn.", "", 1)
        k = k.replace("rpn_head.", "", 1)
        k = k.replace("rcnn.", "", 1)
        k = k.replace("conv_embed.", "conv3x3.", 1)

        if k == "conv3x3.weight":
            k = "rpn_head.conv3x3.weight"
        if k == "conv3x3.bias":
            k = "rpn_head.conv3x3.bias"

        if k == "conv_cls.weight":
            k = "rpn_head.conv_cls.weight"
        if k == "conv_cls.bias":
            k = "rpn_head.conv_cls.bias"

        if k == "conv_loc.weight":
            k = "rpn_head.conv_loc.weight"
        if k == "conv_loc.bias":
            k = "rpn_head.conv_loc.bias"
        """

        if k in model_dict and v.size() == model_dict[k].size():
            matched_dict[k] = v
            print('Matched: {}'.format(k))
        else:
            print('NOT Matched: {}'.format(k))

    model_dict.update(matched_dict)
    model.load_state_dict(model_dict)


    pc.convert(model, [(1, 48, 48)], 'outputs/sm10x_humanface/20180528/filter_face')
    #pc.convert(model, [(8, 75, 125), (1000, 5)], 'outputs/ir_model/rfcn')

    print("Converting complete.")

if __name__ == '__main__':
    main()
