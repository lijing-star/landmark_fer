import torch
import argparse
import numpy as np


def main(opts):
    net = torch.load(opts.model_path)

    inputs1=torch.randn(1,3,112,112).cuda()
    inputs2=np.random.random((1,68,2))
    inputs3=opts
    ## test
    classification = net(inputs1,inputs2,inputs3,mode = "test" )
    print('classification result:')
    print(classification)
    ## train
    # classification,ROI_feature,pred_masked_patch = net(inputs1,inputs2,opts,mode = "train")
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RAF-DB')
    parser.add_argument('--model_path', help='model path', type=str, default='model/model_88.44.pkl')
    opts = parser.parse_args()

    main(opts)
