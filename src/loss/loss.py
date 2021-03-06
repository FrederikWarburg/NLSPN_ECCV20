"""
    Non-Local Spatial Propagation Network for Depth Completion
    Jinsun Park, Kyungdon Joo, Zhe Hu, Chi-Kuei Liu and In So Kweon

    European Conference on Computer Vision (ECCV), Aug 2020

    Project Page : https://github.com/zzangjinsun/NLSPN_ECCV20
    Author : Jinsun Park (zzangjinsun@kaist.ac.kr)

    ======================================================================

    NLSPNLoss implementation
"""


from . import BaseLoss
import torch


class Loss(BaseLoss):
    def __init__(self, args):
        super(Loss, self).__init__(args)

        self.loss_name = []
        self.epoch_num = 1
        self.log_scale = args.log_scale

        for k, _ in self.loss_dict.items():
            self.loss_name.append(k)

    def compute(self, sample, output):
        loss_val = []

        for idx, loss_type in enumerate(self.loss_dict):
            loss = self.loss_dict[loss_type]
            loss_func = loss['func']
            if loss_func is None:
                continue

            pred = output['pred']
            gt = sample['gt']

            if self.log_scale:
                pred = torch.clamp(pred,0)
                pred = torch.log(pred + 1e-6)
                gt = torch.clamp(gt,0)
                gt = torch.log(gt + 1e-6)
 

            if loss_type.lower() in ['l1', 'l2', 'maskedl1', 'maskedl2', 'maskedsmoothl1']:
                loss_tmp = loss_func(pred, gt)
                if 'pred_rgb' in  output:
                    pred_rgb = output['pred_rgb']
                    if self.log_scale:
                        pred_rgb = torch.clamp(pred_rgb,0)
                        pred_rgb = torch.log(pred_rgb + 1e-6) 
                    loss_tmp += loss_func(pred_rgb, gt)

            elif loss_type.lower() in ['confl2', 'confl1', 'conf', 'maskedprobexp','maskedprob']:
                cout = output['confidence']
                loss_tmp = loss_func(pred, gt, cout)
                if 'pred_rgb' in  output:
                    pred_rgb = output['pred_rgb']
                    if self.log_scale:
                        pred_rgb = torch.clamp(pred_rgb,0)
                        pred_rgb = torch.log(pred_rgb + 1e-6) 
                    cout_rgb = output['confidence_rgb'] 
                    loss_tmp += loss_func(pred_rgb, gt, cout_rgb)

            elif loss_type.lower() in ['confdecay', 'confdecaymse']:
                cout = output['confidence']
                loss_tmp = loss_func(pred, gt, cout, self.epoch_num)
                if 'pred_rgb' in  output:
                    pred_rgb = output['pred_rgb']
                    if self.log_scale:
                        pred_rgb = torch.clamp(pred_rgb,0)
                        pred_rgb = torch.log(pred_rgb + 1e-6) 
                    cout_rgb = output['confidence_rgb'] 
                    loss_tmp += loss_func(pred_rgb, gt, cout_rgb, self.epoch_num)

            elif loss_type.lower() in ['inputoutput']:
                inputs = output['dep']
                loss_tmp = loss_func(pred, gt, inputs)
                if 'pred_rgb' in  output:
                    pred_rgb = output['pred_rgb']
                    if self.log_scale:
                        pred_rgb = torch.clamp(pred_rgb,0)
                        pred_rgb = torch.log(pred_rgb + 1e-6) 
                    loss_tmp += loss_func(pred_rgb, gt, inputs)
            else:
                raise NotImplementedError

            loss_tmp = loss['weight'] * loss_tmp
            loss_val.append(loss_tmp)

        loss_val = torch.stack(loss_val)

        loss_sum = torch.sum(loss_val, dim=0, keepdim=True)

        loss_val = torch.cat((loss_val, loss_sum))
        loss_val = torch.unsqueeze(loss_val, dim=0).detach()

        return loss_sum, loss_val
