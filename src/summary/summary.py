"""
    Non-Local Spatial Propagation Network for Depth Completion
    Jinsun Park, Kyungdon Joo, Zhe Hu, Chi-Kuei Liu and In So Kweon

    European Conference on Computer Vision (ECCV), Aug 2020

    Project Page : https://github.com/zzangjinsun/NLSPN_ECCV20
    Author : Jinsun Park (zzangjinsun@kaist.ac.kr)

    ======================================================================

    NLSPNSummary implementation
"""


from . import BaseSummary
import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image

cm = plt.get_cmap('plasma')


class Summary(BaseSummary):
    def __init__(self, log_dir, mode, args, loss_name, metric_name):
        assert mode in ['train', 'val', 'test'], \
            "mode should be one of ['train', 'val', 'test'] " \
            "but got {}".format(mode)

        super(Summary, self).__init__(log_dir, mode, args)

        self.log_dir = log_dir
        self.mode = mode
        self.args = args

        self.loss = []
        self.metric = []

        self.loss_name = loss_name
        self.metric_name = metric_name

        self.path_output = None

        # ImageNet normalization
        self.img_mean = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
        self.img_std = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)

    def update(self, global_step, sample, output):
        if self.loss_name is not None:
            self.loss = np.concatenate(self.loss, axis=0)
            self.loss = np.mean(self.loss, axis=0, keepdims=True)

            msg = [" {:<9s}|  ".format('Loss')]
            for idx, loss_type in enumerate(self.loss_name):
                val = self.loss[0, idx]
                self.add_scalar('Loss/' + loss_type, val, global_step)

                msg += ["{:<s}: {:.4f}  ".format(loss_type, val)]

                if (idx + 1) % 10 == 0:
                    msg += ["\n             "]

            msg = "".join(msg)
            print(msg)

            f_loss = open(self.f_loss, 'a')
            f_loss.write('{:04d} | {}\n'.format(global_step, msg))
            f_loss.close()

        if self.metric_name is not None:
            self.metric = np.concatenate(self.metric, axis=0)
            self.metric = np.mean(self.metric, axis=0, keepdims=True)

            msg = [" {:<9s}|  ".format('Metric')]
            for idx, name in enumerate(self.metric_name):
                val = self.metric[0, idx]
                self.add_scalar('Metric/' + name, val, global_step)

                msg += ["{:<s}: {:.4f}  ".format(name, val)]

                if (idx + 1) % 10 == 0:
                    msg += ["\n             "]

            msg = "".join(msg)
            print(msg)

            f_metric = open(self.f_metric, 'a')
            f_metric.write('{:04d} | {}\n'.format(global_step, msg))
            f_metric.close()

        # Un-normalization
        rgb = sample['rgb'].detach()
        rgb.mul_(self.img_std.type_as(rgb)).add_(self.img_mean.type_as(rgb))
        rgb = rgb.data.cpu().numpy()

        if self.args.dep_src in ['slam', 'sgbm']:
            dep = sample['dep'].detach().data.cpu().numpy()
        else:
            dep0 = sample['dep0'].detach().data.cpu().numpy()
            dep1 = sample['dep1'].detach().data.cpu().numpy()

        gt = sample['gt'].detach().data.cpu().numpy()
        pred = output['pred'].detach().data.cpu().numpy()

        if output['confidence'] is not None:
            confidence = output['confidence'].data.cpu().numpy()
        else:
            confidence = np.zeros_like(dep)

        num_summary = rgb.shape[0]
        if num_summary > self.args.num_summary:
            num_summary = self.args.num_summary

            rgb = rgb[0:num_summary, :, :, :]
            if self.args.dep_src in ['slam', 'sgbm']:
                dep = dep[0:num_summary, :, :, :]
            else:
                dep0 = dep0[0:num_summary, :, :, :]
                dep1 = dep1[0:num_summary, :, :, :]
            gt = gt[0:num_summary, :, :, :]
            pred = pred[0:num_summary, :, :, :]
            confidence = confidence[0:num_summary, :, :, :]

        rgb = np.clip(rgb, a_min=0, a_max=1.0)
        if self.args.dep_src in ['slam', 'sgbm']:
            dep = np.clip(dep, a_min=0, a_max=self.args.max_depth)
        else:
            dep0 = np.clip(dep0, a_min=0, a_max=self.args.max_depth)
            dep1 = np.clip(dep1, a_min=0, a_max=self.args.max_depth)
        gt = np.clip(gt, a_min=0, a_max=self.args.max_depth)
        pred = np.clip(pred, a_min=0, a_max=self.args.max_depth)
        confidence = np.clip(confidence, a_min=0, a_max=1.0)

        abs_err = abs(gt - pred)
        abs_err[gt == 0] = 0

        list_img = []
        token_img_list = []

        for b in range(0, num_summary):
            rgb_tmp = rgb[b, :, :, :]
            if self.args.dep_src in ['slam', 'sgbm']:
                dep_tmp = dep[b, 0, :, :]
            else:
                dep_tmp0 = dep0[b, 0, :, :]
                dep_tmp1 = dep1[b, 0, :, :]
            gt_tmp = gt[b, 0, :, :]
            pred_tmp = pred[b, 0, :, :]
            confidence_tmp = confidence[b, 0, :, :]
            abs_err_tmp = abs_err[b, 0, :, :]

            if self.args.dep_src in ['slam', 'sgbm']:
                dep_tmp = 255.0 * dep_tmp / self.args.max_depth
            else:
                dep_tmp0 = 255.0 * dep_tmp0 / self.args.max_depth
                dep_tmp1 = 255.0 * dep_tmp1 / self.args.max_depth
            gt_tmp = 255.0 * gt_tmp / self.args.max_depth
            pred_tmp = 255.0 * pred_tmp / self.args.max_depth
            confidence_tmp = 255.0 * confidence_tmp
            abs_err_tmp = 255.0 * abs_err_tmp / np.max(abs_err_tmp)
                        
            if self.args.dep_src in ['slam', 'sgbm']:
                dep_tmp = cm(dep_tmp.astype('uint8'))
            else:
                dep_tmp0 = cm(dep_tmp0.astype('uint8'))
                dep_tmp1 = cm(dep_tmp1.astype('uint8'))
            gt_tmp = cm(gt_tmp.astype('uint8'))
            pred_tmp = cm(pred_tmp.astype('uint8'))
            confidence_tmp = cm(confidence_tmp.astype('uint8'))
            abs_err_tmp = cm(abs_err_tmp.astype('uint8'))

            if self.args.dep_src in ['slam', 'sgbm']:
                dep_tmp = np.transpose(dep_tmp[:, :, :3], (2, 0, 1))
            else:
                dep_tmp0 = np.transpose(dep_tmp0[:, :, :3], (2, 0, 1))
                dep_tmp1 = np.transpose(dep_tmp1[:, :, :3], (2, 0, 1))
            gt_tmp = np.transpose(gt_tmp[:, :, :3], (2, 0, 1))
            pred_tmp = np.transpose(pred_tmp[:, :, :3], (2, 0, 1))
            confidence_tmp = np.transpose(confidence_tmp[:, :, :3], (2, 0, 1))
            abs_err_tmp = np.transpose(abs_err_tmp[:, :, :3], (2, 0, 1))

            if self.args.dep_src in ['slam', 'sgbm']:
                img = np.concatenate((rgb_tmp, dep_tmp, pred_tmp, gt_tmp,
                                  confidence_tmp, abs_err_tmp), axis=1)
            else:
                img = np.concatenate((rgb_tmp, dep_tmp0, dep_tmp1, pred_tmp, gt_tmp,
                                  confidence_tmp, abs_err_tmp), axis=1)

            list_img.append(img)

            if 'token_coef' in output:
                token_coef = output['token_coef'].detach().data.cpu().numpy()
                N, h, HW, L = token_coef.shape
                H, W, _ = rgb_tmp.shape
                Hb, Wb = 6, 20
                attention_maps = []

                for h in range(heads):
                    for l in range(L):
                        token_coef_tmp = token_coef[b, h, :, l].reshape(Hb, Wb)
                        token_coef_tmp = 255.0 * token_coef_tmp
                        token_coef_tmp = cm(token_coef_tmp.astype('uint8'))
                        attention_maps.append(token_coef_tmp)

                token_img = np.concatenate(attention_maps)
                token_img_list.append(token_img)

        img_total = np.concatenate(list_img, axis=2)
        img_total = torch.from_numpy(img_total)
        self.add_image(self.mode + '/images', img_total, global_step)

        if 'token_coef' in output:
            img_total = np.concatenate(token_img_list, axis=2)
            img_total = torch.from_numpy(img_total)
            self.add_image(self.mode + '/token_coefs', img_total, global_step)

        if self.args.model_name.lower() == 'nlspn':
            self.add_scalar('Etc/gamma', output['gamma'], global_step)


        self.flush()

        # Reset
        self.loss = []
        self.metric = []

    def save(self, epoch, idx, sample, output):
        with torch.no_grad():
            if self.args.save_result_only:
                self.path_output = '{}/{}/epoch{:04d}'.format(self.log_dir,
                                                              self.mode, epoch)
                os.makedirs(self.path_output, exist_ok=True)

                path_save_pred = '{}/{:010d}.png'.format(self.path_output, idx)

                pred = output['pred'].detach()

                pred = torch.clamp(pred, min=0)

                pred = pred[0, 0, :, :].data.cpu().numpy()

                pred = (pred*256.0).astype(np.uint16)
                pred = Image.fromarray(pred)
                pred.save(path_save_pred)
            else:
                # Parse data
                if self.args.model_name.lower() == 'nlspn':
                    guidance = output['guidance'].data.cpu().numpy()
                    offset = output['offset'].data.cpu().numpy()
                    aff = output['aff'].data.cpu().numpy()
                    gamma = output['gamma'].data.cpu().numpy()
                    feat_init = output['pred_init']
                    list_feat = output['pred_inter']

                rgb = sample['rgb'].detach()
                if self.args.dep_src in ['slam', 'sgbm']:
                    dep = sample['dep'].detach()
                else:
                    dep0 = sample['dep0'].detach()
                    dep1 = sample['dep1'].detach()
                pred = output['pred'].detach()
                gt = sample['gt'].detach()

                pred = torch.clamp(pred, min=0)

                # Un-normalization
                rgb.mul_(self.img_std.type_as(rgb)).add_(
                    self.img_mean.type_as(rgb))

                rgb = rgb[0, :, :, :].data.cpu().numpy()
                if self.args.dep_src in ['slam', 'sgbm']:
                    dep = dep[0, 0, :, :].data.cpu().numpy()
                else:
                    dep0 = dep0[0, 0, :, :].data.cpu().numpy()
                    dep1 = dep1[0, 0, :, :].data.cpu().numpy()
                pred = pred[0, 0, :, :].data.cpu().numpy()
                gt = gt[0, 0, :, :].data.cpu().numpy()

                rgb = 255.0*np.transpose(rgb, (1, 2, 0))
                if self.args.dep_src in ['slam', 'sgbm']:
                    dep = dep / self.args.max_depth
                else:
                    dep0 = dep0 / self.args.max_depth
                    dep1 = dep1 / self.args.max_depth
                pred = pred / self.args.max_depth
                pred_gray = pred
                gt = gt / self.args.max_depth

                #calculate abs error
                abs_err = abs(gt - pred)
                abs_err[gt == 0] = 0

                rgb = np.clip(rgb, 0, 256).astype('uint8')
                if self.args.dep_src in ['slam', 'sgbm']:
                    dep = (255.0*cm(dep)).astype('uint8')
                else:
                    dep0 = (255.0*cm(dep0)).astype('uint8')
                    dep1 = (255.0*cm(dep1)).astype('uint8')
                pred = (255.0*cm(pred)).astype('uint8')
                pred_gray = (255.0*pred_gray).astype('uint8')
                gt = (255.0*cm(gt)).astype('uint8')
                abs_err = (255.0*cm(abs_err)).astype('uint8')

                rgb = Image.fromarray(rgb, 'RGB')
                if self.args.dep_src in ['slam', 'sgbm']:
                    dep = Image.fromarray(dep[:, :, :3], 'RGB')
                else:
                    dep0 = Image.fromarray(dep0[:, :, :3], 'RGB')
                    dep1 = Image.fromarray(dep1[:, :, :3], 'RGB')
                pred = Image.fromarray(pred[:, :, :3], 'RGB')
                pred_gray = Image.fromarray(pred_gray)
                gt = Image.fromarray(gt[:, :, :3], 'RGB')
                abs_err = Image.fromarray(abs_err[:, :, :3], 'RGB')

                if self.args.model_name.lower() == 'nlspn':

                    feat_init = feat_init[0, 0, :, :].data.cpu().numpy()
                    feat_init = feat_init / self.args.max_depth
                    feat_init = (255.0*cm(feat_init)).astype('uint8')
                    feat_init = Image.fromarray(feat_init[:, :, :3], 'RGB')

                    for k in range(0, len(list_feat)):
                        feat_inter = list_feat[k]
                        feat_inter = feat_inter[0, 0, :, :].data.cpu().numpy()
                        feat_inter = feat_inter / self.args.max_depth
                        feat_inter = (255.0*cm(feat_inter)).astype('uint8')
                        feat_inter = Image.fromarray(feat_inter[:, :, :3], 'RGB')

                        list_feat[k] = feat_inter

                self.path_output = '{}/{}/epoch{:04d}/{:08d}'.format(
                    self.log_dir, self.mode, epoch, idx)
                os.makedirs(self.path_output, exist_ok=True)

                path_save_rgb = '{}/01_rgb.png'.format(self.path_output)
                if self.args.dep_src in ['slam', 'sgbm']:
                    path_save_dep = '{}/02_dep.png'.format(self.path_output)
                else:
                    path_save_dep0 = '{}/02_dep0.png'.format(self.path_output)
                    path_save_dep1 = '{}/02_dep1.png'.format(self.path_output)

                path_save_init = '{}/03_pred_init.png'.format(self.path_output)
                path_save_pred = '{}/05_pred_final.png'.format(self.path_output)
                path_save_pred_gray = '{}/05_pred_final_gray.png'.format(self.path_output)
                path_save_gt = '{}/06_gt.png'.format(self.path_output)
                path_save_abs_error = '{}/07_abs_error.png'.format(self.path_output)

                rgb.save(path_save_rgb)
                if self.args.dep_src in ['slam', 'sgbm']:
                    dep.save(path_save_dep)
                else:
                    dep0.save(path_save_dep0)
                    dep1.save(path_save_dep1)
                pred.save(path_save_pred)
                pred_gray.save(path_save_pred_gray)
                gt.save(path_save_gt)
                abs_err.save(path_save_abs_error)

                if self.args.model_name.lower() == 'nlspn':

                    feat_init.save(path_save_init)

                    for k in range(0, len(list_feat)):
                        path_save_inter = '{}/04_pred_prop_{:02d}.png'.format(
                            self.path_output, k)
                        list_feat[k].save(path_save_inter)

                    np.save('{}/guidance.npy'.format(self.path_output), guidance)
                    np.save('{}/offset.npy'.format(self.path_output), offset)
                    np.save('{}/aff.npy'.format(self.path_output), aff)
                    np.save('{}/gamma.npy'.format(self.path_output), gamma)
