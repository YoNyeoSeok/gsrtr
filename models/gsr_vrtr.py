# ----------------------------------------------------------------------------------------------
# GSRTR Official Code
# Copyright (c) Junhyeong Cho. All Rights Reserved
# Licensed under the Apache License 2.0 [see LICENSE for details]
# ----------------------------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved [see LICENSE for details]
# ----------------------------------------------------------------------------------------------

"""
GSRTR model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn
from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, accuracy_swig, accuracy_swig_bbox)
from .backbone import build_backbone
from .transformer import build_transformer
from .pred_heads import build_pred_heads


class GSR_Transformer(nn.Module):
    """ GSR_Transformer transformer model for Grounded Situation Recognition"""
    def __init__(self, backbone, pred_heads, transformer, num_roles, vidx_ridx):
        """ Initialize the model.
        Parameters:
            - backbone: torch module of the backbone to be used. See backbone.py
            - transformer: torch module of the transformer architecture. See transformer.py
            - pred_heads: torch module of the prediction heads for GSR. See pred_heads.py
            - num_roles: the number of role types
            - vidx_ridx: verb index to role index
        """
        super().__init__()
        self.backbone = backbone
        self.pred_heads = pred_heads
        self.transformer = transformer
        self.num_roles = num_roles
        self.vidx_ridx = vidx_ridx

        # hidden dimension for queries and image features
        hidden_dim = transformer.d_model

        # query embeddings
        self.verb_token = nn.Embedding(1, hidden_dim)
        self.verb_classes_embed = nn.Embedding(self.pred_heads.num_verb_classes, hidden_dim)
        self.role_tokens = nn.Embedding(self.num_roles, hidden_dim)

        # 1x1 Conv
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)

    def topk_verb_role_mask(self, vhs, topk, gt_verb=None):
        num_verb_query, bs, _ = vhs.shape

        with torch.no_grad():
            v_pred = self.pred_heads.forward_verb(vhs)
            assert v_pred.shape == torch.Size((num_verb_query, bs, self.pred_heads.num_verb_classes))
            _, topk_v = v_pred[0].topk(topk)  # we have only one verb query
            assert topk_v.shape == torch.Size((bs, topk))

            role_key_padding_mask = torch.zeros((bs, self.num_roles), dtype=bool, device=vhs.device)
            for b in range(bs):
                for v in topk_v[b]:
                    role_key_padding_mask[b, torch.tensor(self.vidx_ridx[v])] = True

            if gt_verb is not None:
                assert gt_verb.shape == torch.Size((bs, ))
                for b in range(bs):
                    v = gt_verb[b]
                    role_key_padding_mask[b, torch.tensor(self.vidx_ridx[v])] = True

        return role_key_padding_mask

    def forward(self, samples, gt_verb=None):
        """ 
        Parameters:
               - samples: The forward expects a NestedTensor, which consists of:
                        - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
        Outputs:
               - out: dict of tensors. 'pred_verb', 'pred_noun', 'pred_bbox' and 'pred_bbox_conf' are keys
        """
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)
        src, mask = features[-1].decompose()
        assert mask is not None

        # query init
        batch_size = src.shape[0]
        verb_token = self.verb_token.weight.unsqueeze(1).repeat(1, batch_size, 1)
        gt_verb_embed = (None if gt_verb is None
                else self.verb_classes_embed.weight[gt_verb].unsqueeze(0).repeat(self.num_roles, 1, 1))
        role_tokens = self.role_tokens.weight.unsqueeze(1).repeat(1, batch_size, 1)

        # encoder
        memory = self.transformer.forward_encoder(src=self.input_proj(src), mask=mask, pos_embed=pos[-1])

        # decoder
        r2v, vhs, v2r, role_mask, rhs = self.transformer.forward_verb_role_decoder(
                verb_token=verb_token,
                role_tokens=role_tokens,
                topk_verb_role_mask=self.topk_verb_role_mask,
                memory=memory,
                mask=mask,
                pos_embed=pos[-1],
                v2r_gt_verb_embed=gt_verb_embed,
                role_mask_gt_verb=gt_verb,
                )

        num_steps = self.transformer.num_steps
        # hidden_dim = self.transformer.d_model
        # assert r2v.shape == torch.Size((1, num_steps, batch_size, hidden_dim))
        # assert vhs.shape == torch.Size((1, num_steps, batch_size, hidden_dim))
        # assert v2r.shape == torch.Size((self.num_roles, num_steps, batch_size, hidden_dim))
        # assert role_mask.shape == torch.Size((self.num_roles, num_steps, batch_size))
        # assert rhs.shape == torch.Size((self.num_roles, num_steps, batch_size, hidden_dim))

        verb_logit, noun_logit, bbox_exist_logit, bbox_coord_logit = self.pred_heads(vhs, rhs)
        verb_logit = verb_logit.squeeze(0)
        assert verb_logit.shape == torch.Size((num_steps, batch_size, self.pred_heads.num_verb_classes))
        assert noun_logit.shape == torch.Size((self.num_roles, num_steps, batch_size, self.pred_heads.num_noun_classes))
        assert bbox_exist_logit.shape == torch.Size((self.num_roles, num_steps, batch_size, 1))
        assert bbox_coord_logit.shape == torch.Size((self.num_roles, num_steps, batch_size, 4))

        # num steps x batch size x (num tokens) x hidden dim
        out = {}
        out['pred_verb'] = verb_logit
        out['pred_noun'] = noun_logit.permute(1, 2, 0, 3)
        out['pred_bbox_conf'] = bbox_exist_logit.permute(1, 2, 0, 3)
        out['pred_bbox'] = bbox_coord_logit.permute(1, 2, 0, 3).sigmoid()

        # out['r2v'] = r2v.permute(1, 2, 0, 3)
        # out['vhs'] = vhs.permute(1, 2, 0, 3)
        # out['v2r'] = v2r.permute(1, 2, 0, 3)
        # out['role_mask'] = role_mask.permute(1, 2, 0)
        # out['rhs'] = rhs.permute(1, 2, 0, 3)

        return out


class LabelSmoothing(nn.Module):
    """ NLL loss with label smoothing """
    def __init__(self, smoothing=0.0):
        """ Constructor for the LabelSmoothing module.
        Parameters:
                - smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class SWiGCriterion(nn.Module):
    """
    Loss for GSRTR with SWiG dataset, and GSRTR evaluation.
    """
    def __init__(self, weight_dict, SWiG_json_train=None, SWiG_json_eval=None, idx_to_role=None):
        """
        Create the criterion.
        """
        super().__init__()
        self.weight_dict = weight_dict
        self.loss_function = LabelSmoothing(0.2)
        self.loss_function_verb = LabelSmoothing(0.3)
        self.SWiG_json_train = SWiG_json_train
        self.SWiG_json_eval = SWiG_json_eval
        self.idx_to_role = idx_to_role

    def forward(self, outputs, targets, eval=False):
        """ This performs the loss computation, and evaluation of GSRTR.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
             eval: boolean, used in evlauation
        """
        NUM_ANNOTATORS = 3

        # gt verb (value & value-all) acc and calculate noun loss
        assert 'pred_noun' in outputs
        step_batch_noun_loss, step_batch_noun_acc, step_batch_noun_correct = [], [], []
        for pred_noun in outputs['pred_noun']:
            device = pred_noun.device
            batch_size = pred_noun.shape[0]
            batch_noun_loss, batch_noun_acc, batch_noun_correct = [], [], []
            for i in range(batch_size):
                p, t = pred_noun[i], targets[i]
                roles = t['roles']
                num_roles = len(roles)
                role_pred = p[roles]
                role_targ = t['labels'][:num_roles]
                role_targ = role_targ.long()
                acc_res = accuracy_swig(role_pred, role_targ)
                batch_noun_acc += acc_res[1]
                batch_noun_correct += acc_res[0]
                role_noun_loss = []
                for n in range(NUM_ANNOTATORS):
                    role_noun_loss.append(self.loss_function(role_pred, role_targ[:, n]))
                batch_noun_loss.append(sum(role_noun_loss))
            noun_loss = torch.stack(batch_noun_loss).mean()
            noun_acc = torch.stack(batch_noun_acc)
            noun_correct = torch.stack(batch_noun_correct)

            step_batch_noun_loss.append(noun_loss)
            step_batch_noun_acc.append(noun_acc)
            step_batch_noun_correct.append(noun_correct)
        noun_loss = sum(step_batch_noun_loss)
        step_batch_noun_acc = torch.stack(step_batch_noun_acc)
        step_batch_noun_correct = torch.stack(step_batch_noun_correct)

        # top-1 & top 5 verb acc and calculate verb loss
        assert 'pred_verb' in outputs
        gt_verbs = torch.stack([t['verbs'] for t in targets])
        step_verb_loss, step_verb_acc_topk = [], []
        step_batch_noun_acc_topk, step_batch_noun_correct_topk = [], []
        for verb_pred_logits in outputs['pred_verb'].squeeze(2):
            verb_loss = self.loss_function_verb(verb_pred_logits, gt_verbs)
            verb_acc_topk = accuracy(verb_pred_logits, gt_verbs, topk=(1, 5))

            step_verb_loss.append(verb_loss)
            step_verb_acc_topk.append(verb_acc_topk)

            # top-1 & top 5 (value & value-all) acc
            batch_noun_acc_topk, batch_noun_correct_topk = [], []
            for verbs in verb_pred_logits.topk(5)[1].transpose(0, 1):
                batch_noun_acc = []
                batch_noun_correct = []
                for i in range(batch_size):
                    v, p, t = verbs[i], pred_noun[i], targets[i]
                    if v == t['verbs']:
                        roles = t['roles']
                        num_roles = len(roles)
                        role_pred = p[roles]
                        role_targ = t['labels'][:num_roles]
                        role_targ = role_targ.long()
                        acc_res = accuracy_swig(role_pred, role_targ)
                        batch_noun_acc += acc_res[1]
                        batch_noun_correct += acc_res[0]
                    else:
                        batch_noun_acc += [torch.tensor(0., device=device)]
                        batch_noun_correct += [torch.tensor([0, 0, 0, 0, 0, 0], device=device)]
                batch_noun_acc_topk.append(torch.stack(batch_noun_acc))
                batch_noun_correct_topk.append(torch.stack(batch_noun_correct))
            noun_acc_topk = torch.stack(batch_noun_acc_topk)
            noun_correct_topk = torch.stack(batch_noun_correct_topk)  # topk x batch x max roles

            step_batch_noun_acc_topk.append(noun_acc_topk)
            step_batch_noun_correct_topk.append(noun_correct_topk)

        verb_loss = sum(step_verb_loss)
        # step_verb_acc_topk = torch.stack(step_verb_acc_topk)
        step_batch_noun_acc_topk = torch.stack(step_batch_noun_acc_topk)
        step_batch_noun_correct_topk = torch.stack(step_batch_noun_correct_topk)

        # bbox prediction
        assert 'pred_bbox' in outputs
        assert 'pred_bbox_conf' in outputs
        step_batch_bbox_acc, step_batch_bbox_acc_top1, step_batch_bbox_acc_top5 = [], [], []
        step_batch_bbox_loss, step_batch_giou_loss, step_batch_bbox_conf_loss = [], [], []
        for step, (pred_bbox, pred_bbox_conf) in enumerate(zip(outputs['pred_bbox'], outputs['pred_bbox_conf'].squeeze(3))):
            batch_bbox_acc, batch_bbox_acc_top1, batch_bbox_acc_top5 = [], [], []
            batch_bbox_loss, batch_giou_loss, batch_bbox_conf_loss = [], [], []
            for i, t in enumerate(targets):
                roles = t['roles']
                num_roles = len(roles)
                mw, mh, target_bboxes = t['max_width'], t['max_height'], t['boxes'][:num_roles]
                bbox_exist = (target_bboxes[:, 0] != -1)
                num_bbox = bbox_exist.sum().item()
                pb, pbc = pred_bbox[i][roles], pred_bbox_conf[i][roles]
                assert target_bboxes.shape == torch.Size((num_roles, 4))
                assert bbox_exist.shape == torch.Size((num_roles,))
                assert pb.shape == torch.Size((num_roles, 4))
                assert pbc.shape == torch.Size((num_roles,))
                cloned_pb, cloned_target_bboxes = pb.clone(), target_bboxes.clone()

                # bbox conf loss
                loss_bbox_conf = F.binary_cross_entropy_with_logits(pbc,
                                                                    bbox_exist.float(), reduction='mean')
                batch_bbox_conf_loss.append(loss_bbox_conf)

                # bbox reg loss and giou loss
                if num_bbox > 0:
                    loss_bbox = F.l1_loss(pb[bbox_exist], target_bboxes[bbox_exist], reduction='none')
                    loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(box_ops.swig_box_cxcywh_to_xyxy(pb[bbox_exist], mw, mh, device=device),
                                                                           box_ops.swig_box_cxcywh_to_xyxy(target_bboxes[bbox_exist], mw, mh, device=device, gt=True)))
                    batch_bbox_loss.append(loss_bbox.sum() / num_bbox)
                    batch_giou_loss.append(loss_giou.sum() / num_bbox)

                # top1 correct noun & top5 correct nouns
                noun_correct = step_batch_noun_correct[step]
                noun_correct_top1 = step_batch_noun_correct_topk[step][0]
                noun_correct_top5 = step_batch_noun_correct_topk[step].sum(dim=0)

                # convert coordinates
                pb_xyxy = box_ops.swig_box_cxcywh_to_xyxy(cloned_pb, mw, mh, device=device)
                gt_bbox_xyxy = box_ops.swig_box_cxcywh_to_xyxy(cloned_target_bboxes, mw, mh, device=device, gt=True)

                # accuracies
                if not eval:
                    batch_bbox_acc += accuracy_swig_bbox(pb_xyxy.clone(), pbc, gt_bbox_xyxy.clone(), num_roles,
                                                         noun_correct[i], bbox_exist, t, self.SWiG_json_train,
                                                         self.idx_to_role)
                    batch_bbox_acc_top1 += accuracy_swig_bbox(pb_xyxy.clone(), pbc, gt_bbox_xyxy.clone(), num_roles,
                                                              noun_correct_top1[i], bbox_exist, t, self.SWiG_json_train,
                                                              self.idx_to_role)
                    batch_bbox_acc_top5 += accuracy_swig_bbox(pb_xyxy.clone(), pbc, gt_bbox_xyxy.clone(), num_roles,
                                                              noun_correct_top5[i], bbox_exist, t, self.SWiG_json_train,
                                                              self.idx_to_role)
                else:
                    batch_bbox_acc += accuracy_swig_bbox(pb_xyxy.clone(), pbc, gt_bbox_xyxy.clone(), num_roles,
                                                         noun_correct[i], bbox_exist, t, self.SWiG_json_eval,
                                                         self.idx_to_role, eval)
                    batch_bbox_acc_top1 += accuracy_swig_bbox(pb_xyxy.clone(), pbc, gt_bbox_xyxy.clone(), num_roles,
                                                              noun_correct_top1[i], bbox_exist, t, self.SWiG_json_eval,
                                                              self.idx_to_role, eval)
                    batch_bbox_acc_top5 += accuracy_swig_bbox(pb_xyxy.clone(), pbc, gt_bbox_xyxy.clone(), num_roles,
                                                              noun_correct_top5[i], bbox_exist, t, self.SWiG_json_eval,
                                                              self.idx_to_role, eval)

            if len(batch_bbox_loss) > 0:
                bbox_loss = torch.stack(batch_bbox_loss).mean()
                giou_loss = torch.stack(batch_giou_loss).mean()
            else:
                bbox_loss = torch.tensor(0., device=device)
                giou_loss = torch.tensor(0., device=device)

            bbox_conf_loss = torch.stack(batch_bbox_conf_loss).mean()
            bbox_acc = torch.stack(batch_bbox_acc)
            bbox_acc_top1 = torch.stack(batch_bbox_acc_top1)
            bbox_acc_top5 = torch.stack(batch_bbox_acc_top5)

            step_batch_bbox_loss.append(bbox_loss)
            step_batch_giou_loss.append(giou_loss)
            step_batch_bbox_conf_loss.append(bbox_conf_loss)
            step_batch_bbox_acc.append(bbox_acc)
            step_batch_bbox_acc_top1.append(bbox_acc_top1)
            step_batch_bbox_acc_top5.append(bbox_acc_top5)
        bbox_loss = sum(step_batch_bbox_loss)
        giou_loss = sum(step_batch_giou_loss)
        bbox_conf_loss = sum(step_batch_bbox_conf_loss)
        step_batch_bbox_acc = torch.stack(step_batch_bbox_acc)
        step_batch_bbox_acc_top1 = torch.stack(step_batch_bbox_acc_top1)
        step_batch_bbox_acc_top5 = torch.stack(step_batch_bbox_acc_top5)

        out = {}
        # losses
        out['loss_vce'] = verb_loss
        out['loss_nce'] = noun_loss
        out['loss_bbox'] = bbox_loss
        out['loss_giou'] = giou_loss
        out['loss_bbox_conf'] = bbox_conf_loss

        # All metrics should be calculated per verb and averaged across verbs.
        # In the dev and test split of SWiG dataset, there are 50 images for each verb (same number of images per verb).
        # Our implementation is correct to calculate metrics for the dev and test split of SWiG dataset.
        # We calculate metrics in this way for simple implementation in distributed data parallel setting.

        for step, (noun_acc_topk, verb_acc_topk, noun_acc, bbox_acc, bbox_acc_top1, bbox_acc_top5) in enumerate(
                zip(step_batch_noun_acc_topk, step_verb_acc_topk, step_batch_noun_acc, step_batch_bbox_acc, step_batch_bbox_acc_top1, step_batch_bbox_acc_top5)):
            # accuracies (for verb and noun)
            out[f'noun_acc_top1_step{step}'] = noun_acc_topk[0].mean()
            out[f'noun_acc_all_top1_step{step}'] = (noun_acc_topk[0] == 100).float().mean()*100
            out[f'noun_acc_top5_step{step}'] = noun_acc_topk.sum(dim=0).mean()
            out[f'noun_acc_all_top5_step{step}'] = (noun_acc_topk.sum(dim=0) == 100).float().mean()*100
            out[f'verb_acc_top1_step{step}'] = verb_acc_topk[0]
            out[f'verb_acc_top5_step{step}'] = verb_acc_topk[1]
            out[f'noun_acc_gt_step{step}'] = noun_acc.mean()
            out[f'noun_acc_all_gt_step{step}'] = (noun_acc == 100).float().mean()*100
            out[f'mean_acc_step{step}'] = torch.stack([v for k, v in out.items() if 'noun_acc' in k or 'verb_acc' in k]).mean()
            # accuracies (for bbox)
            out[f'bbox_acc_gt_step{step}'] = bbox_acc.mean()
            out[f'bbox_acc_all_gt_step{step}'] = (bbox_acc == 100).float().mean()*100
            out[f'bbox_acc_top1_step{step}'] = bbox_acc_top1.mean()
            out[f'bbox_acc_all_top1_step{step}'] = (bbox_acc_top1 == 100).float().mean()*100
            out[f'bbox_acc_top5_step{step}'] = bbox_acc_top5.mean()
            out[f'bbox_acc_all_top5_step{step}'] = (bbox_acc_top5 == 100).float().mean()*100

        out['noun_acc_top1'] = noun_acc_topk[0].mean()
        out['noun_acc_all_top1'] = (noun_acc_topk[0] == 100).float().mean()*100
        out['noun_acc_top5'] = noun_acc_topk.sum(dim=0).mean()
        out['noun_acc_all_top5'] = (noun_acc_topk.sum(dim=0) == 100).float().mean()*100
        out['verb_acc_top1'] = verb_acc_topk[0]
        out['verb_acc_top5'] = verb_acc_topk[1]
        out['noun_acc_gt'] = noun_acc.mean()
        out['noun_acc_all_gt'] = (noun_acc == 100).float().mean()*100
        out['mean_acc'] = torch.stack([v for k, v in out.items() if 'noun_acc' in k or 'verb_acc' in k]).mean()
        # accuracies (for bbox)
        out['bbox_acc_gt'] = bbox_acc.mean()
        out['bbox_acc_all_gt'] = (bbox_acc == 100).float().mean()*100
        out['bbox_acc_top1'] = bbox_acc_top1.mean()
        out['bbox_acc_all_top1'] = (bbox_acc_top1 == 100).float().mean()*100
        out['bbox_acc_top5'] = bbox_acc_top5.mean()
        out['bbox_acc_all_top5'] = (bbox_acc_top5 == 100).float().mean()*100
        return out


def build(args):
    backbone = build_backbone(args)
    pred_heads = build_pred_heads(args)
    transformer = build_transformer(args)

    model = GSR_Transformer(backbone,
                            pred_heads,
                            transformer,
                            num_roles=args.num_roles,
                            vidx_ridx=args.vidx_ridx,
                            )
    criterion = None

    if not args.inference:
        weight_dict = {'loss_nce': args.noun_loss_coef, 'loss_vce': args.verb_loss_coef,
                       'loss_bbox': args.bbox_loss_coef, 'loss_giou': args.giou_loss_coef,
                       'loss_bbox_conf': args.bbox_conf_loss_coef}

        if not args.test:
            criterion = SWiGCriterion(weight_dict=weight_dict,
                                      SWiG_json_train=args.SWiG_json_train,
                                      SWiG_json_eval=args.SWiG_json_dev,
                                      idx_to_role=args.idx_to_role)
        else:
            criterion = SWiGCriterion(weight_dict=weight_dict,
                                      SWiG_json_train=args.SWiG_json_train,
                                      SWiG_json_eval=args.SWiG_json_test,
                                      idx_to_role=args.idx_to_role)

    return model, criterion
