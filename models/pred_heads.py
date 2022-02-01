import torch
from torch import nn


class PredHeads(nn.Module):
    """ Prediction model for Grounded Situation Recognition"""
    def __init__(self, hidden_dim, num_verb_classes, num_noun_classes):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_verb_classes = num_verb_classes
        self.num_noun_classes = num_noun_classes

        # classifer for verb prediction
        self.verb_classifier = nn.Sequential(nn.Linear(hidden_dim, hidden_dim*2),
                                             nn.ReLU(),
                                             nn.Dropout(0.3),
                                             nn.Linear(hidden_dim*2, self.num_verb_classes))

        # classifiers & predictions for grounded noun prediction
        self.noun_classifier = nn.Sequential(nn.Linear(hidden_dim, hidden_dim*2),
                                             nn.ReLU(),
                                             nn.Dropout(0.3),
                                             nn.Linear(hidden_dim*2, self.num_noun_classes))
        self.bbox_exist_classifier = nn.Sequential(nn.Linear(hidden_dim, hidden_dim*2),
                                                   nn.ReLU(),
                                                   nn.Dropout(0.2),
                                                   nn.Linear(hidden_dim*2, 1))
        self.bbox_regression_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim*2),
                                                  nn.ReLU(),
                                                  nn.Dropout(0.2),
                                                  nn.Linear(hidden_dim*2, hidden_dim*2),
                                                  nn.ReLU(),
                                                  nn.Dropout(0.2),
                                                  nn.Linear(hidden_dim*2, 4))

    def forward_verb(self, vhs):
        assert vhs.dim() == 3 or vhs.dim() == 4
        assert vhs.shape[-1] == self.hidden_dim

        verb_logit = self.verb_classifier(vhs)
        assert verb_logit.shape == torch.Size((*vhs.shape[:-1], self.num_verb_classes))

        return verb_logit

    def forward_role(self, rhs):
        assert rhs.dim() == 3 or rhs.dim() == 4  # now only 4
        assert rhs.shape[-1] == self.hidden_dim

        noun_logit = self.noun_classifier(rhs)
        assert noun_logit.shape == torch.Size((*rhs.shape[:-1], self.num_noun_classes))
        bbox_exist_logit = self.bbox_exist_classifier(rhs)
        assert bbox_exist_logit.shape == torch.Size((*rhs.shape[:-1], 1))
        bbox_coord_logit = self.bbox_regression_head(rhs)
        assert bbox_coord_logit.shape == torch.Size((*rhs.shape[:-1], 4))

        return noun_logit, bbox_exist_logit, bbox_coord_logit

    def forward(self, vhs, rhs):
        verb_logit = self.forward_verb(vhs)
        noun_logit, bbox_exist_logit, bbox_coord_logit = self.forward_role(rhs)

        return verb_logit, noun_logit, bbox_exist_logit, bbox_coord_logit

    # def forward(self, verb_feature, role_features):
    #     bs, hidden_dim = verb_feature.shape
    #     assert hidden_dim == self.hidden_dim
    #     _, num_roles, _ = role_features.shape
    #     assert role_features.shape == torch.Size((bs, num_roles, hidden_dim))

    #     verb_logit = self.verb_classifier(verb_feature)
    #     assert verb_logit.shape == torch.Size((bs, self.num_verb_cclasses))
    #     noun_logit = self.noun_classifier(role_features)
    #     assert noun_logit.shape == torch.Size((bs, num_roles, self.num_noun_classes))
    #     bbox_exist_logit = self.bbox_exist_classifier(role_features)
    #     assert bbox_exist_logit == torch.Size((bs, num_roles, 1))
    #     bbox_coord_logit = self.bbox_regression_head(role_features)
    #     assert bbox_coord_logit == torch.Size((bs, num_roles, 4))

    #     return (verb_logit, noun_logit, bbox_exist_logit, bbox_coord_logit)


def build_pred_heads(args):
    pred_heads = PredHeads(hidden_dim=args.hidden_dim,
                           num_verb_classes=args.num_verbs,
                           num_noun_classes=args.num_nouns, )

    return pred_heads
