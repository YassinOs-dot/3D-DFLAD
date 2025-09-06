from sklearn import random_projection
import numpy as np
from sklearn.metrics import roc_auc_score
import timm
import torch
from tqdm import tqdm
from Utils import Util
from Metrics import AU_PRO

class Features(torch.nn.Module):

    def __init__(self, image_size=224, f_coreset=0.1, coreset_eps=0.4):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.deep_feature_extractor = Model(device=self.device)
        self.deep_feature_extractor.to(self.device)
        self.deep_feature_extractor.freeze_parameters(layers=[], freeze_bn=True)

        self.image_size = image_size
        self.f_coreset = f_coreset
        self.coreset_eps = coreset_eps
        self.average = torch.nn.AvgPool2d(3, stride=1)
        self.blur = Util.KNNGaussianBlur(4)
        self.n_reweight = 3
        Util.set_seeds(0)
        self.patch_lib = []
        self.resize = torch.nn.AdaptiveAvgPool2d((24, 24))

        self.image_preds = list()
        self.image_labels = list()
        self.pixel_preds = list()
        self.pixel_labels = list()
        self.gts = []
        self.predictions = []
        self.image_rocauc = 0
        self.pixel_rocauc = 0
        self.au_pro = 0
        self.patch_lib_tensor = None # Added to store the concatenated patch_lib

    def __call__(self, x):
        # Extract the desired feature maps using the backbone model.
        with torch.no_grad():
            feature_maps = self.deep_feature_extractor(x)

        feature_maps = [fmap.to("cpu") for fmap in feature_maps]
        return feature_maps

    def add_sample_to_mem_bank(self, sample):
        raise NotImplementedError

    def predict(self, sample, mask, label):
        raise NotImplementedError

    def compute_s_s_map(self, patch, feature_map_dims, mask, label):
        # Use the concatenated patch_lib_tensor here
        dist = torch.cdist(patch, self.patch_lib_tensor)
        min_val, min_idx = torch.min(dist, dim=1)
        s_idx = torch.argmax(min_val)
        s_star = torch.max(min_val)

        # reweighting
        m_test = patch[s_idx].unsqueeze(0)  # anomalous patch
        m_star = self.patch_lib_tensor[min_idx[s_idx]].unsqueeze(0)  # closest neighbour
        w_dist = torch.cdist(m_star, self.patch_lib_tensor)  # find knn to m_star pt.1
        _, nn_idx = torch.topk(w_dist, k=self.n_reweight, largest=False)  # pt.2
        # equation 7 from the paper
        m_star_knn = torch.linalg.norm(m_test - self.patch_lib_tensor[nn_idx[0, 1:]], dim=1)
        # Softmax normalization trick as in transformers.
        # As the patch vectors grow larger, their norm might differ a lot.
        # exp(norm) can give infinities.
        D = torch.sqrt(torch.tensor(patch.shape[1]))
        w = 1 - (torch.exp(s_star / D) / (torch.sum(torch.exp(m_star_knn / D))))
        s = w * s_star

        # segmentation map
        s_map = min_val.view(1, 1, *feature_map_dims)
        s_map = torch.nn.functional.interpolate(s_map, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False)
        s_map = self.blur(s_map)

        self.image_preds.append(s.numpy())
        self.image_labels.append(label)
        # The mask needs to be resized to the image size before flattening and appending
        resized_mask = torch.nn.functional.interpolate(mask.unsqueeze(0).float().squeeze(1), size=(self.image_size, self.image_size), mode='nearest').squeeze(0).bool()
        self.pixel_preds.extend(s_map.flatten().numpy())
        self.pixel_labels.extend(resized_mask.flatten().numpy())
        self.predictions.append(s_map.detach().cpu().squeeze().numpy())
        self.gts.append(resized_mask.detach().cpu().squeeze().numpy())


    def calculate_metrics(self):
        self.image_preds = np.stack(self.image_preds)
        self.image_labels = np.stack(self.image_labels)
        self.pixel_preds = np.array(self.pixel_preds)
        self.pixel_labels = np.array(self.pixel_labels) # Convert pixel_labels to numpy array here

        self.image_rocauc = roc_auc_score(self.image_labels, self.image_preds)
        self.pixel_rocauc = roc_auc_score(self.pixel_labels, self.pixel_preds)
        self.au_pro, _ = AU_PRO.calculate_au_pro(self.gts, self.predictions)

    def run_coreset(self):

      # Concatenate memory bank to a single tensor
      self.patch_lib = torch.cat(self.patch_lib, 0)

      if self.f_coreset < 1:
          coreset_idx = self.get_coreset_idx_randomp(
            self.patch_lib,
            n=int(self.f_coreset * self.patch_lib.shape[0]),
            eps=self.coreset_eps,
          )
          self.patch_lib = self.patch_lib[coreset_idx]

      # Store the concatenated patch_lib in patch_lib_tensor
      self.patch_lib_tensor = self.patch_lib


    def get_coreset_idx_randomp(self, z_lib, n=1000, eps=0.90, float16=True, force_cpu=False ):

      import os
      from tqdm import tqdm
      from sklearn import random_projection
      import torch

      print(f"   Fitting random projections. Start dim = {z_lib.shape}.")
      try:
        transformer = random_projection.SparseRandomProjection(eps=eps)
        z_lib = torch.tensor(transformer.fit_transform(z_lib))
        print(f"   DONE.                 Transformed dim = {z_lib.shape}.")
      except ValueError:
        print("   Error: could not project vectors. Please increase `eps`.")

      select_idx = 0
      last_item = z_lib[select_idx:select_idx + 1]
      coreset_idx = [torch.tensor(select_idx)]
      min_distances = torch.linalg.norm(z_lib - last_item, dim=1, keepdims=True)

      if float16:
        last_item = last_item.half()
        z_lib = z_lib.half()
        min_distances = min_distances.half()
      if torch.cuda.is_available() and not force_cpu:
        last_item = last_item.to("cuda")
        z_lib = z_lib.to("cuda")
        min_distances = min_distances.to("cuda")

      for i in tqdm(range(n - 1), desc="Coreset Sampling"):
        distances = torch.linalg.norm(z_lib - last_item, dim=1, keepdims=True)
        min_distances = torch.minimum(distances, min_distances)
        select_idx = torch.argmax(min_distances)
        last_item = z_lib[select_idx:select_idx + 1]
        min_distances[select_idx] = 0
        coreset_idx.append(select_idx.to("cpu"))

      return torch.stack(coreset_idx)



class Model(torch.nn.Module):

    def __init__(self, device, backbone_name='wide_resnet50_2', out_indices=(3), checkpoint_path='',
                 pool_last=False):
        super().__init__()
        # Determine if to output features.
        kwargs = {'features_only': True if out_indices else False}
        if out_indices:
            kwargs.update({'out_indices': out_indices})

        self.backbone = timm.create_model(model_name=backbone_name, pretrained=True, checkpoint_path=checkpoint_path,
                                          **kwargs)
        self.device = device
        self.avg_pool = torch.nn.AdaptiveAvgPool2d((1, 1)) if pool_last else None

    def forward(self, x):
        x = x.to(self.device)

        # Backbone forward pass.
        features = self.backbone(x)

        # Adaptive average pool over the last layer.
        if self.avg_pool:
            fmap = features[-1]
            fmap = self.avg_pool(fmap)
            fmap = torch.flatten(fmap, 1)
            features.append(fmap)

        return features


    def freeze_parameters(self, layers, freeze_bn=False):
        """ Freeze resent parameters. The layers which are not indicated in the layers list are freeze. """

        layers = [str(layer) for layer in layers]
        # Freeze first block.
        if '1' not in layers:
            if hasattr(self.backbone, 'conv1'):
                for p in self.backbone.conv1.parameters():
                    p.requires_grad = False
            if hasattr(self.backbone, 'bn1'):
                for p in self.backbone.bn1.parameters():
                    p.requires_grad = False
            if hasattr(self.backbone, 'layer1'):
                for p in self.backbone.layer1.parameters():
                    p.requires_grad = False

        # Freeze second block.
        if '2' not in layers:
            if hasattr(self.backbone, 'layer2'):
                for p in self.backbone.layer2.parameters():
                    p.requires_grad = False

        # Freeze third block.
        if '3' not in layers:
            if hasattr(self.backbone, 'layer3'):
                for p in self.backbone.layer3.parameters():
                    p.requires_grad = False

        # Freeze fourth block.
        if '4' not in layers:
            if hasattr(self.backbone, 'layer4'):
                for p in self.backbone.layer4.parameters():
                    p.requires_grad = False

        # Freeze last FC layer.
        if '-1' not in layers:
            if hasattr(self.backbone, 'fc'):
                for p in self.backbone.fc.parameters():
                    p.requires_grad = False

        if freeze_bn:
            for module in self.backbone.modules():
                if isinstance(module, torch.nn.modules.BatchNorm1d):
                    module.eval()
                if isinstance(module, torch.nn.modules.BatchNorm2d):
                    module.eval()
                if isinstance(module, torch.nn.modules.BatchNorm3d):
                    module.eval()
