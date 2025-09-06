from FeatureExtractor import Features
from FeatureExtractor import GETFPFH

class RGBFPFHFeatures(Features.Features):

    def add_sample_to_mem_bank(self, sample):
        ############### RGB PATCH ###############
        rgb_feature_maps = self(sample[0])
        if self.resize is None:
            largest_fmap_size = rgb_feature_maps[0].shape[-2:]
            self.resize = torch.nn.AdaptiveAvgPool2d(largest_fmap_size)
        rgb_resized_maps = [self.resize(self.average(fmap)) for fmap in rgb_feature_maps]
        rgb_patch = torch.cat(rgb_resized_maps, 1)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T

        ############### FPFH PATCH ###############
        fpfh_feature_maps = GETFPFH.GetFPFH(sample[1])
        fpfh_feature_maps_resized = self.resize(self.average(fpfh_feature_maps))
        fpfh_patch = fpfh_feature_maps_resized.reshape(fpfh_feature_maps_resized.shape[1], -1).T

        concat_patch = torch.cat([rgb_patch, fpfh_patch], dim=1)
        self.patch_lib.append(concat_patch)

    def predict(self, sample, mask, label):
        rgb_sample = sample[0]
        pc_sample = sample[1]

        ############### RGB PATCH ###############
        rgb_feature_maps = self(rgb_sample)
        rgb_resized_maps = [self.resize(self.average(fmap)) for fmap in rgb_feature_maps]
        rgb_patch = torch.cat(rgb_resized_maps, 1)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T

        ############### DEPTH PATCH ###############
        depth_feature_maps = GETFPFH.GetFPFH(pc_sample)
        depth_feature_maps_resized = self.resize(self.average(depth_feature_maps))
        depth_patch = depth_feature_maps_resized.reshape(depth_feature_maps_resized.shape[1], -1).T

        concat_patch = torch.cat([rgb_patch, depth_patch], dim=1)

        concat_feature_maps = torch.cat([rgb_resized_maps[0], depth_feature_maps_resized], dim=1)
        self.compute_s_s_map(concat_patch, concat_feature_maps.shape[-2:], mask, label)

