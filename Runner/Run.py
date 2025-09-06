import torch
from tqdm import tqdm
from FeatureExtractor.Methods import method 
from Data import Dataset

class patchcore():
    def __init__(self, image_size=224):
        self.image_size = image_size
        self.method = Method.RGBFPFHFeatures()

    def fit(self):
        train_loader = Dataset.DATALOADER("train", img_size=self.image_size)
        for sample, _ in tqdm(train_loader, desc='Extracting train features for rope'):
            self.method.add_sample_to_mem_bank(sample)

        print(f'\n\nRunning coreset for DeepInetFeatures on rope...')
        self.method.run_coreset()

    def evaluate(self):
        image_rocauc, pixel_rocauc, au_pro = None, None, None
        test_loader = Dataset.DATALOADER("test", img_size=self.image_size)

        with torch.no_grad():
            for sample, mask, label in tqdm(test_loader, desc='Extracting test features for rope'):
                self.method.predict(sample, mask, label)

        # ✅ Calculate metrics
        self.method.calculate_metrics()
        image_rocauc = round(self.method.image_rocauc, 3)
        pixel_rocauc = round(self.method.pixel_rocauc, 3)
        au_pro = round(self.method.au_pro, 3)

        print(
            f'Class: rope | DeepInetFeatures -> '
            f'Image ROCAUC: {image_rocauc:.3f}, Pixel ROCAUC: {pixel_rocauc:.3f}, AU-PRO: {au_pro:.3f}'
        )

        return image_rocauc, pixel_rocauc, au_pro
