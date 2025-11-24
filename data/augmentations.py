from torchvision import transforms
from data.transforms import GaussianBlur, make_normalize_transform

_AUGMENT_CONFIG_PRINTED = False

class DataAugmentationDINO(object):

    def __init__(
        self,
        global_crops_scale=(0.4, 1.0),
        local_crops_scale=(0.05, 0.3),
        local_crops_number=6,
        global_crops_size=96,
        local_crops_size=48,
    ):
        self.global_crops_scale = global_crops_scale
        self.local_crops_scale = local_crops_scale
        self.local_crops_number = local_crops_number
        self.global_crops_size = global_crops_size
        self.local_crops_size = local_crops_size

        #print augmentations configuration only once
        global _AUGMENT_CONFIG_PRINTED
        if not _AUGMENT_CONFIG_PRINTED:
            print("Using data augmentation parameters:")
            print(f" global_crops_scale: {global_crops_scale}")
            print(f" local_crops_scale:  {local_crops_scale}")
            print(f" local_crops_number: {local_crops_number}")
            print(f" global_crops_size:  {global_crops_size}")
            print(f" local_crops_size:   {local_crops_size}")
            _AUGMENT_CONFIG_PRINTED = True

        # ========= geometric augmentations ==========
        self.geometric_augmentation_global = transforms.Compose([
            transforms.RandomResizedCrop(
                global_crops_size,
                scale=global_crops_scale,
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
        ])

        self.geometric_augmentation_local = transforms.Compose([
            transforms.RandomResizedCrop(
                local_crops_size,
                scale=local_crops_scale,
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
        ])

        # ========= color transforms ==========
        color_jittering = transforms.Compose([
            transforms.RandomApply(
                [transforms.ColorJitter(
                    brightness=0.4,
                    contrast=0.4,
                    saturation=0.4,
                    hue=0.1
                )],
                p=0.8,
            ),
            transforms.RandomGrayscale(p=0.2),
        ])


        global_transfo1_extra = GaussianBlur(p=1.0)
        global_transfo2_extra = transforms.Compose([
            GaussianBlur(p=0.1),
            transforms.RandomSolarize(threshold=128, p=0.2),
        ])
        local_transfo_extra = GaussianBlur(p=0.5)

        # ========= normalization ==========
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            make_normalize_transform(),
        ])

        # ========= final pipeline ==========
        self.global_transfo1 = transforms.Compose([
            color_jittering,
            global_transfo1_extra,
            self.normalize,
        ])

        self.global_transfo2 = transforms.Compose([
            color_jittering,
            global_transfo2_extra,
            self.normalize,
        ])

        self.local_transfo = transforms.Compose([
            color_jittering,
            local_transfo_extra,
            self.normalize,
        ])

    def __call__(self, image):
  
        if hasattr(image, "convert"):
            image = image.convert("RGB")

        output = {}

        # ===== two global crops =====
        im1 = self.geometric_augmentation_global(image)
        im2 = self.geometric_augmentation_global(image)

        global_crop_1 = self.global_transfo1(im1)
        global_crop_2 = self.global_transfo2(im2)

        output["global_crops"] = [global_crop_1, global_crop_2]

        output["global_crops_teacher"] = [global_crop_1, global_crop_2]

        # ===== local crops =====
        local_crops = [
            self.local_transfo(self.geometric_augmentation_local(image))
            for _ in range(self.local_crops_number)
        ]
        output["local_crops"] = local_crops

        output["offsets"] = ()

        return output
