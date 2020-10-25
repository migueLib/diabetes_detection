from torchvision import transforms

def get_transform(crop_size, resize=(299, 299), grayscale=False):
    
    if grayscale:
        TRANSFORM = transforms.Compose([
            transforms.CenterCrop(crop_size),
            transforms.Resize(resize),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor()
            ])
    else:
        TRANSFORM = transforms.Compose([
            transforms.CenterCrop(crop_size),
            transforms.Resize(resize),
            transforms.ToTensor()
            ])
        
    return TRANSFORM