from torchvision import transforms

class BaseHandler:

    # define transformations
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # define transformations
    transform_gray = transforms.Compose([
        transforms.Resize(256),
        transforms.Grayscale(),
        transforms.ToTensor(),
    ])



    @classmethod
    def transform_image(cls,
                        img):

        return cls.transform(img)

    @classmethod
    def transform_image_gray(cls,
                        img):
        return cls.transform_gray(img)

    @classmethod
    def transform_image_object(cls,
                               img,
                               resize_tuple,
                               padding_tuple):

        # define transformations
        transform_objects = transforms.Compose([
            transforms.Resize(resize_tuple),
            transforms.Pad(padding_tuple),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        return transform_objects(img)