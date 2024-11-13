import pandas as pd
from torchvision import datasets

import lotus
from lotus.dtype_extensions import ImageArray
from lotus.models import LM

lm = LM(model="gpt-4o-mini")
lotus.settings.configure(lm=lm)

mnist_data = datasets.MNIST(root='mnist_data', train=True, download=True, transform=None)

images = [image for image, _ in mnist_data]
labels = [label for _, label in mnist_data]

df = pd.DataFrame({
    "image": ImageArray(images),
    "label": labels
})

df.sem_filter("{image} represents number 1")