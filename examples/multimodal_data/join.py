import pandas as pd
from torchvision import datasets

import lotus
from lotus.dtype_extensions import ImageArray
from lotus.models import LM

lm = LM(model="gpt-4o-mini")
lotus.settings.configure(lm=lm)

mnist_data = datasets.MNIST(root="mnist_data", train=True, download=True, transform=None)

images = [image for image, _ in mnist_data]
labels = [label for _, label in mnist_data]

df = pd.DataFrame({"image": ImageArray(images[:5]), "label": labels[:5]})

df2 = pd.DataFrame({"image": ImageArray(images[5:10]), "label": labels[5:10]})

df = df.sem_join(df2, "{image:left} represents the same number as {image:right}", strategy="zs-cot")

print(df)
