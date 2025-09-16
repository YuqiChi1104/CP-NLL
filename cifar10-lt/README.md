---
annotations_creators:
- crowdsourced
language_creators:
- found
language:
- en
license: apache-2.0
multilinguality:
- monolingual
size_categories:
- 10K<n<100K
source_datasets:
- cifar10
task_categories:
- image-classification
task_ids: []
paperswithcode_id: cifar-10
pretty_name: Cifar10-LT
dataset_info:
  features:
  - name: img
    dtype: image
  - name: label
    dtype:
      class_label:
        names:
          '0': airplane
          '1': automobile
          '2': bird
          '3': cat
          '4': deer
          '5': dog
          '6': frog
          '7': horse
          '8': ship
          '9': truck
  config_name: cifar10
  splits:
  - name: train
  - name: test
    num_bytes: 22772838
    num_examples: 10000
  download_size: 170498071
---
 
# Dataset Card for CIFAR-10-LT (Long Tail)

## Table of Contents
- [Dataset Description](#dataset-description)
  - [Dataset Summary](#dataset-summary)
  - [Supported Tasks and Leaderboards](#supported-tasks-and-leaderboards)
  - [Languages](#languages)
- [Dataset Structure](#dataset-structure)
  - [Data Instances](#data-instances)
  - [Data Fields](#data-fields)
  - [Data Splits](#data-splits)
- [Additional Information](#additional-information)
  - [Licensing Information](#licensing-information)
  - [Citation Information](#citation-information)
  - [Contributions](#contributions)

## Dataset Description

- **Homepage:** [CIFAR Datasets](https://www.cs.toronto.edu/~kriz/cifar.html)
- **Paper:** [Paper imbalanced example](https://openaccess.thecvf.com/content_CVPR_2019/papers/Cui_Class-Balanced_Loss_Based_on_Effective_Number_of_Samples_CVPR_2019_paper.pdf)
- **Leaderboard:** [r-10](https://paperswithcode.com/sota/long-tail-learning-on-cifar-10-lt-r-10) [r-100](https://paperswithcode.com/sota/long-tail-learning-on-cifar-10-lt-r-100)
 
### Dataset Summary
 
The CIFAR-10-LT imbalanced dataset is comprised of under 60,000 color images, each measuring 32x32 pixels, 
distributed across 10 distinct classes. 
The number of samples within each class decreases exponentially with factors of 10 and 100. 
The dataset includes 10,000 test images, with 1000 images per class, 
and fewer than 50,000 training images. 
Each image is assigned one label.

### Supported Tasks and Leaderboards

- `image-classification`: The goal of this task is to classify a given image into one of 10 classes. The leaderboard is available [here](https://paperswithcode.com/sota/long-tail-learning-on-cifar-10-lt-r-100).

### Languages

English

## Dataset Structure

### Data Instances

A sample from the training set is provided below:

```
{
  'img': <PIL.PngImagePlugin.PngImageFile image mode=RGB size=32x32 at 0x2767F58E080>, 'label': 0
}
```

### Data Fields

- img: A `PIL.Image.Image` object containing the 32x32 image. Note that when accessing the image column: `dataset[0]["image"]` the image file is automatically decoded. Decoding of a large number of image files might take a significant amount of time. Thus it is important to first query the sample index before the `"image"` column, *i.e.* `dataset[0]["image"]` should **always** be preferred over `dataset["image"][0]`
- label: 0-9 with the following correspondence
         0 airplane
         1 automobile
         2 bird
         3 cat
         4 deer
         5 dog
         6 frog
         7 horse
         8 ship
         9 truck

 
### Data Splits
 
|   name   |train|test|
|----------|----:|---------:|
|cifar10|<50000|     10000|
 
### Licensing Information
Apache License 2.0
 

<h2>How to Run</h2>

<h3>Step 1: Generate data</h3>

<p>Modify the parameters in <code>ltdataset_gen.sh</code> file as you need, and then run the sh file.</p>

<pre><code>bash /your/code/file/CP-NLL/ltdataset_gen.sh
</code></pre>

<h3>Step 2: Randomly shuffle</h3>

<p>Use the following command to run the <code>shuffle_arrow.py</code> script and randomly shuffle the generated data.:</p>

<pre><code>python shuffle_arrow.py \
    --input /your/data/file/cifar-10-lt-200/cifar10-train.arrow \
    --output /your/data/file/cifar-10-lt-200/cifar10-train_s \
    --seed 42
</code></pre>




### Citation Information
 
```
@TECHREPORT{Krizhevsky09learningmultiple,
    author = {Alex Krizhevsky},
    title = {Learning multiple layers of features from tiny images},
    institution = {},
    year = {2009}
}
```

### Contributions

Thanks to [@gchhablani](https://github.com/gchablani) and all contributors for adding the original balanced cifar10 dataset.