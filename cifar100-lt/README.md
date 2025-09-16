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
- cifar100
task_categories:
- image-classification
task_ids: []
paperswithcode_id: cifar-100
pretty_name: Cifar100-LT
dataset_info:
  features:
  - name: img
    dtype: image
  - name: fine_label
    dtype:
      class_label:
        names:
          '0': apple
          '1': aquarium_fish
          '2': baby
          '3': bear
          '4': beaver
          '5': bed
          '6': bee
          '7': beetle
          '8': bicycle
          '9': bottle
          '10': bowl
          '11': boy
          '12': bridge
          '13': bus
          '14': butterfly
          '15': camel
          '16': can
          '17': castle
          '18': caterpillar
          '19': cattle
          '20': chair
          '21': chimpanzee
          '22': clock
          '23': cloud
          '24': cockroach
          '25': couch
          '26': cra
          '27': crocodile
          '28': cup
          '29': dinosaur
          '30': dolphin
          '31': elephant
          '32': flatfish
          '33': forest
          '34': fox
          '35': girl
          '36': hamster
          '37': house
          '38': kangaroo
          '39': keyboard
          '40': lamp
          '41': lawn_mower
          '42': leopard
          '43': lion
          '44': lizard
          '45': lobster
          '46': man
          '47': maple_tree
          '48': motorcycle
          '49': mountain
          '50': mouse
          '51': mushroom
          '52': oak_tree
          '53': orange
          '54': orchid
          '55': otter
          '56': palm_tree
          '57': pear
          '58': pickup_truck
          '59': pine_tree
          '60': plain
          '61': plate
          '62': poppy
          '63': porcupine
          '64': possum
          '65': rabbit
          '66': raccoon
          '67': ray
          '68': road
          '69': rocket
          '70': rose
          '71': sea
          '72': seal
          '73': shark
          '74': shrew
          '75': skunk
          '76': skyscraper
          '77': snail
          '78': snake
          '79': spider
          '80': squirrel
          '81': streetcar
          '82': sunflower
          '83': sweet_pepper
          '84': table
          '85': tank
          '86': telephone
          '87': television
          '88': tiger
          '89': tractor
          '90': train
          '91': trout
          '92': tulip
          '93': turtle
          '94': wardrobe
          '95': whale
          '96': willow_tree
          '97': wolf
          '98': woman
          '99': worm
  - name: coarse_label
    dtype:
      class_label:
        names:
          '0': aquatic_mammals
          '1': fish
          '2': flowers
          '3': food_containers
          '4': fruit_and_vegetables
          '5': household_electrical_devices
          '6': household_furniture
          '7': insects
          '8': large_carnivores
          '9': large_man-made_outdoor_things
          '10': large_natural_outdoor_scenes
          '11': large_omnivores_and_herbivores
          '12': medium_mammals
          '13': non-insect_invertebrates
          '14': people
          '15': reptiles
          '16': small_mammals
          '17': trees
          '18': vehicles_1
          '19': vehicles_2
  config_name: cifar100
  splits:
  - name: train
  - name: test
    num_bytes: 22605519
    num_examples: 10000
  download_size: 169001437
---
 
# Dataset Card for CIFAR-100-LT (Long Tail)

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
- **Leaderboard:** [r-10](https://paperswithcode.com/sota/long-tail-learning-on-cifar-100-lt-r-10) [r-100](https://paperswithcode.com/sota/long-tail-learning-on-cifar-100-lt-r-100)
 
### Dataset Summary
 
The CIFAR-100-LT imbalanced dataset is comprised of under 60,000 color images, each measuring 32x32 pixels, 
distributed across 100 distinct classes. 
The number of samples within each class decreases exponentially with factors of 10 and 100. 
The dataset includes 10,000 test images, with 100 images per class, 
and fewer than 50,000 training images. 
These 100 classes are further organized into 20 overarching superclasses. 
Each image is assigned two labels: a fine label denoting the specific class, 
and a coarse label representing the associated superclass.

### Supported Tasks and Leaderboards

- `image-classification`: The goal of this task is to classify a given image into one of 100 classes. The leaderboard is available [here](https://paperswithcode.com/sota/long-tail-learning-on-cifar-100-lt-r-100).

### Languages

English

## Dataset Structure

### Data Instances

A sample from the training set is provided below:

```
{
  'img': <PIL.PngImagePlugin.PngImageFile image mode=RGB size=32x32 at 0x2767F58E080>, 'fine_label': 19,
  'coarse_label': 11
}
```

### Data Fields

- `img`: A `PIL.Image.Image` object containing the 32x32 image. Note that when accessing the image column: `dataset[0]["image"]` the image file is automatically decoded. Decoding of a large number of image files might take a significant amount of time. Thus it is important to first query the sample index before the `"image"` column, *i.e.* `dataset[0]["image"]` should **always** be preferred over `dataset["image"][0]`
- `fine_label`: an `int` classification label with the following mapping:

  `0`: apple

  `1`: aquarium_fish

  `2`: baby

  `3`: bear

  `4`: beaver
  
  `5`: bed

  `6`: bee

  `7`: beetle

  `8`: bicycle

  `9`: bottle

  `10`: bowl

  `11`: boy

  `12`: bridge

  `13`: bus

  `14`: butterfly

  `15`: camel

  `16`: can

  `17`: castle

  `18`: caterpillar

  `19`: cattle

  `20`: chair

  `21`: chimpanzee

  `22`: clock

  `23`: cloud

  `24`: cockroach

  `25`: couch

  `26`: cra

  `27`: crocodile

  `28`: cup

  `29`: dinosaur

  `30`: dolphin

  `31`: elephant

  `32`: flatfish

  `33`: forest

  `34`: fox

  `35`: girl

  `36`: hamster

  `37`: house

  `38`: kangaroo

  `39`: keyboard

  `40`: lamp

  `41`: lawn_mower

  `42`: leopard

  `43`: lion

  `44`: lizard

  `45`: lobster

  `46`: man

  `47`: maple_tree

  `48`: motorcycle

  `49`: mountain

  `50`: mouse

  `51`: mushroom

  `52`: oak_tree

  `53`: orange

  `54`: orchid

  `55`: otter

  `56`: palm_tree

  `57`: pear

  `58`: pickup_truck

  `59`: pine_tree

  `60`: plain

  `61`: plate

  `62`: poppy

  `63`: porcupine

  `64`: possum

  `65`: rabbit

  `66`: raccoon

  `67`: ray

  `68`: road

  `69`: rocket

  `70`: rose

  `71`: sea

  `72`: seal

  `73`: shark

  `74`: shrew

  `75`: skunk

  `76`: skyscraper

  `77`: snail

  `78`: snake

  `79`: spider

  `80`: squirrel

  `81`: streetcar

  `82`: sunflower

  `83`: sweet_pepper

  `84`: table

  `85`: tank

  `86`: telephone

  `87`: television

  `88`: tiger

  `89`: tractor

  `90`: train

  `91`: trout

  `92`: tulip

  `93`: turtle

  `94`: wardrobe

  `95`: whale

  `96`: willow_tree

  `97`: wolf

  `98`: woman

  `99`: worm

- `coarse_label`: an `int` coarse classification label with following mapping:

  `0`: aquatic_mammals

  `1`: fish

  `2`: flowers

  `3`: food_containers

  `4`: fruit_and_vegetables

  `5`: household_electrical_devices

  `6`: household_furniture

  `7`: insects

  `8`: large_carnivores

  `9`: large_man-made_outdoor_things

  `10`: large_natural_outdoor_scenes

  `11`: large_omnivores_and_herbivores

  `12`: medium_mammals

  `13`: non-insect_invertebrates

  `14`: people

  `15`: reptiles

  `16`: small_mammals

  `17`: trees

  `18`: vehicles_1

  `19`: vehicles_2


 
### Data Splits
 
|   name   |train|test|
|----------|----:|---------:|
|cifar100|<50000|     10000|
 
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
    --input /your/data/file/cifar-100-lt-50/cifar100-train.arrow \
    --output /your/data/file/cifar-100-lt-20/cifar100-train_s \
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

Thanks to [@gchhablani](https://github.com/gchablani) and all contributors for adding the original balanced cifar100 dataset.