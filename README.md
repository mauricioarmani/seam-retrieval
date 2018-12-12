# Multimodal Retrieval Using a GAN Text Encoder

## Dependencies
We recommended to use Anaconda for the following packages.

* Python 2.7
* [PyTorch](http://pytorch.org/) (>0.1.12)
* [NumPy](http://www.numpy.org/) (>1.12.1)
* [TensorBoard](https://github.com/TeamHG-Memex/tensorboard_logger)

* Punkt Sentence Tokenizer:
```python
import nltk
nltk.download()
> d punkt
```

## Download data

Download the dataset files and pre-trained models. We use splits produced by [Andrej Karpathy](http://cs.stanford.edu/people/karpathy/deepimagesent/). To use full image encoders, download the images from their original sources [here](http://nlp.cs.illinois.edu/HockenmaierGroup/Framing_Image_Description/KCCA.html), [here](http://shannon.cs.illinois.edu/DenotationGraph/) and [here](http://mscoco.org/).

```bash
wget http://lsa.pucrs.br/jonatas/seam-data/irv2_precomp.tar.gz
wget http://lsa.pucrs.br/jonatas/seam-data/resnet152_precomp.tar.gz
wget http://lsa.pucrs.br/jonatas/seam-data/vocab.tar.gz
```

** Models not avaiable yet.

## Training new models
Run `train.py`:

```bash
python train.py --data_name resnet152_precomp --logger_name runs/model --text_encoder gru --max_violation --lr_update 10 --learning_rate 1e-4 --resume /models/txt_enc.tar --resume2 models/txt_enc_epoch_600.pth
```

## Evaluate pre-trained models

```python
from vocab import Vocabulary
import evaluation
evaluation.evalrank("$RUN_PATH/model_best.pth.tar", data_path="$DATA_PATH", split="test", fold5=True)'
```

To do cross-validation on MSCOCO, pass `fold5=True` with a model trained using 
`--data_name coco`.


## Reference

If you found this code useful, please cite the following papers:

    @article{wehrmann2018fast,
      title={Fast Self-Attentive Multimodal Retrieval},
      author={Wehrmann, Jônatas and Armani, Maurício and More, Martin D. and Barros, Rodrigo C.},
      journal={IEEE Winter Conf. on Applications of Computer Vision (WACV'18)},
      year={2018}
    }
    
    @article{faghri2017vse++,
      title={VSE++: Improved Visual-Semantic Embeddings},
      author={Faghri, Fartash and Fleet, David J and Kiros, Jamie Ryan and Fidler, Sanja},
      journal={arXiv preprint arXiv:1707.05612},
      year={2017}
    }

## License

[Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0)