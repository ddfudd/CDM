# Contrastive Distengled Meta-Learning for Signer-Independent Sign Language Translation (ACM MM'21)

This code is based on [Joey NMT](https://github.com/joeynmt/joeynmt) but modified to realize joint continuous sign language recognition and translation. For text-to-text translation experiments, you can use the original Joey NMT framework.
 
## Requirements
* Download the feature files using the `data/download.sh` script.

* [Optional] Create a conda or python virtual environment.

* Install required packages using the `requirements.txt` file.

    `pip install -r requirements.txt`

## Usage (Training)

  `python -m signjoey train configs/sign.yaml` 

! Note that the default data directory is `./data`. If you download them to somewhere else, you need to update the `data_path` parameters in your config file.   

## Usage (Testing)

  `python -m signjoey test configs/sign.yaml`
 
The pre-trained model can be downloaded in this place.

## Reference

Please cite the paper below if you use this code in your research:

    @inproceedings{camgoz2020sign,
      author = {Necati Cihan Camgoz and Oscar Koller and Simon Hadfield and Richard Bowden},
      title = {Sign Language Transformers: Joint End-to-end Sign Language Recognition and Translation},
      booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
      year = {2020}
    }
    
    @inproceedings{jin2021contrastive,
      title={Contrastive Disentangled Meta-Learning for Signer-Independent Sign Language Translation},
      author={Jin, Tao and Zhao, Zhou},
      booktitle={Proceedings of the 29th ACM International Conference on Multimedia},
      pages={5065--5073},
      year={2021}
    }