# Text Classification Models with Tensorflow
Tensorflow implementation of Text Classification Models.

Implemented Models:

1. Word-level CNN [[paper](https://arxiv.org/abs/1408.5882)]
2. Character-level CNN [[paper](https://arxiv.org/abs/1509.01626)]
3. Very Deep CNN [[paper](https://arxiv.org/abs/1606.01781)]
4. Word-level Bidirectional RNN
5. Attention-Based Bidirectional RNN [[paper](http://www.aclweb.org/anthology/P16-2034)]
6. RCNN [[paper](https://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/download/9745/9552)]

Semi-supervised text classification(Transfer learning) models are implemented at [[dongjun-Lee/transfer-learning-text-tf]](https://github.com/dongjun-Lee/transfer-learning-text-tf).


## Requirements
- Python3
- Tensorflow
- pip install -r requirements.txt


## Usage

### Train
To train classification models for dbpedia dataset,
```
$ python train.py --model="<MODEL>"
```
(\<Model>: word_cnn | char_cnn | vd_cnn | word_rnn | att_rnn | rcnn)

### Test
To test classification accuracy for test data after training,
```
$ python test.py --model="<TRAINED_MODEL>"
```


### Sample Test Results
Trained and tested with dbpedia dataset. (```dbpedia_csv/train.csv```, ```dbpedia_csv/test.csv```)

Model    | WordCNN    | CharCNN   | VDCNN    | WordRNN   | AttentionRNN | RCNN    | *SA-LSTM | *LM-LSTM |
:---:    | :---:      | :---:     | :---:    | :---:     | :---:        | :---:   | :---:    | :---:    |
Accuracy | 98.42%     | 98.05%    | 97.60%   | 98.57%     | 98.61%      | 98.68% | 98.88%    | 98.86%   |

(SA-LSTM and LM-LSTM are implemented at [[dongjun-Lee/transfer-learning-text-tf]](https://github.com/dongjun-Lee/transfer-learning-text-tf).)


## Models

### 1. Word-level CNN
Implementation of [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882).

<img width="600" src="https://user-images.githubusercontent.com/6512394/41590312-b1c28fca-73f1-11e8-9123-e26a03853cc7.png">


### 2. Char-level CNN
Implementation of [Character-level Convolutional Networks for Text Classification](https://arxiv.org/abs/1509.01626).

<img width="600" src="https://user-images.githubusercontent.com/6512394/41590359-c6c94f8a-73f1-11e8-8bda-976ddf09e817.png">


### 3. Very Deep CNN (VDCNN)
Implementation of [Very Deep Convolutional Networks for Text Classification](https://arxiv.org/abs/1606.01781).

<img height="600" src="https://user-images.githubusercontent.com/6512394/41590802-e68f71cc-73f2-11e8-88c6-4bf84bf3410e.png">

### 4. Word-level Bi-RNN
Bi-directional RNN for Text Classification.

1. Embedding layer
2. Bidirectional RNN layer
3. Concat all the outputs from RNN layer
4. Fully-connected layer


### 5. Attention-Based Bi-RNN
Implementation of [Attention-Based Bidirectional Long Short-Term Memory Networks for Relation Classification](http://www.aclweb.org/anthology/P16-2034).

<img width="600" src="https://user-images.githubusercontent.com/6512394/41424160-42520358-7038-11e8-8db0-859346a1fa3a.PNG">

### 6. RCNN
Implementation of [Recurrent Convolutional Neural Networks for Text Classification](https://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/download/9745/9552).

<img width="600" src="https://user-images.githubusercontent.com/6512394/42731530-c4ff3c34-8849-11e8-89fb-a49743255b0a.png">

## References
- [dennybritz/cnn-text-classification-tf](https://github.com/dennybritz/cnn-text-classification-tf)
- [zonetrooper32/VDCNN](https://github.com/zonetrooper32/VDCNN)
