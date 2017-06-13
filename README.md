# Pytorch Seq2Seq Machine Translation Tutorial
This project is for beginners who want to learn about sequence to sequence model with pytorch. The repository gives Machine Translation model practice code to understand sequence to sequence model. You can run this model by using jupyter notebook which is a main practice place. During following this practice, you need to fill out some code lines to run this projects. The parts you get to fill out the blanks are mostly model parts and training module parts. Tutorial directions will be written on  practice jupyter notebook file. When you finish this tutorial, you can see a plot that shows how a loss value goes down and translate evaluation from the model. When you are stucked on this tutorial there are '*_completed' files which have an answer from your practice. This sources are referred from [this page](http://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html)

------------------------------------------------------------------------------------------------------------
## Getting Started 
This setup is assumed that you've already set Python virtualenv. If you haven't set the virtualenv, this may cause some confliction among other modules you've already setup.

```bash
$ git clone https://github.com/Taekyoon/Pytorch_Seq2Seq_Tutorial.git
$ pip install -r requirements.txt
$ cd Pytorch_Seq2Seq_Tutorial/data
$ wget http://www.statmt.org/europarl/v6/fr-en.tgz
$ cd ..
$ jupyter notebook
```

## Dependencies
* [Python 3.6.1](https://www.continuum.io/downloads)
* [Pytorch 0.1.12](http://pytorch.org/)
