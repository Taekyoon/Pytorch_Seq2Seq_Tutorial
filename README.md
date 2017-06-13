# Pytorch Seq2Seq Machine Translation Tutorial
This project is for beginners who want to find an easy way to practice sequence to sequence model with pytorch. The repository gives Machine Translation model practice code to understand sequence to sequence model. 

### What do I need to do in this project?
During following this practice, you need to fill out some code lines to run this projects. The parts you get to fill out the blanks are mostly model parts and training module parts.

### How do I run my model?
You can run this model by using jupyter notebook which is a main practice place.  

### Is there any instructions for practice?
Tutorial directions will be written on jupyter notebook file named '*_practice.ipynb'. 

### What results I can expect? 
When you finish this tutorial, you can see a plot that shows how a loss value goes down and translate evaluation from the model. 

### Are there any solutions for this tutorial?
When you are stucked on this tutorial, you can check '*_completed.py' files which have answers for your practice. 

------------------------------------------------------------------------------------------------------------
## Getting Started 
This setup is assumed that you've already set Python virtualenv. If you haven't set the virtualenv, this may cause some confliction among other modules you've already setup.

```bash
$ git clone https://github.com/Taekyoon/Pytorch_Seq2Seq_Tutorial.git
$ pip install -r requirements.txt
$ cd Pytorch_Seq2Seq_Tutorial/data
$ wget http://www.statmt.org/europarl/v6/fr-en.tgz
$ tar -xvzf fr-en.tgz
$ cd ..
$ jupyter notebook
```

## Dependencies
This is my environment settings while running this project, so this is recommened settings. You don't have to follow this dependencies.
* [Python 3.6.1](https://www.continuum.io/downloads)
* [Pytorch 0.1.12](http://pytorch.org/)

## Reference
* [Translation with a Sequence to Sequence Network and Attention](http://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html), written by [Sean Robertson](https://github.com/spro/practical-pytorch)
