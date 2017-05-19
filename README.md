# sgd-feedback-diy

This is code for the paper
**[Improving Stochastic Gradient Descent with Feedback](https://arxiv.org/abs/1611.01505)**,
<br>
[Jayanth Koushik](https://www.cs.cmu.edu/~jkoushik)\*,
[Hiroaki Hayashi](https://www.cs.cmu.edu/~hiroakih)\*,
<br>
(\* equal contribution)
<br>

## Usage
All results from the paper, and more are in the `data` folder. For example `data/cnn/cifar10/eve.pkl` has the results for using Eve to optimize a CNN on CIFAR10. The pickle files contain the loss history and cross-validation parameters. Additionally, all results are visualized in a jupyter notebook `src/compare_opts.ipynb`. The fixed models used in the paper are in `src/models.py`. The models are implemented in Keras. The experiments can be run using `src/runexp.py`. Run this script with `--help` as an argument to see the interface. The code for the character language model is in `src/charnn.py`. It is implemented in Theano. A keras implementation of our algorithm Eve is in `src/eve.py`. A theano implementation is also available in `src/theano_utils.py`.

## Scripts

common models or datasets:

python3 runexp.py --optimizer=adam --opt-args='{"lrs":[0.009,0.092],"decays":[0.8,0.9]}' --model=logistic --dataset=mnist --batch-size=120 --epochs=20000 --save-path=PATH_PKL/filename.pkl

--------------------------------------------------------------------
adam,adamax,adagrad,adadelta,rmsprop(without momentums):

python3 runexp.py --optimizer=adam --opt-args='{"lrs":[0.01,0.001,0.0001],"decays":[0,0.01,0.001,0.0001]}' --model=cnn --dataset=cifar10 --batch-size=120 --epochs=20000 --save-path=pkls/cnn_cifar10.pkl

----------------------------------------------------------------------
sgdnesterov,sgdmomentum(with momentums):

python3 runexp.py --optimizer=sgdnesterov --opt-args='{"lrs":[0.01,0.001,0.0001],"momentums":[0.9],"decays":[0,0.01,0.001,0.0001]}' --model=cnn --dataset=cifar10 --batch-size=120 --epochs=20000 --save-path=pkls/cnn_cifar10.pkl

-----------------------------------------------------------------

How to run Eve?

there is a mistake here:

AttributeError: 'float' object has no attribute 'name'

-----------------------------------------------------------------
ps: 运行脚本的编写，刚开始自己不知道的前提下，直接通过在代码中按照自己的想法来改，没有报错，然后注释掉自己的代码，将修改的地方移到脚本，测试OK。代码运行需要python3的支持，同时注意--opt-args参数的设置，因为使用gridSearch，故value是list，同时命令行传递的key是变量名称，故意暴露了自己没有使用过argParser，囧。

## Citation
If you find this code useful, please cite
```
@article{koushik2016improving,
  title={Improving Stochastic Gradient Descent with Feedback},
  author={Koushik, Jayanth and Hayashi, Hiroaki},
  journal={arXiv preprint arXiv:1611.01505},
  year={2016}
}
```


