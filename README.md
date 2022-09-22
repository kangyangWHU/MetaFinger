## This is the implement of the papaer titled "MetaFinger: Fingerprinting the Deep Neural Networks with Meta-training" on IJCAI22

![Framework](https://github.com/kangyangWHU/MetaFinger/blob/main/framework.png)



### Run the Code

#### 0. Requirements

The code was tested using Pytorch 1.8.0, python 3.7.

#### 1. Prepare the models

You can train your models according to the paper or download the models form [Google Driver](https://drive.google.com/drive/folders/10VBwiBBkT-XEJUQVZqO9aT4eeeaI6CnS?usp=sharing), then put them under the main folder.



#### 2. Generate the query set

meta_learning.py is the main file to generate the query set. 

```python
python meta_learning.py
```

#### 3. Eval the query set

input_trans.py obtains the query set accuracy under various input modification operations like image blur or noising.  eval_model.py evaluates the query set accuracy under various model modifications such as fine-tuning and weight pruning.









