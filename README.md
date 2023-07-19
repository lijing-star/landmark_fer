# Facial expression recognition based on key region masked and reconstruction
Our model is implemented in Pytorch

## 1. Contents:
    1. The LandmarkNet folder contains our main model, which is composed of a CNN backbone and a Transformer encoder.
    2. The trained code has been uploaded to the network disk, you can click the link below to download it.
      link：https://pan.baidu.com/s/1en6eBRBYd69plzOy8Yiz4Q 
      password：wejs

## 2. Model usage

   We provide an example for understanding our model. 

   * inputs1 is a Tensor with shape (1,3,112,112), representing a RGB face image. It is randomly generated by torch.randn() function, with values ranging from -1 to 1.
   * inputs2 is an array with shape (1,68,2), representing 68 facial landmarks’ coordinates. It is randomly generated by np.random.random() function, with values ranging from 0 to 1.
   * inputs3 is an object containing model parameters, such as model_path.
   * mode is a string, indicating the running mode of the model. It can be “train” or “test”.
   
   You can use the example data in the example.py file to understand our model. You can use the following command:
   ```
   python example.py --model_path=<model_name>
   For example: python example.py --model_path=model/model_88.44.pkl
   ```
   You can also change the model name according to your needs. The model name should match the one in the model folder.

## 3. Training
   If you want to train our model, you need to prepare your own training data, labels and facial landmarks. You should change the train_data, train_label and train_point arguments to match your own data paths. The same applies to the test data, labels and landmarks. You can use the following command to start training:
   
   ```
   python train.py --model_path model --train_data data/train_data --train_label data/train_label --train_point data/train_point --test_data data/test_data --test_label data/test_label --test_point data/test_point --batch_size 16 --epochs 80 
   ```

   You can also change the other parameters according to your needs. The specific parameters settings can be found in our paper.
