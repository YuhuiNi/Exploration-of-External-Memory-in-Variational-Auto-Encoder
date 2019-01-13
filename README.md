# Exploration of External Memory in Variational Auto-Encoder

## Introduciton

Many algorithms in deep learning are inspired by neuro-biology. In this project, we try to apply Neural Turing Machine, which is also a neural science inspired algorithm, to the context of deep generative model, specifically, Variational Auto-Encoder (VAE). 

Neural Turing Machine (NTM) is one of the earliest works that discuss the effect of external memory in deep learning. Just like human can extract information from memory, we want neural network can gain useful information stored in memory. A NTM is fundamentally composed of a neural network, called the controller, and a 2D matrix called the memory matrix. At each time step, the neural network receives some input from the outside world, and sends some output to the outside world. However, the network also has the ability to read from select memory locations and the ability to write to select memory locations by attentional processes. The intuition is that by putting something important in memory as a kind of template, the neural network only needs to learn how to extract the template properly. NTM shows much better generalization results on learning sequential outputs (such as copying a binary sequence), which can be hard for plain RNN. However, since its appearance in 2014, a good contexts for its application has not been discovered.

VQ-VAE is a recently proposed generative model that is based on VAE. Unlike VAE which uses Gaussian distribution prior and approximate posterior, VQ-VAE learned a discrete latent representation by assuming a delta posterior (q(z|x) is a delta distribution among a fixed set of latent variables) and a separately learned prior (usually learned by PixcelCNN or PixelRNN). Specifically, an input (can be image, sequence, audio or video) is fed into a encoder which maps it to a vector with fixed length, then the vector is compared with a external embedding matrix by finding a nearest neighbor, and then the embedding that is closest to the output of encoder is sent to the decoder. The nearest neighbor operation is not differentiable, so the derivative of this step is manually set to 1 (known as Straight-Through gradient), and the embedding is updated by vector quantization. VQ-VAE learns a discrete representation of inputs, which is a natural fit in many contexts. However, the process is not end-to-end, as there is a non-differentiable operation that needs to be handled separately. 

We combined these two ideas by using a soft, attention-like read/write operation to replace the nearest neighbor and VQ in VQ-VAE. See next section for more details.

## Proposed model

We propose a VAE based model that has an memory matrix as in VQ-VAE, but with soft and differentiable extraction operation. Specifically, we access information by similarity, but instead of choosing the closets embedding, we extract a weighted average of the embedding (implement as the row of memory matrix) with weights given by the similarity. Mathematically, we call the memory matrix M, and let the extraction r be defined as 

<a href="https://www.codecogs.com/eqnedit.php?latex=$$r=\sum_{i}w_iM(i),\&space;\sum&space;w_i=1,\&space;0\leq&space;w_i\geq&space;1$$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$$r=\sum_{i}w_iM(i),\&space;\sum&space;w_i=1,\&space;0\leq&space;w_i\geq&space;1$$" title="$$r=\sum_{i}w_iM(i),\ \sum w_i=1,\ 0\leq w_i\geq 1$$" /></a>

We try to use weights defined in different forms of similarity measure: L2 difference and cosine similarity. Take cosine similarity as an example:

<a href="https://www.codecogs.com/eqnedit.php?latex=$$k[u,v]=\frac{u\cdot&space;v}{||u|||v||}$$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$$k[u,v]=\frac{u\cdot&space;v}{||u|||v||}$$" title="$$k[u,v]=\frac{u\cdot v}{||u|||v||}$$" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=$$w_i=\frac{exp(\beta_tK(z,M(i))}{\sum_jexp(\beta_tK(z,M(j))}$$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$$w_i=\frac{exp(\beta_tK(z,M(i))}{\sum_jexp(\beta_tK(z,M(j))}$$" title="$$w_i=\frac{exp(\beta_tK(z,M(i))}{\sum_jexp(\beta_tK(z,M(j))}$$" /></a>

In other words, the similarity is measured between the output of encoder, denoted by z, and every vector in the memory matrix. Then the similarity measure are processed by a SoftMax to produce the weights. The parameter β is a learned parameter that determined the strictness of SoftMax.
We compared both L2 similarity and cosine similarity, and we found that L2 similarity will produce better results, although in NTM paper, cosine similarity is used. 
After the model is trained, we can reconstruct inputs through encoder, memory addressing and decoder. To fully make this into a generative model, we need a prior so that we can sample from and generate new, unseen samples through decoder. Note that the prior of VQ-VAE is not static as VAE (where prior is standard Gaussian). Instead, it learns a PixelCNN prior over the embeddings. We can also use the same idea here, but we omit it because training PixelCNN is rather complicated. However, if we can do good in reconstruction, we are expected to do good on generation. So we only compare results on reconstruction.


## Results

We trained our model to reconstruct images from CIFAR-10 database. CIFAR-10 consists of 60,000 32 by 32 images from 10 classes. Since it is a generative model, we ignore the label of the image. 

We use a convolutional neural network consists of 3 plain convolutional layers and 2 ResNet blocks as encoder, and we use the converse of encoder as decoder (with deconvolutional layers). The embedding dimension is 64, and we use 512 embeddings. Batch size is 32, and we train our model for 25,000 iterations using Adam optimizer with learning rate 3e-4. All these hyperparameters are the same as in VQ-VAE paper. 
We find that our model converges faster, and results in a lower reconstruction error compared to VQ-VAE. See the following Figure.

![](https://github.com/YuhuiNi/Exploration-of-External-Memory-in-Variational-Auto-Encoder/raw/master/images/convergence_speed.png)

The improvements in reconstruction error possibly means that the vector quantization operation, which is not differentiable, may have negative effect on learning the VAE model. This makes sense, because we have to handle separately and adding extra term to the objective function, while our model only optimize the log likelihood of decoder. 

See following images for examples of reconstruction samples, obtained from VQ-VAE and our model(original picture vs old model vs our model). We can see that both model cab reconstruct the image very well, and there is almost no difference by visual inspection.

![](https://github.com/YuhuiNi/Exploration-of-External-Memory-in-Variational-Auto-Encoder/raw/master/images/result1.jpg)

![](https://github.com/YuhuiNi/Exploration-of-External-Memory-in-Variational-Auto-Encoder/raw/master/images/result2.jpg)



























