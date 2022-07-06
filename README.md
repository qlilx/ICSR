# ICSR

Improving image clustering through sample ranking and its application to remote sensing images

ABSTRACT

Image clustering is a very useful technique that is widely applied to various areas, including remote sensing. Recently, visual representations learning by self-supervised learning have greatly improved the performance of image clustering. To further improve the well-trained clustering models, this paper proposes a novel method by first ranking samples within each cluster based on their confidence belonging to the current cluster and then using the ranking to formulate a weighted cross entropy loss to train the model. For ranking samples, we have developed a method for computing the likelihood of samples belonging to the current clusters based on whether they are situated in densely populated neighbourhoods while for training the model, we have given a strategy for weighting the ranked samples. We present extensive experimental results which demonstrate that the new technique can be used to improve the state-of-the-art image clustering models, achieving accuracy performance gains ranging from $2.1\%$ to $15.9\%$. Performing our method on a variety of datasets from remote sensing, we show that our method can be effectively applied to  remote sensing images.

The pretrained model could be obtained from clustering-based representation learning method https://github.com/qlilx/OTL


PyTorch 1.8.0
CUDA 11.0
Python 3.7
