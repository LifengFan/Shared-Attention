# CVPR2018 - Inferring Shared Attention in Social Scene Videos

Introduction
----

The project is described in our paper [Inferring Shared Attention in Social Scene Videos](https://openaccess.thecvf.com/content_cvpr_2018/papers/Fan_Inferring_Shared_Attention_CVPR_2018_paper.pdf) (CVPR2018).   

This paper addresses a new problem of inferring shared attention in third-person social scene videos. Shared attention is a phenomenon that two or more individuals simultaneously look at a common target in social scenes. Perceiving and identifying shared attention in videos plays crucial roles in social activities and social scene understanding. We propose a spatial-temporal neural network to detect shared attention intervals in videos and predict shared attention locations in frames. In each video frame, human gaze directions and potential target boxes are two key features for spatially detecting shared attention in the social scene. In temporal domain, a convolutional Long ShortTerm Memory network utilizes the temporal continuity and transition constraints to optimize the predicted shared attention heatmap. We collect a new dataset VideoCoAtt from public TV show videos, containing 380 complex video sequences with more than 492,000 frames that include diverse social scenes for shared attention study. Experiments on this dataset show that our model can effectively infer shared attention in videos. We also empirically verify the effectiveness of different components in our model.
![](https://github.com/LifengFan/Shared-Attention/blob/master/doc/cvpr_intro.jpg)  


Dataset
----

You can download the dataset [here](https://docs.qq.com/form/page/DTnpmc2pFeWNVZENx). The dataset is available for free only for research purposes.

Demo
----

Here is a [demo](https://vimeo.com/985435528?share=copy) to show more vivid and dynamic results.


Citation
----

Please cite our paper if you find the project and the dataset useful:


```
@inproceedings{FanCVPR2018,
  title     = {Inferring Shared Attention in Social Scene Videos},
  author    = {Lifeng Fan and Yixin Chen and Ping Wei and Wenguan Wang and Song-Chun Zhu},
  year      = {2018},
  booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)}
}
```
