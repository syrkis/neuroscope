---
title: Decoding the Brain's Visual Encoding System Using Multimodal Deep Learning
author: Noah Syrkis
geometry: margin=3cm
fontsize: 12pt
date: \today
---

## Abstract
pandoc list_2.md --bibliography=biblio.bib --citeproc -o list_2.pdf
Understanding how the brain encodes visual information is a key challenge in neuroscience. In this project, we attempt to address this challenge by constructing a multimodal encoding model based on the Algonauts Project 2023 dataset. In addition to the dataset's image modality, we incorporate a semantic feature vector that describes object categories contained in the image shown to the subject during the functional Magnetic Resonance Imaging (fMRI) data collection. We combine various linear modules to construct two models: one predicting the fMRI data from both the associated image and the image's associated semantic feature vector; the other predicting both the fMRI data and the semantic vector from the image alone. Bayesian hyperparameter optimization suggests that the latter approach could potentially enhance model performance during inference without increasing the number of parameters. The model's performance was evaluated using a 5-fold cross-validation strategy and the median Pearson correlation coefficient as the metric. The code for this project is accessible at github.com/syrkis/neuroscope, and training logs are available at wandb.ai/syrkis/neuroscope.

## Introduction

Visual processing is the principal modality through which we interact and decipher our environment. Over the years, substantial progress has been made in understanding how the brain processes visual information, with even surprising parallels observed between artificial and biological vision processing [Cite]. However, as reality can exhibit extraordinarily different visual fingerprints—from simple geometric shapes to complex landscapes and visual noise—any system capable of visual perception is necessarily complicated. Fully capturing this complexity and intricacy remains a challenge. It is this challenge that is the focus of the 2023 Algonauts Project^[[http://algonauts.csail.mit.edu/](http://algonauts.csail.mit.edu/)]. The Algonauts Project's 2023 dataset is based on the Natural Scenes Dataset (NSD), which couples images from the Common Objects in Context (COCO) dataset [@lin2015] with fMRI responses to those images from various participants.

Neuroimaging techniques like fMRI have facilitated valuable insights into the neural correlates of visual perception. However, the potential of these techniques has been somewhat constrained by computational model limitations and the expense and time required to collect large-scale fMRI datasets. Amid these challenges, deep learning has proven to be a powerful tool, that has facilitated a better understanding and emulation of human visual perception. Recent efforts to incorporate multimodality into deep learning models have opened promising avenues to bridge the gap between computational models and the brain's complexity.

The experiment presented here explores how an additional modality might contribute to developing a model of the brain's visual encoding system, _without_ a large increase in complexity/parameter count. The additional modality used here is a semantic feature vector, derived from the COCO dataset, describing the object categories contained in each image. The two models we developed are tasked with 1) predicting the brain response given the image and knowledge of what is in the image, and 2) predicting the brain response and the semantic contents of the image. The second model is there as an additional source of potential confirmation of the usefulness of adding a second modality. The first model is the heart of our experiment, allowing us to explore the question: "_How does the inclusion of an additional modality have on the performance of a brain encoding model during inference?_" It is to attempt an answer to this question that we have constructed the two models and their corresponding unimodal baselines.

## Literature review

#### Visual information processing

Visual information processing, characterized by its hierarchical nature and intricate interconnectivity, plays a vital role in our understanding of the brain and perception. Traditionally, the process is categorized into low, mid, and high-level processing, focusing respectively on elementary visual features, their conjunctions, and abstract representations (@gonzalez-casillas2018,@groen2017). However, the hierarchical categorization is insufficient to capture the full complexity of real-world scene perception. It underrepresents the multimodal and interconnected nature of visual perception, particularly when processing complex stimuli such as natural scenes (@allen2022, @groen2017). 

The development of fMRI has opened up the possibility to examine and visualize brain activity associated with visual perception in real-time(@allen2022, @haxby2001). However, despite these advances, the understanding of the intricate interconnectivity in visual perception, particularly for natural scenes, remains limited. Furthermore, it is increasingly evident that a holistic understanding of visual perception requires the acquisition and analysis of massive amounts of data (@chang2019, @allen2022). To meet this need for data-intensive analysis in the field of visual neuroscience, deep learning has emerged as a promising tool.

#### Deep learning through visual encoding models

The power of deep learning models in the field of neuroscience is attributed to their ability to process and learn about high volumes of data, their inherent flexibility, and their structure, which is inspired by mirroring the brain’s hierarchical organization ((@kriegeskorte2015); (@kell2018)). In this field, deep neural networks (DNNs), particularly convolutional neural networks (CNNs), have been extensively used to predict brain activity in response to visual stimuli, known as visual encoding models. Findings from several studies demonstrate that DNNs and CNNs can accurately predict neural responses to various visual stimuli ((khaligh-razavi2014), (yamins2014), (cichy2016)). For instance, Zheng et al. (2021) applied a convolutional recurrent neural network (CRNN) to decipher the computational elements of the retinal circuit involved in interpreting natural scenes. Findings showed that recurrent spatiotemporal receptive fields of ganglion cells were key in encoding dynamic visual scenes and that the inherent recurrence of the model enhanced the prediction of the neural response (@zheng2021). 


Despite the richness of information available in fMRI data, remarkably few studies have exploited deep learning encoding models to predict associated brain responses. One study, by Güçlü and van Gerven (2015), utilized a DNN trained to predict object categories of over a million natural images and then used it to predict fMRI BOLD responses to complex naturalistic stimuli. The study identified a gradient of increasing complexity in the ventral visual pathway and found that the receptive fields in this region are attuned to object categorization (güçlü2015). In another study, Zhang et al (2019), constructed a new visual encoding model that was based on a unique combination of techniques: it employed transfer learning to use a pre-trained Deep Neural Network (AlexNet) and used a non-linear mapping to translate visual features into brain activity. The researchers found that their model yielded significant predictions for over 20% of the voxels in the early visual area and resulted in the outperformance of conventional linear mapping models, offering a new approach to leverage pre-trained visual features in brain activity prediction (@zhang2019). Notably, these DNNbased models can even outperform traditional hand-engineered models ((guclu2015), (kell2018), (@zhang2019)), signifying a substantial leap forward in uncovering the complexity of visual information processing.

#### Multi modality in visual perception and deep learning 

Most current models rely on single-modal input data, typically the visual stimuli themselves. Yet, the evermore revealed complexity of visual perception indicates a need for a shift towards multimodal deep learning models to encode brain activity.  

Multimodality, in the context of neuroscience and cognitive science, refers to the brain's ability to integrate and process information through multiple sensory systems by which humans perceive and interact with the world ((parcalabescu2021); (@gibbons2012)). In the context of machine learning, multimodality carries a similar theme but is applied to data. It refers to utilizing and integrating multiple information sources to enhance algorithms’ performance. Several studies have pointed out that multimodal learning can help build more robust models that provide richer information about underlying data patterns and create more complex feature representations ((@ngiam2011); (@gu2017)). As visual information processing involves a complex interplay of interconnected factors and systems (@groen2017), infusing deep learning models with a multimodal approach could provide a more comprehensive and rich perspective. Few studies have investigated the potential of multimodal deep learning for predicting brain activity. One study by Horikawa and Kamitani (2017) combined deep neural networks (DNN) and fMRI to predict and decode perceived and imagined objects. The model incorporated both visual and semantic (textual) data to create a multimodal encoding model.Very few studies have done this with fMRI responses. To the best of our knowledge, this is a niche approach. This approach could provide a more comprehensive understanding of visual perception and its underlying neural correlates, particularly when dealing with complex stimuli such as natural scenes. Multimodality is also an increasingly popular topic, partylu due to the release of GPT-4 which includes image and text modality.

To summarise; there are various approaches to brain encoding, using deep learning. Some architectures are rather advanced, utilizing recurrent and convolutional neural networks. We have opted for using a simpler model as a building block in a modular, multimodal approach, to better gauge the unique contributions of a second modality, and simple experimentation.

## Methodology

Our methodology is that of a supervised machine learning experiment. We have access to preprocessed fMRI scans showcasing the ​blood oxygen level-dependent (BOLD) response to a variety of images. Our primary objective is to construct a multimodal model that has as many parameters during inference as its unimodal counterpart, and yet better predicts the brain's response to a given image. This section outlines the steps and components involved in the execution of our experiment. It should be noted that, following the Algonauts Project, the median Pearson correlation between voxels in the ground truth and the prediction is used as the target metric (though not as a loss function).

#### Data

The data underpinning our experiment is provided by the Algonauts Project [@gifford2023] and is initially derived from the Natural Scenes Dataset (NSD) [@allen2022]. The NSD is currently the largest dataset of its kind, encompassing cortical surface vertices from the left and right hemispheres of eight participants' brains. These vertices correspond to the neurological responses triggered by 73,000 COCO images used by the NSD, each image depicting natural scenes. In addition to category information for each image, the COCO dataset provides other valuable metadata such as object location boxes and caption lists. Our experiment focused on the COCO object category information. The images in the NSD contain 80 different kinds of objects, with most images containing multiple object kinds (for example a horse and a person). As per the Algonauts guide^[https://colab.research.google.com/drive/1bLJGP3bAo_hAOwZPHpiSHKlt97X9xsUw], we represented each image using the dimensionality reduction method principal component analysis (PCA) of the all the image's activations in the 2012 image model Alexnet's second layer [@krizhevsky2012]. As in the Algonauts guide, PCA was performed reducing each image to a vector of size 100.

Over a year, each participant in the NSD study was exposed to 10,000 unique images, with each image presented three times, resulting in 30,000 image trials per participant. The corresponding fMRI data comprises 19,004 and 20,544 voxels for the left and right hemispheres, respectively. These voxel counts were selected based on preprocessed, high-quality 7T fMRI responses measuring as BOLD response amplitudes. Also included in the dataset are region of interest (ROI) masks for each subject, which aid in extracting specific fMRI data from certain locations in the brain. The fMRI data has been mapped to Harvard’s FsAverage atlas such that the voxels are comparable across individuals.  We eliminated subjects 6 and 8 from the experiment due to missing data (voxel counts differed from 19,004 and 20,544 for the left and right hemispheres respectively). We thus trained on six subjects.

#### Models

The purpose of our models is to infer the BOLD response from a given image. The architecture of our primary model involves taking a vector representation of an image $x$, and outputting a tuple consisting of the left hemisphere BOLD response $y_{lh}$, right hemisphere BOLD response $y_{rh}$, and a semantic feature vector $y_c$ for optimization against the COCO data. This model is partitioned into four submodules, each an MLP processing one of the four variables ($x$, $y_{lh}$, $y_{rh}$, and $y_c$). Our baseline will be a unimodal version of this model. We aim to test if including the semantic modality improves performance. 

The first module, referred to as the image encoding module, maps the input image vector $x$ onto a latent space, thereby generating a latent vector $z$, which is subsequently fed into the remaining three modules responsible for predicting the outputs. As suggested by the Algonauts challenge baseline, the latent vector $z$ maintains a dimensionality of 100. Given that each hemisphere's voxel count is approximately 20k, the linear mapping from the latent space to the voxel space demands around 2 million parameters. Therefore, even with such a compact latent space, the minimum required parameter count is approximately 4 million.

Our second model used $y_c$ as input, concatenating it with $x$. The purpose of this model was to gauge the potential of multimodality on the input side of the network. This model is not our main focus, but rather a test to gauge the usefulness of this particular kind of multimodality.

All hidden layers used the tanh activation function, dropout of 0.1, and weight decay of 0.0001 with the AdamW optimizer from Optax. The models were implemented in Jax with Haiku(CITE). The shared (first) module had two layers, with 100 units each, to create some flexibility as the input to all other modules (the latent vector $z$) flowed through that initial module. The rest of the modules mapped the latent vector input to whatever output dimension their modality had. The learning rate was 0.001 and the batch size was 32. Hyperparameter optimization was not done on the aforementioned hyperparameters due to computational constraints.

The primary model (with the auxiliary task of predicting $y_c$ during training), had two experiment-specific hyperparameters, $\alpha$ and $\beta$, weighing $y_c$ and whatever hemisphere was not being optimized for respectively in the loss function. The model used mean squared error for optimizing the fMRI predictions and binary soft f1 loss for $y_c$ due to a heavy imbalance between categories. Using regular binary cross entropy would yield a low loss by guessing all zeros, as most images contain only a few categories.

#### Incorporating category vector modality and semantic vector representation

To unlock the potential utility of the semantic vector, we designed our experiment with a multimodal approach. This involved integrating the category vector modality (model 2) by concatenating it with the image vector derived from AlexNet, an auxiliary task to predict the category during training (model 1), and tuning the $\alpha$ and $\beta$ parameters weighting the importance of the auxiliary tasks in the loss function. Additional motivation for the inclusion of the auxiliary modalities is the potential avoidance of overfitting; finding inappropriate shortcuts in the data becomes more difficult if the shortcuts also have to make sense of the semantic vector.

#### Model training, auxiliary tasks, and hemisphere balancing

Two key hyperparameters, $\alpha$, and $\beta$, were used to balance the different aspects of our model's performance. $\alpha$ controlled the balance between fMRI loss and category prediction loss, thereby providing weight to the auxiliary task of category prediction. This strategy was based on our hypothesis that having the model solve an auxiliary classification problem could lead to more generalized and versatile representations beneficial for the primary task of predicting fMRI responses. $\beta$ modulated the balance between the losses of the two hemispheres. By tuning this parameter, we hoped to find out if there is a balance that might contribute to a better overall model performance on the subjects.

#### Hyperparameter optimization and loss function design

The cornerstone of our experiment involves hyperparameter optimization, carried out using the Weights & Biases (wandb) sweeps with wandb's Bayesian optimization techniques. The loss function is expressed as (1 - $\alpha$)((1 - $\beta$)$Loss{y_{lh}}$ + $\beta$ $Loss{y_{rh}}$) + $\alpha$ $Loss_{y_c}$, when optimizing for $y_{lh}$ and flipping the $\beta$ when optimizing for $y_{rh}$. $\alpha$ serves as a weighting factor determining the trade-off between the fMRI prediction task and the category prediction task, while $\beta$ controls the balance between the losses of the two hemispheres.

#### Bayesian optimization and cross-validation

To search for the optimal values of $\alpha$ and $beta$, we initiated a wandb sweep with Bayesian optimization and optimized concerning validation of left hemisphere correlation in one sweep, and validation of right hemisphere correlation in the other sweep. This strategy enables a directed search in hyperparameter space, making it a more efficient and effective approach for hyperparameter tuning than random search or grid search. Additionally, we employed a K-fold cross-validation technique for model evaluation, providing a more robust estimate of the model's performance and optimal hyperparameters. K was set to 5. Every fold for every subject ran twice to get samples during the Bayesian optimization.

## Results and analysis

The following will showcase our numerical results, and do an analysis and brief discussion of these. The results seem to indicate a positive answer to our question about the potentially regularising effect of including an additional modality during inference. As __table 2__ shows, focusing on model 2, the semantic vector does contain some useful information as including it seems to increase the mean median correlation between the voxels by about more than 0.1 during training compared to the unimodal counterpart of model 2. For model 2 the benefit during training of the additional input modality still seems to be positive, though it is an extremely small increase in mean median (0.005 and 0.0042 for left and right hemispheres respectively). Bear in mind that the median is computed across the entirety of the voxel vectors, which are 19,004 and 20,544 long respectively, so a subtle increase in the median might be significant. It might also _not_ be significant necessitating further exploration. 

Exploring our primary model (with potential for the auxiliary task during training) in __table 1__ we see the mean median voxel correlations for the two hemispheres trained with and without $\alpha$ and $\beta$ set to 0 (setting these parameters to zero turns the model into our unimodal baseline). The none baseline has $\alpha = 0.05$ and $\beta = 0.25$. __Table 1__ shows us that our multimodal model outperforms the baseline on the test data, yielding a median correlation that is about 0.1 higher across both hemispheres in both train and test data, except the test data on the left hemisphere that is only 0.07 higher in median correlation. Thus, it appears that all else being equal, there is more benefit to using the second modality as an auxiliary task during training, than to receiving that information during inference. To the extent that this finding is significant, we find it surprising, as direct access to the second modality during inference seems more information-carrying, than indirect during training.
Further analysis would however be needed to explore the significance hereof. We see slight overfitting on both the baseline and the multimodal model 2, with slightly more overfitting on the baseline, again being an indication of a slightly regularising effect from the second modality.

An understanding of the hyperparameter search for  $\alpha$ and $\beta$ can be gathered from inspecting __Appendix E__ where we see that the correlation between  $\alpha$ and the median-based metrics is small, though positive. The correlation between $\beta$ and the median metrics is also slight, but negative. However, as seen in __Appendix C__ and __Appendix D__, the Bayesian hyperparameter optimization left the low $\beta$ space slightly underexplored. Setting $\alpha$ and $\beta$ values were made with human intuition having inspected the sweep logs. Again more elaborate statical analysis would be needed.




| Hemisphere | Train, Alex/COCO | Train, Alex | Test, Alex/COCO | Test, Alex |
|------------|-----------------:|------------:|----------------:|-----------:|
| Left       | 0.2558           | _0.2676_    | _0.1869_        | 0.1812     |
| Right      | 0.255            | _0.265_     | _0.1881_        | 0.1782     |

Table: Mean Median Voxel Correlation (Model 1).



In __table 2__ we see mean median voxel correlations across all subjects and folds of model 2 with and without the COCO vector concatenated to the Alex vector. We see a similarly subtle advantage to including the second modality here. The reader should again note that the metrics here displayed are mean _median_ correlations.

| Hemisphere | Train, Alex/COCO | Train, Alex | Test, Alex/COCO | Test, Alex |
|------------|-----------------:|------------:|----------------:|-----------:|
| Left       | _0.2176_         | 0.2059      | _0.1932_        | 0.1927     |
| Right      | _0.2155_         | 0.2046      | _0.195_         | 0.1908     |

Table: Mean Median Voxel Correlation (Model 2).

Also relevant is that the correlation between prediction and truth is not uniformly spread throughout the visual cortex, but rather concentrated on earlier regions of interest, as showcased in __Appendix A__, __Appendix B__ and interactively at neuroscope.streamlit.app







## Future Work

As seen in the previous section, it appears that the semantic vector modality is not particularly useful for the model. A logical next step would be to experiment with extracting the image representations from different, or multiple AlexNet layers, or using an entirely different model for the image representation extraction. We might also explore using more rich COCO modalities such as image captions and object bounding boxes. Lastly, from a neuroscientific perspective, the ROIs of the brain are considered to be different modalities: they function by vastly different rules. Processing the ROIs separately might allow for models tailoring to specific ROI idiosyncrasies.

## Conclusion

It appears that including the semantic vector, and creating a multimodal model increases performance slightly at inference time probably due to a subtle regularising effect, though the significance of the increase merits further study.

## References

<div id="refs"></div>

\pagebreak

## Appendix

### Appendix A

![Subject 1 Left hemisphere correlations](../plots/left_hem.png)

\pagebreak

### Appendix B

![Subject 3 Right hemisphere correlations](../plots/right_hem.png)

\pagebreak

### Appendix C

|Name                |alpha                 |beta                | __val_lh_corr__       |val_rh_corr        |
|-------------------------------:|---------------------:|-------------------:|------------------:|------------------:|
|giddy-sweep-48      |0.031                 |0.347               |0.245              |0.251              |
|avid-sweep-12       |0.051                 |0.223               |0.245              |0.23               |
|woven-sweep-13      |0.046                 |0.082               |0.242              |0.215              |
|solar-sweep-11      |0.076                 |0.167               |0.24               |0.226              |
|leafy-sweep-18      |0.025                 |0.36                |0.235              |0.253              |
|dainty-sweep-44     |0.044                 |0.018               |0.229              |0.224              |
|effortless-sweep-47 |0.033                 |0.042               |0.228              |0.209              |
|fresh-sweep-17      |0.034                 |0.072               |0.225              |0.177              |
|silvery-sweep-41    |0.096                 |0.218               |0.225              |0.225              |
|fresh-sweep-50      |0.014                 |0.211               |0.224              |0.209              |
|generous-sweep-5    |0.072                 |0.294               |0.22               |0.22               |
|autumn-sweep-16     |0.026                 |0.469               |0.218              |0.218              |
|crimson-sweep-15    |0.08                  |0.41                |0.215              |0.202              |
|skilled-sweep-45    |0.049                 |0.186               |0.215              |0.201              |
|sandy-sweep-4       |0.057                 |0.473               |0.214              |0.177              |
|fearless-sweep-7    |0.031                 |0.398               |0.213              |0.212              |
|fanciful-sweep-20   |0.042                 |0.429               |0.212              |0.22               |
|sunny-sweep-14      |0.029                 |0.353               |0.211              |0.2                |
|genial-sweep-8      |0.07                  |0.488               |0.211              |0.195              |
|colorful-sweep-43   |0.017                 |0.158               |0.206              |0.186              |
|hardy-sweep-57      |0.007                 |0.126               |0.202              |0.198              |
|glamorous-sweep-42  |0.025                 |0.201               |0.201              |0.185              |
|dark-sweep-2        |0.012                 |0.391               |0.196              |0.18               |
|likely-sweep-46     |0.056                 |0.067               |0.194              |0.184              |
|pleasant-sweep-51   |0.051                 |0.004               |0.193              |0.171              |
|pious-sweep-31      |0.065                 |0.347               |0.19               |0.223              |
|cool-sweep-19       |0.027                 |0.152               |0.19               |0.223              |
|hearty-sweep-1      |0.073                 |0.045               |0.189              |0.166              |
|resilient-sweep-60  |0.049                 |0.061               |0.184              |0.175              |
|super-sweep-10      |0.015                 |0.437               |0.183              |0.176              |
|bright-sweep-34     |0.065                 |0.066               |0.181              |0.203              |
|light-sweep-23      |0.092                 |0.191               |0.177              |0.165              |
|firm-sweep-28       |0.044                 |0.117               |0.177              |0.172              |
|volcanic-sweep-49   |0.0                   |0.412               |0.174              |0.179              |
|breezy-sweep-6      |0.054                 |0.451               |0.173              |0.165              |
|tough-sweep-22      |0.027                 |0.058               |0.173              |0.149              |
|icy-sweep-9         |0.085                 |0.373               |0.173              |0.157              |
|distinctive-sweep-30|0.083                 |0.084               |0.168              |0.166              |
|decent-sweep-58     |0.09                  |0.499               |0.159              |0.148              |
|true-sweep-3        |0.042                 |0.354               |0.157              |0.144              |
|firm-sweep-59       |0.025                 |0.31                |0.155              |0.155              |
|sweepy-sweep-56     |0.075                 |0.27                |0.153              |0.112              |
|clean-sweep-27      |0.021                 |0.179               |0.151              |0.126              |
|fallen-sweep-35     |0.042                 |0.078               |0.147              |0.163              |
|celestial-sweep-36  |0.088                 |0.244               |0.146              |0.146              |
|golden-sweep-55     |0.015                 |0.396               |0.146              |0.131              |
|splendid-sweep-53   |0.048                 |0.07                |0.143              |0.149              |
|efficient-sweep-26  |0.02                  |0.387               |0.143              |0.135              |
|fiery-sweep-54      |0.04                  |0.442               |0.138              |0.139              |
|firm-sweep-37       |0.056                 |0.275               |0.136              |0.131              |
|rosy-sweep-38       |0.049                 |0.395               |0.134              |0.148              |
|wandering-sweep-32  |0.025                 |0.249               |0.132              |0.149              |
|twilight-sweep-40   |0.035                 |0.089               |0.127              |0.132              |
|glorious-sweep-33   |0.021                 |0.422               |0.126              |0.135              |
|dainty-sweep-52     |0.008                 |0.283               |0.124              |0.108              |
|peach-sweep-25      |0.029                 |0.401               |0.115              |0.116              |
|devoted-sweep-21    |0.093                 |0.412               |0.11               |0.1                |
|true-sweep-24       |0.04                  |0.259               |0.107              |0.111              |
|fancy-sweep-39      |0.004                 |0.083               |0.106              |0.105              |
|radiant-sweep-29    |0.061                 |0.427               |0.104              |0.106              |

Table: lh sweep log (sorted by lh corr).



### Appendix D

|Name                |alpha                 |beta                |val_lh_corr        | __val_rh_corr__        |
|---------------------------:|---------------------:|-------------------:|------------------:|------------------:|
|wobbly-sweep-19     |0.059                 |0.234               |0.273              |0.28               |
|generous-sweep-49   |0.089                 |0.045               |0.234              |0.237              |
|deep-sweep-9        |0.067                 |0.083               |0.216              |0.228              |
|treasured-sweep-16  |0.088                 |0.379               |0.221              |0.228              |
|quiet-sweep-11      |0.014                 |0.18                |0.225              |0.224              |
|radiant-sweep-50    |0.003                 |0.429               |0.236              |0.223              |
|splendid-sweep-7    |0.07                  |0.464               |0.236              |0.223              |
|fallen-sweep-48     |0.057                 |0.486               |0.243              |0.222              |
|genial-sweep-42     |0.038                 |0.037               |0.226              |0.221              |
|misty-sweep-14      |0.053                 |0.224               |0.227              |0.22               |
|rosy-sweep-17       |0.084                 |0.155               |0.215              |0.219              |
|sweepy-sweep-34     |0.078                 |0.333               |0.193              |0.218              |
|ancient-sweep-18    |0.007                 |0.011               |0.22               |0.216              |
|major-sweep-45      |0.068                 |0.452               |0.223              |0.215              |
|peach-sweep-12      |0.072                 |0.292               |0.222              |0.209              |
|winter-sweep-47     |0.018                 |0.198               |0.211              |0.207              |
|warm-sweep-25       |0.052                 |0.326               |0.203              |0.206              |
|chocolate-sweep-46  |0.01                  |0.103               |0.202              |0.205              |
|crisp-sweep-20      |0.095                 |0.02                |0.19               |0.203              |
|effortless-sweep-44 |0.03                  |0.33                |0.236              |0.201              |
|warm-sweep-41       |0.084                 |0.058               |0.212              |0.198              |
|deep-sweep-10       |0.02                  |0.331               |0.21               |0.196              |
|kind-sweep-4        |0.028                 |0.317               |0.207              |0.19               |
|hearty-sweep-60     |0.057                 |0.383               |0.192              |0.186              |
|solar-sweep-1       |0.034                 |0.302               |0.192              |0.186              |
|absurd-sweep-43     |0.003                 |0.129               |0.206              |0.186              |
|dry-sweep-58        |0.064                 |0.475               |0.169              |0.186              |
|drawn-sweep-6       |0.028                 |0.346               |0.184              |0.185              |
|decent-sweep-35     |0.097                 |0.225               |0.167              |0.184              |
|worldly-sweep-5     |0.042                 |0.396               |0.172              |0.183              |
|kind-sweep-54       |0.08                  |0.405               |0.175              |0.182              |
|olive-sweep-31      |0.022                 |0.222               |0.158              |0.181              |
|sparkling-sweep-3   |0.033                 |0.306               |0.194              |0.18               |
|cosmic-sweep-37     |0.079                 |0.143               |0.155              |0.177              |
|light-sweep-13      |0.058                 |0.253               |0.204              |0.177              |
|devoted-sweep-57    |0.08                  |0.167               |0.177              |0.175              |
|proud-sweep-8       |0.025                 |0.436               |0.197              |0.174              |
|rural-sweep-53      |0.016                 |0.423               |0.192              |0.173              |
|earnest-sweep-2     |0.071                 |0.16                |0.178              |0.172              |
|visionary-sweep-36  |0.075                 |0.223               |0.16               |0.171              |
|floral-sweep-30     |0.075                 |0.433               |0.169              |0.165              |
|super-sweep-38      |0.048                 |0.035               |0.155              |0.165              |
|astral-sweep-28     |0.044                 |0.285               |0.168              |0.163              |
|neat-sweep-33       |0.026                 |0.006               |0.16               |0.162              |
|jolly-sweep-40      |0.067                 |0.213               |0.15               |0.162              |
|firm-sweep-59       |0.002                 |0.3                 |0.15               |0.16               |
|cerulean-sweep-15   |0.055                 |0.489               |0.162              |0.159              |
|lilac-sweep-27      |0.007                 |0.472               |0.177              |0.157              |
|glad-sweep-56       |0.085                 |0.451               |0.16               |0.156              |
|dandy-sweep-32      |0.089                 |0.048               |0.138              |0.151              |
|glad-sweep-52       |0.06                  |0.463               |0.156              |0.149              |
|valiant-sweep-22    |0.04                  |0.096               |0.13               |0.143              |
|cerulean-sweep-51   |0.065                 |0.208               |0.142              |0.143              |
|still-sweep-39      |0.046                 |0.062               |0.125              |0.142              |
|smart-sweep-23      |0.023                 |0.036               |0.132              |0.139              |
|youthful-sweep-55   |0.009                 |0.249               |0.134              |0.138              |
|fresh-sweep-29      |0.02                  |0.416               |0.108              |0.129              |
|driven-sweep-26     |0.079                 |0.167               |0.133              |0.125              |
|autumn-sweep-24     |0.079                 |0.484               |0.125              |0.109              |
|skilled-sweep-21    |0.041                 |0.338               |0.069              |0.086              |

Table: rh sweep log (sorted by rh corr).

### Appendix E


| Hemisphere | $\alpha$ correlation | $\beta$ correlation |
|------------|---------------------:|--------------------:|
| Left       | 0.063                | - 0.147             |
| Right      | 0.076                | - 0.087             |

Table: Bayesian Hyperparameter Sweep (Model 1).

