Finding the number of topics
Choosing a ‘k’ that marks the end of a rapid growth of topic coherence usually offers meaningful and interpretable topics. Picking an even higher value can sometimes provide more granular sub-topics.

If you see the same keywords being repeated in multiple topics, it’s probably a sign that the ‘k’ is too large.

The compute_coherence_values() (see below) trains multiple LDA models and provides the models and their corresponding coherence scores.

Varying the number of passes while keeping the other hyperparameters constant and
these parameters are:

NUMBER_TOPICS = 20
ALPHA = 'auto' ( automatically learn which can improve the quality and interpretability)
ETA = 'auto'
CHUNK_SIZE = 150 ( number of documents processed in each training chunk)
As the number of passes increases, the model refines the topics by adjusting the word distributions within topics. As the Coherence scores improve and typically reach a peak, indicating the topics have become more semantically meaningful. But Identify the optimal number of passes that balance between computational efficiency and topic quality i.e Maximize coherence without unnecessary computation.

![image](https://github.com/user-attachments/assets/a2a5aa7b-a050-498b-8477-87242be86c79)

shows that among the evaluated number of passes, the coherence score slightly fluctuates after 20 passes. It achieves the highest coherence score of 0.5400143812889777 at 30 passes. To avoid the limitations of overfitting risks and excessive computation, the 30 passes strike an effective balance between representational capacity and computational efficiency. Therefore, it is recommended as the suitable number of passes for this application. However, we choose the optimal number of 20 passes for further solutions to ensure maximum coherence score and computational efficiency.
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


Varying the number of topic while keeping the other hyperparameters constant and
these parameters are:

PASSES = 20
CHUNK SIZE = 150
ALPHA = ’auto’
ETA = ’auto’
With a very low number of topics(<5) , the model struggles to capture the diversity of topics in the documents and coherence scores are generally low because distinct topics are forced to be traced as the same topic which decreases interpretability. Beyond a certain point (>22), increasing the number of topics results in splitting coherent topics into smaller, less meaningful subtopics. Coherence scores decrease, because the topics become too specific and less interpretable. The model starts to capture noise and overfits the data, reducing the overall quality of the topics.

![image](https://github.com/user-attachments/assets/f06e71fa-a1f2-421c-b884-c884b06a89ab)

shows that among the evaluated number of topics, the coherence score fluctuates between 0.4207565431342476 and 0.6049269193617294, But after 19, number of topics, the coherence score have increased the 22nd number of topics which is having 0.6049269193617294 maximum coherence score at 22nd number of topics. So we choose the optimal number of 22 topics for further solutions to ensure maximum coherence score and computational efficiency.


------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Varying the chunk size while keeping the other hyperparameters constant and
These parameters are:

NUMBER_TOPICS = 22
PASSES = 20
ALPHA = ’auto’
ETA = ’auto’
At low chunk size, the coherence scores low due to frequent updates and hence, bring in instability. Small chunks allow frequent updates, which can capture diverse patterns quickly but also introduces noise then reducing coherence score, but increasing clock time. Large chunks reduce the update frequency, which can slow convergence and lead to capturing less meaningful patterns and decreasing coherence score.

![image](https://github.com/user-attachments/assets/eead28e3-e480-41e6-96ae-61d328b99a82)

we evaluate the coherence score against different chunk sizes. The coherence score fluctuates within the range of 0.49589908184922275 to 0.64030572542999215 for chunk sizes between 50 and 300. The highest coherence score of 0.64030572542999215 is observed at a chunk size of 100.

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

we have got the best 22 topics to be selected in our data:

Perplexity:  -17.915056449779602

Coherence Score:  0.6403057254299922


 fig, ax = plt.subplots(figsize=(10, 10))
pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(lda_model, corpus = corpus_matrix, dictionary = id2word)
vis

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Bertopic model:

![image](https://github.com/user-attachments/assets/63359417-e6b6-45eb-a4e2-6efcba3afeda)

Result:

Varying the number of neighbors while keeping the other hyperparameters constant
and these parameters are:

➢ Using the all-MiniLM-L6-v2 embeddings model
NUMBER_TOPICS = 20
MIN_CLUSTER_SIZE = 10
MIN_SAMPLES = 10
ALPHA = ’none’
Number of neighbors determines the number of neighboring points used for local approximation in the manifold learning process.The number of neighbors (n_neighbors) affects the topic coherence score. With a small n_neighbors, the model captures too much noise and focuses on local details, missing the bigger picture and resulting in unclear topics so that shows the low coherence score. With a large n_neighbors, the model captures broad patterns but misses important details. This makes topics too general and less useful and thus decreases the coherence score again. So Balancing n_neighbors is crucial for capturing both detailed and overall patterns, leading to better coherence scores and more understandable topics

![image](https://github.com/user-attachments/assets/3e6cb212-70b3-4230-ae9a-5d5b677e3047)

we evaluate the coherence score against different numbers of neighbors . The coherence score fluctuates within the range of 0.48582252434505047 to 0.55294548912511 for neighbors between 5 and 45. The highest coherence score of 0.55294548912511 is observed at the neighbor 15. So we choose the optimal number of 15 neighbors for further solutions to ensure maximum coherence score and computational efficiency.

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Varying the minimum cluster size while keeping the other hyperparameters constant
and these parameters are:

➢ Using the all-MiniLM-L6-v2 embeddings model
NUMBER_TOPICS = 20
N_NEIGHBORS = 15
MIN_SAMPLES = 10
ALPHA = ’none’
The minimum cluster size determines the smallest group of points that can be considered a cluster. Initially, having a small minimum cluster size may result in low and fluctuating coherence scores, as small clusters can capture minor variations and noise, thereby reducing topic coherence. Conversely, with a large minimum cluster size, coherence scores may plateau or decrease because larger clusters tend to generalize too broadly, reducing topic coherence. Therefore, we select an optimal minimum cluster size to ensure that HDBSCAN identifies meaningful topics without being overly specific or broad, leading to higher coherence scores.

![image](https://github.com/user-attachments/assets/a4ee0e97-174b-43f6-9022-1a9e65577c59)

we evaluate the coherence score against different minimum cluster size. The coherence score fluctuates within the range of 0.55294548912511 to 0.6472625012887551 for minimum cluster size between 10 and 100. The highest coherence score of 0.6472625012887551 is observed at a minimum cluster size of 60. So we choose the optimal number of 60 minimum cluster size for further solutions to ensure maximum coherence score and computational efficiency.

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Varying the minimum sample while keeping the other hyperparameters constant and
these parameters are:

➢ Using the all-MiniLM-L6-v2 embeddings model
NUMBER_TOPICS = 20
N_NEIGHBORS = 15
MIN_CLUSTER_SIZE = 60
ALPHA = ’none’
The minimum sample (min_samples) determines the minimum sample size, which is crucial as it determines the minimum number of points required to form a dense region, which affects the definition of noise. The larger the min_samples value, the more conservative the clustering becomes, as more points are declared as noise and the clusters are restricted to denser regions. This affects the coherence values, as small min_samples may contain too much noise, which reduces coherence, while large min_samples may overgeneralize the clusters, which also reduces coherence values. We choose Balancing min_samples to get better coherence score and more interpretable topics.

![image](https://github.com/user-attachments/assets/8803e349-050b-4c57-bd58-3f8c81ba2c7a)

we evaluate the coherence score against different minimum samples. The coherence score fluctuates within the range of 0.5934293650110775 to 0.7025142070734569 for minimum sample size between 1 and 20. The highest coherence score of 0.7025142070734569 is observed at a minimum sample size of 1. So we choose the optimal number of 1 minimum sample size for further solutions to ensure maximum coherence score and more interpretable topics.

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Varying the Alpha while keeping the other hyperparameters constant and
these parameters are:

MIN_SAMPLES = 1
N_NEIGHBORS = 15
MIN_CLUSTER_SIZE = 60
NUM OF TOPIC=20

![image](https://github.com/user-attachments/assets/ed295f23-033b-4de7-be36-f5bc6a377a58)

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

There are using two different embedding model to compare the results:

Model A: The all-MiniLM-L6-v2 embedding model is designed to balance speed and performance. It has 6 layers and uses 384-dimensional vectors to represent text. This model is efficient, providing decent semantic quality while being relatively lightweight, making it faster for processing large datasets.
Model B: The all-mpnet-base-v2 embedding model is more advanced, offering richer and more nuanced semantic embeddings. It uses 12 layers and 768-dimensional vectors, providing high-quality embeddings that capture complex relationships in the data
When using the all-MiniLM-L6-v2 embedding model, coherence scores start low with few broad topics. As the number of topics increases to a moderate range, coherence scores improve due to the model's balanced semantic representation, resulting in clearer and more meaningful topics. However, with too many topics, coherence scores plateau or decline as topics become overly specific and fragmented coherent topics. In contrast, the all-mpnet-base-v2 model, with its richer and nuanced semantic embeddings, shows a similar trend but maintains higher coherence scores longer, providing clearer and more interpretable topics before also facing a plateau or decline with excessive topics

Varying the number of topics while keeping the other hyperparameters constant and
these parameters are:

➢ Using the all-MiniLM-L6-v2 embeddings model
MIN_SAMPLES = 1
N_NEIGHBORS = 15
MIN_CLUSTER_SIZE = 60
ALPHA = 1.0 (for all value of alpha give the same coherence score)
Initially, with few topics, coherence scores are low because each topic covers a wide range of concepts, making them less clear. As the number of topics increases to an optimal range, the coherence scores improve because the topics become more focused and meaningful. However, if the number of topics increases too much, the coherence scores stagnate or decrease because the topics become too specific and coherent topics are split into smaller, which become less meaningful subtopics, creating noise.

![image](https://github.com/user-attachments/assets/28525553-bc3d-45bf-8ece-3ebb68d08ab9)

Varying the number of topics while keeping the other hyperparameters constant and

these parameters are:

➢ Using the all-mpnet-base-v2 embeddings model
MIN_SAMPLES = 1
N_NEIGHBORS = 15
MIN_CLUSTER_SIZE = 60
ALPHA = 1.0 (for all value of alpha give the same coherence score)

![image](https://github.com/user-attachments/assets/b4ea377e-c825-42ac-b34b-ef82822787ee)

Compare between  model_A and model_B

![image](https://github.com/user-attachments/assets/86c08421-8b16-4529-a796-b6e724e56dd7)

we evaluate the coherence score as a function of the number of topics with different embedding models. For model A, the coherence score varies between 0.4625748773569187 and 0.7202945489125116 and for model B, the coherence score varies between 0.49594481921233756 and 0.7537944683203674 for the number of topics between 1 and 30. The highest coherence score of 0.7202945489125116 is observed for the number of topics of model A with 18, while for model B, the highest coherence score of 0.7537944683203674 is observed for the number of topics of 22.

