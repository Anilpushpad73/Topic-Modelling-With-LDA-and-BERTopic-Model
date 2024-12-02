# Topic Modeling Analysis with LDA and BERTopic

## Finding the Number of Topics

Choosing the optimal number of topics (`k`) ensures meaningful and interpretable results:
- **Low `k`**: Topics are too broad, reducing interpretability.
- **High `k`**: Keywords repeat across topics, indicating overfitting.

---

## LDA Model Tuning

### Parameters
- ➢ **NUMBER_TOPICS**: 20  
- ➢ **ALPHA**: `auto` (automatically optimizes topic quality and interpretability)  
- ➢ **ETA**: `auto`  
- ➢ **CHUNK_SIZE**: 150  

---

### Passes Variation
- ➢ **PASSES**: Varying from 10 to 40  
- ➢ **Optimal Number of Passes**: **20**  
- ➢ **Max Coherence Score**: 0.540 at 30 passes (choosing 20 for balance).  

![Coherence vs. Passes](https://github.com/user-attachments/assets/a2a5aa7b-a050-498b-8477-87242be86c79)

---

### Topic Number Variation
- ➢ **Optimal Topics**: **22**  
- ➢ **Max Coherence Score**: 0.605 at 22 topics.  

![Coherence vs. Topics](https://github.com/user-attachments/assets/f06e71fa-a1f2-421c-b884-c884b06a89ab)

---

### Chunk Size Variation
- ➢ **Optimal Chunk Size**: **100**  
- ➢ **Max Coherence Score**: 0.640  

![Coherence vs. Chunk Size](https://github.com/user-attachments/assets/eead28e3-e480-41e6-96ae-61d328b99a82)

---

### Final Model Summary
- ➢ **Perplexity**: -17.92  
- ➢ **Coherence Score**: 0.640

## Interactive LDA Visualization

To view the interactive LDA visualization, [click here](C:\Users\Anil\Downloads).

  

---

## BERTopic Model Tuning

### Parameters:
- ➢ **Embeddings Model**: `all-MiniLM-L6-v2`  
- ➢ **N_NEIGHBORS**: 15  
- ➢ **MIN_CLUSTER_SIZE**: 60  
- ➢ **MIN_SAMPLES**: 1  

---

### Neighbor Variation
- ➢ **Optimal Neighbors**: **15**  
- ➢ **Coherence Score Range**: 0.486 – 0.553  
- ➢ **Max Coherence**: 0.553  

![Coherence vs. Neighbors](https://github.com/user-attachments/assets/3e6cb212-70b3-4230-ae9a-5d5b677e3047)

---

### Cluster Size Variation
- ➢ **Optimal Cluster Size**: **60**  
- ➢ **Coherence Score Range**: 0.553 – 0.647  
- ➢ **Max Coherence**: 0.647  

![Coherence vs. Cluster Size](https://github.com/user-attachments/assets/a4ee0e97-174b-43f6-9022-1a9e65577c59)

---

### Min Samples Variation
- ➢ **Optimal Min Samples**: **1**  
- ➢ **Coherence Score Range**: 0.593 – 0.703  
- ➢ **Max Coherence**: 0.703  

![Coherence vs. Min Samples](https://github.com/user-attachments/assets/8803e349-050b-4c57-bd58-3f8c81ba2c7a)

---

## Embedding Model Comparison

### Model A: `all-MiniLM-L6-v2`
- ➢ Lightweight (6 layers, 384 dimensions)  
- ➢ Balances speed and semantic quality  
- ➢ Coherence Range: 0.463 – 0.720  

### Model B: `all-mpnet-base-v2`
- ➢ Advanced (12 layers, 768 dimensions)  
- ➢ Rich semantic embeddings  
- ➢ Coherence Range: 0.496 – 0.754  

---

### Coherence Comparison:
- ➢ **Model A (18 Topics)**: 0.720  
- ➢ **Model B (22 Topics)**: 0.754  

![Model Comparison](https://github.com/user-attachments/assets/86c08421-8b16-4529-a796-b6e724e56dd7)

---

### Final Visualization
For interactive visualization, use `pyLDAvis`:
```python
import pyLDAvis
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 10))
pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(lda_model, corpus=corpus_matrix, dictionary=id2word)
vis
