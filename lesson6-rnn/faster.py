

def generate_bigrams(x):
    n_grams = set(zip(*[x[i:] for i in range(2)]))
    for n_gram in n_grams:
        x.append(' '.join(n_gram))
    return x


# As an example:

# In[2]:


generate_bigrams(['This', 'film', 'is', 'terrible'])


# TorchText `Field`s have a `preprocessing` argument. A function passed here will be applied to a sentence after it has been tokenized (transformed from a string into a list of tokens), but before it has been indexed (transformed from a list of tokens to a list of indexes). This is where we'll pass our `generate_bigrams` function.

# In[3]:


import torch
from torchtext import data
from torchtext import datasets

SEED = 1234

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

TEXT = data.Field(tokenize='spacy', preprocessing=generate_bigrams)
LABEL = data.LabelField(dtype=torch.float)


# As before, we load the IMDb dataset and create the splits.

# In[4]:


import random

train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)

train_data, valid_data = train_data.split(random_state=random.seed(SEED))


# Build the vocab and load the pre-trained word embeddings.

# In[5]:


TEXT.build_vocab(train_data, max_size=25000, vectors="glove.6B.100d")
LABEL.build_vocab(train_data)


# And create the iterators.

# In[6]:


BATCH_SIZE = 64

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, valid_data, test_data), 
    batch_size=BATCH_SIZE, 
    device=device)


# ## Build the Model
# 
# This model has far fewer parameters than the previous model as it only has 2 layers that have any parameters, the embedding layer and the linear layer. There is no RNN component in sight!
# 
# Instead, it first calculates the word embedding for each word using the `Embedding` layer (purple), then calculates the average of all of the word embeddings and feeds this through the `Linear` layer (silver), and that's it!
# 
# ![](https://i.imgur.com/e0sWZoZ.png)
# 
# We implement the averaging with the `avg_pool2d` (average pool 2-dimensions) function. Initially, you may think using a 2-dimensional pooling seems strange, surely our sentences are 1-dimensional, not 2-dimensional? However, you can think of the word embeddings as a 2-dimensional grid, where the words are along one axis and the dimensions of the word embeddings are along the other. The image below is an example sentence after being converted into 5-dimensional word embeddings, with the words along the vertical axis and the embeddings along the horizontal axis. Each element in this [4x5] tensor is represented by a green block.
# 
# ![](https://i.imgur.com/SSH25NT.png)
# 
# The `avg_pool2d` uses a filter of size `embedded.shape[1]` (i.e. the length of the sentence) by 1. This is shown in pink in the image below.
# 
# ![](https://i.imgur.com/U7eRnIe.png)
# 
# The filter then slides to the right, calculating the average of the next column of embedding values for each word in the sentence. After the filter has covered all embedding dimensions, we get a [1x5] tensor. This tensor is then passed through the linear layer to produce our prediction.

# In[7]:


import torch.nn as nn
import torch.nn.functional as F

class FastText(nn.Module):
    def __init__(self, vocab_size, embedding_dim, output_dim):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc = nn.Linear(embedding_dim, output_dim)
        
    def forward(self, x):
        
        #x = [sent len, batch size]
        
        embedded = self.embedding(x)
                
        #embedded = [sent len, batch size, emb dim]
        
        embedded = embedded.permute(1, 0, 2)
        
        #embedded = [batch size, sent len, emb dim]
        
        pooled = F.avg_pool2d(embedded, (embedded.shape[1], 1)).squeeze(1) 
        
        #pooled = [batch size, embedding_dim]
                
        return self.fc(pooled)


# As previously, we'll create an instance of our `FastText` class.

# In[8]:


INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 100
OUTPUT_DIM = 1

model = FastText(INPUT_DIM, EMBEDDING_DIM, OUTPUT_DIM)


# And copy the pre-trained vectors to our embedding layer.

# In[9]:


pretrained_embeddings = TEXT.vocab.vectors

model.embedding.weight.data.copy_(pretrained_embeddings)


# ## Train the Model

# Training the model is the exact same as last time.
# 
# We initialize our optimizer...

# In[10]:


import torch.optim as optim

optimizer = optim.Adam(model.parameters())


# We define the criterion and place the model and criterion on the GPU (if available)...

# In[11]:


criterion = nn.BCEWithLogitsLoss()

model = model.to(device)
criterion = criterion.to(device)


# We implement the function to calculate accuracy...

# In[12]:


def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """

    #round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float() #convert into float for division 
    acc = correct.sum()/len(correct)
    return acc


# We define a function for training our model...
# 
# **Note**: we are no longer using dropout so we do not need to use `model.train()`, but as mentioned in the 1st notebook, it is good practice to use it.

# In[13]:


def train(model, iterator, optimizer, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.train()
    
    for batch in iterator:
        
        optimizer.zero_grad()
        
        predictions = model(batch.text).squeeze(1)
        
        loss = criterion(predictions, batch.label)
        
        acc = binary_accuracy(predictions, batch.label)
        
        loss.backward()
        
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


# We define a function for testing our model...
# 
# **Note**: again, we leave `model.eval()` even though we do not use dropout.

# In[14]:


def evaluate(model, iterator, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()
    
    with torch.no_grad():
    
        for batch in iterator:

            predictions = model(batch.text).squeeze(1)
            
            loss = criterion(predictions, batch.label)
            
            acc = binary_accuracy(predictions, batch.label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


# Finally, we train our model...

# In[15]:


N_EPOCHS = 5

for epoch in range(N_EPOCHS):

    train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
    valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)
    
    print(f'| Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}% | Val. Loss: {valid_loss:.3f} | Val. Acc: {valid_acc*100:.2f}% |')


# ...and get the test accuracy!
# 
# The results are comparable to the results in the last notebook, but training takes considerably less time.

# In[16]:


test_loss, test_acc = evaluate(model, test_iterator, criterion)

print(f'| Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}% |')


# ## User Input
# 
# And as before, we can test on any input the user provides.

# In[17]:


import spacy
nlp = spacy.load('en')

def predict_sentiment(sentence):
    tokenized = [tok.text for tok in nlp.tokenizer(sentence)]
    indexed = [TEXT.vocab.stoi[t] for t in tokenized]
    tensor = torch.LongTensor(indexed).to(device)
    tensor = tensor.unsqueeze(1)
    prediction = torch.sigmoid(model(tensor))
    return prediction.item()


# An example negative review...

# In[18]:


predict_sentiment("This film is terrible")


# An example positive review...

# In[19]:


predict_sentiment("This film is great")


# ## Next Steps
# 
# In the final notebook we'll use convolutional neural networks (CNNs) to perform sentiment analysis, and get our best accuracy yet!
