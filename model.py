#!/usr/bin/env python
# coding: utf-8

# In[15]:


import torch
import torch.nn as nn
import torch.nn.functional as F

with open('poems_preprocessed.txt', 'r', encoding='utf-8') as f:
    data = f.read()


# In[16]:


chars = sorted(list(set(data)))

ctoi = { ch:i for i,ch in enumerate(chars) }
itoc = { i:ch for i,ch in enumerate(chars) }


# In[17]:


def get_stats(ids):
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


# In[18]:


most_common = sorted([(v,k) for k, v in get_stats(data).items()], reverse=True)[:15]

print(f'Most Common pairs: {most_common}')


# In[19]:


vocab_size = len(ctoi.items()) + 1

for index, pair in most_common:
    pair_cat = pair[0] + pair[1]
    ctoi[pair_cat] = vocab_size
    itoc[vocab_size] = pair_cat
    vocab_size += 1


# In[20]:


print('Charactor to Index Dict: ')
print(sorted(list(ctoi.items()), key=lambda x: x[1], reverse=True))


# In[21]:


def encode(s):
    encoded = []
    i = 0
    while i < len(s):
        if i + 1 < len(s) and s[i:i+2] in ctoi:
            encoded.append(ctoi[s[i:i+2]])
            i += 2
        elif s[i] in ctoi:
            encoded.append(ctoi[s[i]])
            i += 1
        else:
            raise ValueError(f"Character {s[i]} not in ctoi dictionary.")
    return encoded

decode = lambda l    : ''.join([itoc[i] for i in l])


# In[22]:


text = 'سلام آرش خوبی؟'
encoded = encode(text)
decoded = decode(encoded)

print("Original text:", text)
print("Encoded text:", encoded)
print("Decoded text:", decoded)


# In[23]:


data = torch.tensor(encode(data), dtype=torch.long)


# In[24]:


n = int(0.9*len(data)) 
train = data[:n]
valid = data[n:]


# In[25]:


# Hyper parameters

batch_size = 128
block_size = 256
n_emb = 384

n_head = 3
n_layer = 2

eval_iters = 4
dropout = 0.2


# In[26]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[27]:


def get_batch(block_size, batch_size, is_training = True):
    data = train if is_training else valid
    
    n = len(data)
    
    indices = torch.randint(0, n - block_size, (batch_size,))
    
    x = [data[i : i+block_size] for i in indices]
    y = [data[i+1 : i+block_size+1] for i in indices]
    
    x, y = torch.stack(x), torch.stack(y)
    
    x, y = x.to(device), y.to(device)
    
    return x, y


# In[28]:


class Head(nn.Module):
    def __init__(self, n_emb, block_size, head_size):
        super().__init__()
        self.key   = nn.Linear(n_emb, head_size, bias=False)
        self.query = nn.Linear(n_emb, head_size, bias=False)
        self.value = nn.Linear(n_emb, head_size, bias=False)
        
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))   #  not to be considered a model parameter
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        k = self.key(x)
        q = self.query(x)
        
        weight = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        weight = weight.masked_fill(self.tril[:x.shape[-1]][:x.shape[-1]] == 0, float('-inf'))
        weight = F.softmax(weight, -1)
        weight = self.dropout(weight)

        v = self.value(x)
        out = weight @ v
        
        return out


# In[29]:


class MultiHeadAttention(nn.Module):
    def __init__(self, n_emb, block_size, head_size, n_heads):
        super().__init__()
        self.heads = nn.ModuleList([Head(n_emb, block_size, head_size) for _ in range(n_heads)])
        self.linear = nn.Linear(n_emb, n_emb)
        self.dropout = nn.Dropout(dropout)

        
    def forward(self, x):
        cat = torch.cat([h(x) for h in self.heads], -1)
        out = self.dropout(self.linear(cat))
        return out


# In[30]:


class FeedForward(nn.Module):
    def __init__(self, n_emb):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_emb, n_emb * 4),
            nn.ReLU(),
            nn.Linear(n_emb * 4, n_emb),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        return self.net(x)


# In[31]:


class Block(nn.Module):
    def __init__(self, n_emb, block_size, n_head):
        super().__init__()
        self.layer1 = nn.LayerNorm(n_emb)
        self.layer2 = nn.LayerNorm(n_emb)
        
        head_size = n_emb // n_head 
        self.sa_heads = MultiHeadAttention(n_emb, block_size, head_size, n_head)
        self.ffwd = FeedForward(n_emb)
        
        self.bn1 = nn.BatchNorm1d(block_size)
        self.bn2 = nn.BatchNorm1d(block_size)
        
    def forward(self, x):
        x = x + self.sa_heads(self.layer1(x))
        x = self.bn1(x)
        x = x + self.ffwd(self.layer2(x))
        x = self.bn2(x)
        
        return x


# In[32]:


class MultiLayerBigram(nn.Module):
    def __init__(self, vocab_size, block_size, n_emb, n_layer, n_head):
        super().__init__()
        
        self.out_emb_table = nn.Embedding(vocab_size, n_emb)
        self.pos_emb_table = nn.Embedding(block_size, n_emb)
        self.blocks = nn.Sequential(*[Block(n_emb, block_size, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_emb)
        self.lm_head = nn.Linear(n_emb, vocab_size)
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        
    def forward(self, x, targets=None):
        _, T = x.shape
        out_emb_table = self.out_emb_table(x)
        pos_emb_table = self.pos_emb_table(torch.arange(T, device=device))
        
        x = out_emb_table + pos_emb_table
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        if targets == None:
            loss = None
        else:
            logits = logits.view(-1, vocab_size)
            targets = targets.view(-1)
        
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    
    def generate(self, x, block_size, max_new_tokens):
        for _ in range(max_new_tokens):
            
            xcropped = x[:, -block_size:]
            logits, _ = self(xcropped)
            logits = logits[:, -1, :] # B C        
            probs = F.softmax(logits, 1)
            xnext = torch.multinomial(probs, num_samples=1)
            
            x = torch.cat((x, xnext), dim=1)
            
        return x


# In[33]:


@torch.no_grad()
def estimate_loss(model, eval_iters, block_size, batch_size):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(block_size, batch_size, split=='train')
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# In[34]:


model = MultiLayerBigram(vocab_size, block_size, n_emb, n_layer, n_head)
model.train()
model = model.to(device)


# In[40]:


# model.load_state_dict(torch.load('model03.pth'))


# In[ ]:


num_it = 2500
lr = 1e-7
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-8)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=num_it - num_it / 10, gamma=0.1)


# In[ ]:


lossi = []
ud = [] 
for i in range(num_it+1):
    if(i % 500 == 0):
        losses = estimate_loss(model, eval_iters, block_size, batch_size)
        print(f"step {i}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        
    x_batch, y_batch = get_batch(block_size, batch_size)
    
    _, loss = model(x_batch, y_batch)
    
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    
    scheduler.step()
    
    lossi.append(loss.log10().item())
    with torch.no_grad():
        ud.append([((lr*p.grad).std() / p.data.std()).log10().item() for p in model.parameters()])


# In[39]:


xgen = torch.zeros((8, block_size), dtype=torch.long, device=device)

model.eval()
with torch.no_grad():
    generations = model.generate(xgen, block_size, 1000)
    out = []
    for i in range(generations.shape[0]):
        words = generations[i].tolist()
        decoded = decode(words)
        out.append(decoded)
        
with open("poems_output.txt", "w", encoding="utf-8") as file:
    file.write(''.join(out))


# In[111]:


# model_path = 'model03.pth'
# torch.save(model.state_dict(), model_path)


# In[ ]:




