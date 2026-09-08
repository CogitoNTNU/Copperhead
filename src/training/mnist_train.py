import numpy as np
import wandb

from src.data.mnist_loader import load_mnist
from src.nn.kristian import skeleton_nn   #bytte dette i deres trening

def softmax_cross_entropy_loss(logits,y_true):
    #softmax for å gjøre om nn outputs til sannsynlighetsfordeling
    #cross entropy loss for dette er vanlig i klassifiseringsproblemer
    shifted = logits - logits.max(axis = 1, keepdims = True)
    exp = np.exp(shifted)
    probs = exp / exp.sum(axis=1 , keepdims = True)

    n = logits.shape[0]

    log_likelyhood = -np.log(probs[np.arange(n), y_true] + 1e-12)

    loss = log_likelyhood.mean()

    grad = probs.copy()
    grad[np.arange(n), y_true] -= 1
    grad /= n
    return loss, grad

def accuracy(logits,y_true):
    preds = logits.argmax(axis=1)
    return (preds == y_true).mean()

def sgd_step(params,lr):
    for param, grad in params:
        param -= lr* grad

def iterate_batches(X , y , batch_size , rng):
    n = len(X)
    perm = rng.permutation(n)
    for start in range(0 , n ,batch_size):
        idx = perm[start : start + batch_size]
        yield  X[idx], y[idx]

def train(epochs = 5, batch_size = 64, lr = 0.1, seed = 0, engine_name='kristian'):
    #logge i wandb initalisering
    wandb.init(
        project='copperhead-mnist',
        name = f'{engine_name}-lr{lr}-bs{batch_size}',
        config = {
            'epochs' : epochs, 
            'batch_size' : batch_size,
            'lr': lr,
            }
    )
    
    #trening
    X_train, y_train, X_val, y_val = load_mnist(seed=seed)
    rng = np.random.default_rng(seed)

    nn = skeleton_nn.SkeletonNN()

    for epoch in range(1, epochs+1):
        epoch_losses = []
        for x_batch, y_batch in iterate_batches(X_train,y_train,batch_size,rng):
            logits = nn.forward(x_batch)
            loss, grad_loss = softmax_cross_entropy_loss(logits, y_batch)
            nn.backward(grad_loss)
            sgd_step(nn.params(),lr)
            epoch_losses.append(loss)

        val_logits = nn.forward(X_val)
        val_acc = accuracy(val_logits, y_val)
        train_loss = np.mean(epoch_losses)

        wandb.log({'train_loss' : train_loss,
                   'val_acc' : val_acc,
                   'epoch' : epoch})

        print(
            f'epoch {epoch} av {epochs}'
            f'trenings loss = {train_loss:.4f}'
            f'accuracy = {val_acc:.4f}'
        )

    wandb.finish()
    return nn

if __name__ == '__main__':
    train()