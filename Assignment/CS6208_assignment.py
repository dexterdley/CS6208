from torch_geometric.datasets import MNISTSuperpixels
from torch_geometric.loader import DataLoader
import torch

#Define model
import torch.nn as nn
import torch.nn.functional as F 
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch_geometric.utils

from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.conv import MessagePassing
import matplotlib.pyplot as plt

# Load the MNISTSuperpixel dataset
train_data = MNISTSuperpixels(root=".", train=True)
test_data = MNISTSuperpixels(root=".", train=False)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

embedding_size = 64

def convert_to_onehot(array, num_classes=10):
    
    B = len(array) #batch size
    
    out = torch.zeros((B, num_classes))
    out[range(B), array.to("cpu")] = 1
    
    return torch.FloatTensor(out)

def negative_log_likelihood(pred, y):

    ce_loss = y * torch.log(pred) 
    ce_loss = -torch.sum(ce_loss, dim=1)

    return torch.mean(ce_loss)


class my_GCNConv(MessagePassing):
    
    def __init__(self, in_channels: int, out_channels: int):
        super(my_GCNConv, self).__init__(aggr='add')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.linear = Linear(in_channels, out_channels, weight_initializer='glorot')
        
    def forward(self, x, edge_index):
        edge_index, _ = torch_geometric.utils.add_self_loops(edge_index, num_nodes=x.size(0) )
        x = self.linear(x)
        
        #normalize here
        row, col = edge_index
        deg = torch_geometric.utils.degree(col, x.size(0), dtype=x.dtype)
        D = 1/torch.sqrt(deg)
        norm = D[row] * D[col]

        out = self.propagate(edge_index, x=x, norm=norm)

        return out
    
    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j


class GCN(torch.nn.Module):
    def __init__(self):
        # Init parent
        super(GCN, self).__init__()
        torch.manual_seed(42)
        # GCN layers
        self.initial_conv = GCNConv(train_data.num_features, embedding_size)
        self.attention = my_GATLayer(75, train_data.num_features)
        self.conv1 = GCNConv(embedding_size, embedding_size)
        self.conv2 = GCNConv(embedding_size, embedding_size)
        self.conv3 = GCNConv(embedding_size, embedding_size)
        # Output layer
        self.out = nn.Linear(embedding_size*2, train_data.num_classes)
    def forward(self, x, edge_index, batch_index):
        # First Conv layer
        #x_prime = self.attention(x, edge_index, batch_index)
        #hidden = self.initial_conv(x_prime, edge_index)
        hidden = self.initial_conv(x, edge_index)
        hidden = F.tanh(hidden)
        
        # Other Conv layers
        hidden = self.conv1(hidden, edge_index)
        hidden = F.tanh(hidden)
        hidden = self.conv2(hidden, edge_index)
        hidden = F.tanh(hidden)
        hidden = self.conv3(hidden, edge_index)
        hidden = F.tanh(hidden)
        # Global Pooling (stack different aggregations)
        hidden = torch.cat([gmp(hidden, batch_index), 
                            gap(hidden, batch_index)], dim=1)
        # Apply a final (linear) classifier.
        out = self.out(hidden)
        return out, hidden


class my_GATLayer(nn.Module):
    def __init__(self, in_features, out_features ):
        super(my_GATLayer, self).__init__()
        self.in_features   = in_features
        self.out_features  = out_features

        self.W = nn.Parameter(torch.zeros(size=(self.in_features, self.out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(0.2)

    def forward(self, x, edge_index, batch_index):
        
        X = torch_geometric.utils.to_dense_batch(x, batch_index)[0]
        A = torch_geometric.utils.to_dense_adj(edge_index, batch_index) #(B, 75, 75)
                         
        e = self.leakyrelu(X @ self.W.T)
        
        #Attention mask
        attention = -9e15 * torch.ones_like(e)
        attention[A>0] = e[A>0] #replace parts of adjacency that are non-zero with e values

        attention = F.softmax(attention, dim=1)

        X_prime = (attention @ X) + X
        
        return F.elu( X_prime.reshape(X_prime.shape[0] * X_prime.shape[1], 1) )



class My_GCN(torch.nn.Module):
    def __init__(self, attention):
        # Init parent
        super(My_GCN, self).__init__()
        torch.manual_seed(42)
        # GCN layers
        self.initial_conv = my_GCNConv(train_data.num_features, embedding_size)
        self.attention_layer = my_GATLayer(75, train_data.num_features)
        self.conv1 = my_GCNConv(embedding_size, embedding_size)
        self.conv2 = my_GCNConv(embedding_size, embedding_size)
        self.conv3 = my_GCNConv(embedding_size, embedding_size)
        self.out = nn.Linear(embedding_size*2, train_data.num_classes)
        self.attention = attention

    def forward(self, x, edge_index, batch_index):

        # First Conv layer
        if self.attention:
            x = self.attention_layer(x, edge_index, batch_index)
            
        hidden = self.initial_conv(x, edge_index)
        hidden = F.tanh(hidden)
        hidden = self.conv1(hidden, edge_index)
        hidden = F.tanh(hidden)
        hidden = self.conv2(hidden, edge_index)
        hidden = F.tanh(hidden)
        hidden = self.conv3(hidden, edge_index)
        hidden = F.tanh(hidden)
        # Global Pooling (stack different aggregations)
        hidden = torch.cat([gmp(hidden, batch_index), 
                            gap(hidden, batch_index)], dim=1)
        # Apply a final (linear) classifier.
        out = self.out(hidden)
        return out, hidden


#my_model = My_GCN().to("cuda")

pyg_model = GCN().to("cuda")
model =  My_GCN(attention=False).to("cuda")
my_model =  My_GCN(attention=True).to("cuda")

def testing_data(model, data_loader):
    
    ce_losses = []
    num_correct = 0
    num_samples = 0

    with torch.no_grad():
        model.eval()

        for batch in data_loader:
            batch.to(device)

            pred, _ = model(batch.x.float(), batch.edge_index, batch.batch)

            num_correct += torch.sum(torch.eq(pred.argmax(1), batch.y)).item()
            num_samples += pred.size(0)

    return(num_correct/num_samples) #test accuracy

def run_training(model):

	batch_sz = 16
	max_epochs = 75

	train_loader = DataLoader(train_data, batch_size=batch_sz, shuffle=True, num_workers=2, pin_memory=True)
	test_loader = DataLoader(test_data, batch_size=batch_sz, shuffle=False, num_workers=2, pin_memory=True)

	optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

	num_correct = 0
	num_samples = 0
	

	accuracy = []
	test_accuracy = []

	for epoch in range(max_epochs):
		losses = []

		for batch in train_loader:
			batch.to(device)

			pred, _ = model(batch.x.float(), batch.edge_index, batch.batch) # Passing the node features and the connection info
			probs = F.softmax(pred, dim=1)

			onehot = convert_to_onehot(batch.y).to(device)
			loss = negative_log_likelihood(probs, onehot)

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			num_correct += torch.sum(torch.eq(pred.argmax(1), batch.y)).item()
			num_samples += pred.size(0)

			losses.append(loss)

		accuracy.append(num_correct/num_samples)
		print("Epoch", epoch, "Loss", sum(losses)/len(losses), "Train acc", num_correct/num_samples)

		test_acc = testing_data(model, test_loader)
		test_accuracy.append(test_acc)

	return accuracy, test_accuracy

pyg_train_acc, pyg_test_acc = run_training(pyg_model)
my_train_acc, my_test_acc = run_training(model)
attn_train_acc, attn_test_acc = run_training(my_model)
#%%
plt.figure(1)
plt.title("Training set")
plt.plot(range(len(pyg_train_acc)), pyg_train_acc, 'r-')
plt.plot(range(len(my_train_acc)), my_train_acc, 'b-')
plt.plot(range(len(my_train_acc)), attn_train_acc, 'g-')
plt.ylabel("Accuracy")
plt.xlabel("Epochs")

plt.figure(2)
plt.title("Test set")
plt.plot(range(len(pyg_test_acc)), pyg_test_acc, 'r-')
plt.plot(range(len(my_test_acc)), my_test_acc, 'b-')
plt.plot(range(len(my_train_acc)), attn_test_acc, 'g-')

plt.ylabel("Accuracy")
plt.xlabel("Epochs")
plt.legend(['PyG GCNConv', 'My GCNConv', 'My GCNConv + Attention'], loc='lower right')


plt.show()

