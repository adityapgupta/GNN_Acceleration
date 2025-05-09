import os
import torch

from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from ogb.nodeproppred import PygNodePropPredDataset

from torch_geometric.utils import scatter
from torch_geometric.utils.dropout import dropout_edge
from torch_geometric.loader import GraphSAINTNodeSampler, NeighborLoader, RandomNodeLoader

from sparsification.normal import NormalGCN
from sparsification.dropedge import DropoutGCN
from sparsification.sparse import GumbelGCN

torch.manual_seed(0)


def train_eval(
    data_dir='./data', 
    sparse='none',
    sample='random',
    train_parts=100, 
    val_parts=25, 
    test_parts=25, 
    epochs=100, 
    lr=1e-3, 
    weight_decay=5e-4, 
    temperature=0.05, 
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), 
    hidden1=16, 
    hidden2=16,
    k=5,
    input_dim=8,
    output_dim=112,
    edge_feature_dim=8,
    dropout_ratio=0.2,
    ):
    
    # Load the dataset
    dataset = PygNodePropPredDataset('ogbn-proteins', root=data_dir)
    splitted_idx = dataset.get_idx_split()
    data = dataset[0]
    data.node_species = None
    data.y = data.y.to(torch.float)

    # Initialize features of nodes by aggregating edge features
    row, col = data.edge_index
    data.x = scatter(data.edge_attr, col, dim_size=data.num_nodes, reduce='sum')

    # Create masks for training, validation, and test nodes
    mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    mask[splitted_idx["train"]] = True
    data['train_mask'] = mask

    mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    mask[splitted_idx["valid"]] = True
    data['valid_mask'] = mask

    mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    mask[splitted_idx["test"]] = True
    data['test_mask'] = mask

    # Create data loaders
    if sample == 'graphsage':
        train_loader = NeighborLoader(data, num_neighbors=[10, 10], batch_size=data.num_nodes//train_parts, input_nodes=splitted_idx["train"])
        val_loader = NeighborLoader(data, num_neighbors=[10, 10], batch_size=data.num_nodes//val_parts, input_nodes=splitted_idx["valid"])
        test_loader = NeighborLoader(data, num_neighbors=[10, 10], batch_size=data.num_nodes//test_parts, input_nodes=splitted_idx["test"])
    elif sample == 'graphsaint':
        train_loader = GraphSAINTNodeSampler(data, batch_size=data.num_nodes//train_parts, num_steps=train_parts)
        val_loader = GraphSAINTNodeSampler(data, batch_size=data.num_nodes//val_parts, num_steps=val_parts)
        test_loader = GraphSAINTNodeSampler(data, batch_size=data.num_nodes//test_parts, num_steps=test_parts)
    else:
        train_loader = RandomNodeLoader(data, num_parts=train_parts, shuffle=True)
        val_loader = RandomNodeLoader(data, num_parts=val_parts, shuffle=False)
        test_loader = RandomNodeLoader(data, num_parts=test_parts, shuffle=False)

    # Initialize the model
    if sparse == 'dropedge':
        model = DropoutGCN(input_dim, output_dim, edge_feature_dim, k, device, hidden1, hidden2, temperature).to(device)
    elif sparse == 'neural':
        model = GumbelGCN(input_dim, output_dim, edge_feature_dim, k, device, hidden1, hidden2, temperature).to(device)
    else:
        model = NormalGCN(input_dim, output_dim, edge_feature_dim, k, device, hidden1, hidden2, temperature).to(device)

    # Save path
    save_path = f"./models/{sparse}_{sample}/"
    os.makedirs(save_path, exist_ok=True)

    # Set up the optimizer with weight decay for the first convolutional layer
    optimizer = torch.optim.Adam([
    {'params': model.conv.conv1.parameters(), 'weight_decay': weight_decay},
    {'params': [p for n, p in model.named_parameters() if 'conv.conv1' not in n], 'weight_decay': 0}
    ], lr)
    
    # Define the loss function
    criterion = torch.nn.BCEWithLogitsLoss()

    patience = 10
    best_val = float('-inf')
    patience_counter = 0

    # Training loop
    for epoch in range(1, epochs + 1):
        model.train()
        pbar = tqdm(total=len(train_loader))
        pbar.set_description(f'Training epoch: {epoch:03d}')

        train_total_loss = train_total_examples = 0
        
        if sparse == 'dropedge':
            data.edge_index, _ = dropout_edge(data.edge_index, dropout_ratio)

        # Iterate over the training data
        for data in train_loader:
            edge_index = data.edge_index
            edge_attr = data.edge_attr
            x = data.x.to(device)
            edge_index = edge_index.to(device)
            edge_attr = edge_attr.to(device)
            num_nodes = data.num_nodes
            train_mask = data.train_mask
            y = data.y[train_mask].to(device)

            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            logits = model(num_nodes, edge_index, edge_attr, x, train_mask, training=True)

            # Compute the loss
            loss = criterion(logits, y)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            train_total_loss += loss.item() * train_mask.sum().item()
            train_total_examples += train_mask.sum().item()

            # Update the progress bar
            pbar.update(1)

        pbar.close()
        torch.cuda.empty_cache()

        # Validation loop
        model.eval()
        pbar = tqdm(total=len(val_loader))
        pbar.set_description(f'Validation epoch: {epoch:03d}')

        val_total = val_total_examples = 0

        # Iterate over the validation data
        for data in val_loader:
            valid_mask = data.valid_mask
            if valid_mask.sum() == 0:
                pbar.update(1)
                continue

            edge_index = data.edge_index
            edge_attr = data.edge_attr
            x = data.x.to(device)
            edge_index = edge_index.to(device)
            edge_attr = edge_attr.to(device)
            num_nodes = data.num_nodes
            y = data.y[valid_mask].to(device)

            # Forward pass
            logits = model(num_nodes, edge_index, edge_attr, x, valid_mask, training=False)

            probs = torch.sigmoid(logits)

            val_total+= roc_auc_score(y.cpu().numpy(), probs.detach().cpu().numpy())
            val_total_examples+=1

            # Update the progress bar
            pbar.update(1)

        pbar.close()

        torch.save(model.state_dict(), save_path + 'curr_model.pth')

        # Early stopping
        if val_total > best_val:
            best_val = val_total
            patience_counter = 0
            torch.save(model.state_dict(), save_path + 'best_model.pth')

        else:
            patience_counter += 1
            if patience_counter == patience:
                print(f'Early stopping at epoch: {epoch}')
                break

        print(f'Epoch: {epoch:03d}, Train Loss: {train_total_loss / train_total_examples:.4f}, Val ROC AUC: {val_total / val_total_examples:.4f}')
        torch.cuda.empty_cache()
        print()

    del model

    # Test the model
    if sparse == 'neural':
        model = GumbelGCN(input_dim, output_dim, edge_feature_dim, k, device, hidden1, hidden2, temperature).to(device)
    elif sparse == 'dropedge':
        model = DropoutGCN(input_dim, output_dim, edge_feature_dim, k, device, hidden1, hidden2, temperature).to(device)
    else:
        model = NormalGCN(input_dim, output_dim, edge_feature_dim, k, device, hidden1, hidden2, temperature).to(device)

    model.load_state_dict(torch.load(save_path + 'best_model.pth'))
    model.to(device)
    model.eval()
    pbar = tqdm(total=len(test_loader))
    pbar.set_description(f'Testing')

    test_total = test_total_examples = 0

    # Iterate over the test data
    for data in test_loader:
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        x = data.x.to(device)
        edge_index = edge_index.to(device)
        edge_attr = edge_attr.to(device)
        num_nodes = data.num_nodes
        test_mask = data.test_mask
        y = data.y[test_mask].to(device)

        if test_mask.sum() == 0:
            pbar.update(1)
            continue

        # Forward pass
        logits = model(num_nodes, edge_index, edge_attr, x, test_mask, training=False)

        probs = torch.sigmoid(logits)

        test_total+= roc_auc_score(y.cpu().numpy(), probs.detach().cpu().numpy())
        test_total_examples+=1

        # Update the progress bar
        pbar.update(1)

    pbar.close()

    print(f'Test ROC AUC: {test_total / test_total_examples:.4f}')
    torch.cuda.empty_cache()
