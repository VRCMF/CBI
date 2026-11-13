# =====================================================================================
# START: GNN-based code to be added
# =====================================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, f1_score
from collections import defaultdict
import pandas as pd
import numpy as np
from tqdm import tqdm
import ipdb # If you need debugging

# Keep this line if you use GPU, it's defined in your main script
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Dataset configurations
DATASET_CONFIGS = {
    'matres': {
        'path': 'data/temporal_relations_matres.csv',
        'labels': ['BEFORE', 'AFTER', 'SIMULTANEOUS']
    },
    'tbd': {
        'path': 'data/temporal_relations_tbd.csv',
        'labels': ['BEFORE', 'AFTER', 'INCLUDES', 'IS_INCLUDED', 'SIMULTANEOUS']
    },
    'tcr': {
        'path': 'data/temporal_relations_tcr.csv',
        'labels': ['BEFORE', 'AFTER', 'SIMULTANEOUS']
    },
    'tdd_auto': {
        'path': 'data/temporal_relations_tdd_auto.csv',
        'labels': ['BEFORE', 'AFTER', 'INCLUDES', 'IS_INCLUDED', 'SIMULTANEOUS']
    },
    'tdd_man': {
        'path': 'data/temporal_relations_tdd_man.csv',
        'labels': ['BEFORE', 'AFTER', 'INCLUDES', 'IS_INCLUDED', 'SIMULTANEOUS']
    }
}

class GATLayer(nn.Module):
    """
    Graph Attention Layer based on the paper's formulation, which incorporates edge features.
    This implementation uses a dense attention mechanism, suitable for the fully-connected
    graphs described in the paper for each document.
    """
    def __init__(self, in_node_dim, in_edge_dim, out_node_dim, num_heads, dropout=0.1, alpha=0.2):
        super(GATLayer, self).__init__()
        self.num_heads = num_heads
        self.out_node_dim = out_node_dim

        # Linear transformations for node features, one for each head
        self.W = nn.Linear(in_node_dim, num_heads * out_node_dim, bias=False)
        # The attention mechanism's learnable vector 'a_k'
        self.a = nn.Parameter(torch.empty(size=(1, num_heads, 2 * out_node_dim + in_edge_dim)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(alpha)
        self.dropout = nn.Dropout(dropout)

    def forward(self, node_features, edge_features, adj_mask):
        """
        Args:
            node_features (Tensor): Input node features of shape (N, in_node_dim), where N is num of nodes.
            edge_features (Tensor): Edge features of shape (N, N, in_edge_dim).
            adj_mask (Tensor): Adjacency mask (N, N) to handle non-existent edges in attention softmax.
        """
        N = node_features.size(0)

        # 1. Linearly transform node features and reshape for multi-head attention
        h_proj = self.W(node_features)  # (N, num_heads * out_node_dim)
        h_proj = h_proj.view(N, self.num_heads, self.out_node_dim) # (N, H, d_out)

        # 2. Prepare features for attention score calculation
        # Create all pairs of node features
        h_i = h_proj.unsqueeze(1).expand(N, N, self.num_heads, self.out_node_dim) # (N, N, H, d_out)
        h_j = h_proj.unsqueeze(0).expand(N, N, self.num_heads, self.out_node_dim) # (N, N, H, d_out)

        # Expand edge features to match head dimension
        p_ij = edge_features.unsqueeze(2).expand(N, N, self.num_heads, -1) # (N, N, H, d_edge)

        # 3. Calculate attention scores as per Equation 3
        # Concatenate features: [W*h_i || W*h_j || p_ij]
        concat_features = torch.cat([h_i, h_j, p_ij], dim=-1) # (N, N, H, 2*d_out + d_edge)

        # Apply attention vector `a`
        e = (concat_features * self.a).sum(dim=-1) # (N, N, H)
        e = self.leakyrelu(e)

        # 4. Apply softmax to get attention weights
        attention = e.masked_fill(adj_mask.unsqueeze(-1) == 0, -1e9) # Mask non-existent edges
        attention = F.softmax(attention, dim=1) # Softmax over neighbors j for each i
        attention = self.dropout(attention)

        # 5. Aggregate features from neighbors
        # h_j has shape (N, N, H, d_out). We want to aggregate h_j for each i.
        h_prime = torch.einsum('ijh,jhd->ihd', attention, h_proj) # (N, H, d_out)
        h_prime = h_prime.reshape(N, self.num_heads * self.out_node_dim) # (N, H * d_out)

        return F.elu(h_prime)


class GNN_Classifier(nn.Module):
    """
    The complete GNN module, stacking GAT layers and a final classifier.
    """
    def __init__(self, num_events, node_embed_dim, edge_feature_dim, gnn_hidden_dim, num_classes, num_heads=8, dropout=0.1):
        super(GNN_Classifier, self).__init__()
        # Initial node features: A learnable embedding for each potential event in a document.
        # NOTE: A more powerful approach, as hinted in the paper, would be to extract these
        # features from the LLM's hidden states for each event. Using nn.Embedding is a
        # practical alternative when that is not feasible.
        self.node_embeddings = nn.Embedding(num_events, node_embed_dim)

        self.gat1 = GATLayer(node_embed_dim, edge_feature_dim, gnn_hidden_dim // num_heads, num_heads, dropout)
        self.gat2 = GATLayer(gnn_hidden_dim, edge_feature_dim, gnn_hidden_dim // num_heads, num_heads, dropout)
        
        # Final classifier that takes [h_i^(2) || p_ij || h_j^(2)]
        classifier_input_dim = 2 * gnn_hidden_dim + edge_feature_dim
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, gnn_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gnn_hidden_dim, num_classes)
        )

    def forward(self, node_ids, edge_features, edge_index, adj_mask):
        """
        Args:
            node_ids (Tensor): Tensor of unique node identifiers for embedding lookup.
            edge_features (Tensor): Dense edge feature matrix (N, N, d_edge).
            edge_index (Tensor): List of edges (2, E).
            adj_mask (Tensor): Adjacency mask (N, N).
        """
        # 1. Get initial node features
        h0 = self.node_embeddings(node_ids)

        # 2. Pass through GAT layers
        h1 = self.gat1(h0, edge_features, adj_mask)
        h2 = self.gat2(h1, edge_features, adj_mask)

        # 3. Prepare features for final classification for each edge
        src_nodes, dst_nodes = edge_index[0], edge_index[1]
        h_i = h2[src_nodes] # (E, d_hidden)
        h_j = h2[dst_nodes] # (E, d_hidden)
        p_ij = edge_features[src_nodes, dst_nodes] # (E, d_edge)

        # Concatenate features: [h_i^(2) || p_ij || h_j^(2)]
        final_edge_features = torch.cat([h_i, p_ij, h_j], dim=-1)

        # 4. Classify each edge
        logits = self.classifier(final_edge_features)
        return logits


class GraphDataset(Dataset):
    """Custom Dataset for graph objects."""
    def __init__(self, graph_data):
        self.graph_data = graph_data

    def __len__(self):
        return len(self.graph_data)

    def __getitem__(self, idx):
        return self.graph_data[idx]

# =====================================================================================
# END: GNN-based code to be added
# =====================================================================================

# # =====================================================================================
# # START: GNN training function to be added
# # =====================================================================================
# def run_gnn_downstream_task(
#     llm_model,
#     llm_tokenizer,
#     train_df,
#     val_df,
#     test_df,
#     dataset_name,
#     args,
#     device
# ):
#     """
#     Main function to prepare graph data, train, and evaluate the GNN classifier.
#     """
#     print("\n" + "="*60)
#     print("Starting GNN Downstream Task")
#     print("="*60)

#     # === A. Data Preparation: Convert flat dataframe into graphs ===
#     print("Step 1: Preparing graph data...")
#     all_data = pd.concat([train_df, val_df, test_df], ignore_index=True)
    
#     # Sort by document ID to process documents sequentially
#     all_data_sorted = all_data.sort_values(by='doc_id').reset_index(drop=True)

#     llm_model.eval()
#     llm_model.to(device)
    
#     graph_data_list = []
    
#     with torch.no_grad():
#         for doc_id, group in tqdm(all_data_sorted.groupby('doc_id'), desc="Processing Documents"):
#             events = sorted(list(set(group['event_1'].tolist() + group['event_2'].tolist())))
#             event_to_node_id = {event: i for i, event in enumerate(events)}
#             num_nodes = len(events)
            
#             if num_nodes <= 1:
#                 continue

#             # Create a map from (event1, event2) to its data row
#             pair_map = {(row['event_1'], row['event_2']): row for _, row in group.iterrows()}
            
#             # Initialize graph tensors
#             label_map = DATASET_CONFIGS[dataset_name]['labels']
#             label_to_idx = {label: i for i, label in enumerate(label_map)}
#             num_classes = len(label_map)
            
#             edge_features = torch.zeros(num_nodes, num_nodes, num_classes)
#             adj_mask = torch.zeros(num_nodes, num_nodes, dtype=torch.long)
#             edge_index_list = []
#             labels_list = []
#             split_list = []
            
#             # Populate graph tensors by iterating through all possible pairs
#             for i in range(num_nodes):
#                 for j in range(num_nodes):
#                     if i == j:
#                         continue
                        
#                     event1_str = events[i]
#                     event2_str = events[j]
                    
#                     row = pair_map.get((event1_str, event2_str))
                    
#                     if row is not None:
#                         # This pair exists in the dataset, get its features
#                         text1 = str(row["event_ctx_1"]).strip()
#                         text2 = str(row["event_ctx_2"]).strip()
#                         label = label_to_idx[str(row["relation"]).strip()]
#                         split = row["split"]
                        
#                         # Get probability distribution from the LLM
#                         if args.use_routing.lower() == 'true':
#                             # This part assumes a collate function similar to your training one
#                             enc = llm_tokenizer([text1, text2], return_tensors="pt", padding=True, truncation=True)
#                             logits = llm_model(
#                                 input_ids=enc['input_ids'].to(device),
#                                 attention_mask=enc['attention_mask'].to(device),
#                                 vision_input_ids=enc['input_ids'].clone().to(device), # Dummy vision input if needed by model arch
#                                 vision_attention_mask=enc['attention_mask'].clone().to(device)
#                             )
#                         else:
#                             enc = llm_tokenizer(f"{text1} [SEP] {text2}", return_tensors="pt", padding=True, truncation=True)
#                             logits = llm_model(
#                                 input_ids=enc['input_ids'].to(device),
#                                 attention_mask=enc['attention_mask'].to(device)
#                             )
                        
#                         probabilities = F.softmax(logits, dim=-1).squeeze().cpu()
                        
#                         # Store features
#                         edge_features[i, j, :] = probabilities
#                         adj_mask[i, j] = 1
#                         edge_index_list.append([i, j])
#                         labels_list.append(label)
#                         split_list.append(split)

#             if not edge_index_list:
#                 continue

#             graph_data_list.append({
#                 'node_ids': torch.arange(num_nodes),
#                 'edge_features': edge_features,
#                 'adj_mask': adj_mask,
#                 'edge_index': torch.tensor(edge_index_list, dtype=torch.long).t(),
#                 'labels': torch.tensor(labels_list, dtype=torch.long),
#                 'split': split_list # Keep track of which split each edge belongs to
#             })

#     # Split graph data back into train/val/test
#     train_graphs, val_graphs, test_graphs = [], [], []
#     for graph in graph_data_list:
#       # A graph can contain edges from multiple splits if a document spans them.
#       # We create a graph for each split containing only the relevant edges.
#       for split_type in ['train', 'validation', 'test']:
#           edge_indices_in_split = [i for i, s in enumerate(graph['split']) if s == split_type]
#           if not edge_indices_in_split:
#               continue
          
#           new_graph = {
#               'node_ids': graph['node_ids'],
#               'edge_features': graph['edge_features'],
#               'adj_mask': graph['adj_mask'],
#               'edge_index': graph['edge_index'][:, edge_indices_in_split],
#               'labels': graph['labels'][edge_indices_in_split]
#           }
#           if split_type == 'train':
#               train_graphs.append(new_graph)
#           elif split_type == 'validation':
#               val_graphs.append(new_graph)
#           else:
#               test_graphs.append(new_graph)


#     print(f"Created {len(train_graphs)} train, {len(val_graphs)} val, {len(test_graphs)} test graphs.")

#     # Create DataLoaders
#     train_gnn_dataset = GraphDataset(train_graphs)
#     val_gnn_dataset = GraphDataset(val_graphs)
#     test_gnn_dataset = GraphDataset(test_graphs)
    
#     # NOTE: Batching graphs is complex. For simplicity, we use batch_size=1.
#     # This means we process one document (graph) at a time.
#     train_gnn_loader = DataLoader(train_gnn_dataset, batch_size=1, shuffle=True)
#     val_gnn_loader = DataLoader(val_gnn_dataset, batch_size=1, shuffle=False)
#     test_gnn_loader = DataLoader(test_gnn_dataset, batch_size=1, shuffle=False)

#     # === B. GNN Model Initialization ===
#     print("\nStep 2: Initializing GNN model...")
#     gnn_model = GNN_Classifier(
#         num_events=500, # Max possible events in a document, adjust if needed
#         node_embed_dim=128,
#         edge_feature_dim=len(DATASET_CONFIGS[dataset_name]['labels']),
#         gnn_hidden_dim=256,
#         num_classes=len(DATASET_CONFIGS[dataset_name]['labels']),
#         num_heads=8
#     ).to(device)

#     # === C. GNN Training and Evaluation ===
#     print("\nStep 3: Training GNN model...")
#     optimizer = torch.optim.AdamW(gnn_model.parameters(), lr=1e-4, weight_decay=1e-5)
#     criterion = nn.CrossEntropyLoss()
#     best_val_f1 = 0.0

#     for epoch in range(100): # Train GNN for more epochs if needed
#         gnn_model.train()
#         total_loss = 0
#         for batch in train_gnn_loader:
#             # Unpack and move to device (with batch_size=1, squeeze is needed)
#             node_ids = batch['node_ids'].squeeze(0).to(device)
#             edge_features = batch['edge_features'].squeeze(0).to(device)
#             edge_index = batch['edge_index'].squeeze(0).to(device)
#             adj_mask = batch['adj_mask'].squeeze(0).to(device)
#             labels = batch['labels'].squeeze(0).to(device)

#             if labels.numel() == 0: continue
            
#             optimizer.zero_grad()
#             logits = gnn_model(node_ids, edge_features, edge_index, adj_mask)
#             loss = criterion(logits, labels)
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item()
        
#         print(f"Epoch {epoch+1}/10 - GNN Train Loss: {total_loss / len(train_gnn_loader):.4f}")

#         # Validation
#         gnn_model.eval()
#         all_preds, all_labels = [], []
#         with torch.no_grad():
#             for batch in val_gnn_loader:
#                 node_ids = batch['node_ids'].squeeze(0).to(device)
#                 edge_features = batch['edge_features'].squeeze(0).to(device)
#                 edge_index = batch['edge_index'].squeeze(0).to(device)
#                 adj_mask = batch['adj_mask'].squeeze(0).to(device)
#                 labels = batch['labels'].squeeze(0).to(device)
                
#                 if labels.numel() == 0: continue

#                 logits = gnn_model(node_ids, edge_features, edge_index, adj_mask)
#                 preds = torch.argmax(logits, dim=1)
#                 all_preds.extend(preds.cpu().numpy())
#                 all_labels.extend(labels.cpu().numpy())
        
#         val_f1 = f1_score(all_labels, all_preds, average='micro')
#         print(f"GNN Validation F1 (micro): {val_f1:.4f}")
        
#         if val_f1 > best_val_f1:
#             best_val_f1 = val_f1
#             torch.save(gnn_model.state_dict(), f"{args.output_dir}/best_gnn_model_{dataset_name}.pt")

#     # Final Test Evaluation
#     print("\nStep 4: Evaluating GNN model on Test Set...")
#     gnn_model.load_state_dict(torch.load(f"{args.output_dir}/best_gnn_model_{dataset_name}.pt"))
#     gnn_model.eval()
#     all_preds, all_labels = [], []
#     with torch.no_grad():
#         for batch in test_gnn_loader:
#             node_ids = batch['node_ids'].squeeze(0).to(device)
#             edge_features = batch['edge_features'].squeeze(0).to(device)
#             edge_index = batch['edge_index'].squeeze(0).to(device)
#             adj_mask = batch['adj_mask'].squeeze(0).to(device)
#             labels = batch['labels'].squeeze(0).to(device)
            
#             if labels.numel() == 0: continue

#             logits = gnn_model(node_ids, edge_features, edge_index, adj_mask)
#             preds = torch.argmax(logits, dim=1)
#             all_preds.extend(preds.cpu().numpy())
#             all_labels.extend(labels.cpu().numpy())

#     report = classification_report(
#         all_labels,
#         all_preds,
#         target_names=DATASET_CONFIGS[dataset_name]['labels'],
#         digits=4
#     )
#     f1_micro = f1_score(all_labels, all_preds, average='micro')
    
#     print("="*60)
#     print("GNN FINAL TEST RESULTS")
#     print("="*60)
#     print(f"Micro-F1: {f1_micro:.4f}\n")
#     print(report)
    
# # =====================================================================================
# # END: GNN training function to be added
# # =====================================================================================

# =====================================================================================
# START: GNN training function to be added (CORRECTED VERSION)
# =====================================================================================
def run_gnn_downstream_task(
    llm_model,
    llm_tokenizer,
    train_df,
    val_df,
    test_df,
    dataset_name,
    args,
    device,
    logger,
):
    """
    Main function to prepare graph data, train, and evaluate the GNN classifier.
    """
    logger.info("\n" + "="*60)
    logger.info("Starting GNN Downstream Task")
    logger.info("="*60)

    # === A. Data Preparation: Convert flat dataframe into graphs ===
    logger.info("Step 1: Preparing graph data...")
    all_data = pd.concat([train_df, val_df, test_df], ignore_index=True)

    # Sort by document ID to process documents sequentially
    all_data_sorted = all_data.sort_values(by='doc_id').reset_index(drop=True)

    llm_model.eval()
    llm_model.to(device)
    
    graph_data_list = []
    
    with torch.no_grad():
        for doc_id, group in tqdm(all_data_sorted.groupby('doc_id'), desc="Processing Documents"):
            events = sorted(list(set(group['event_1'].tolist() + group['event_2'].tolist())))
            event_to_node_id = {event: i for i, event in enumerate(events)}
            num_nodes = len(events)
            
            if num_nodes <= 1:
                continue

            # Create a map from (event1, event2) to its data row
            pair_map = {(row['event_1'], row['event_2']): row for _, row in group.iterrows()}
            
            # Initialize graph tensors
            label_map = DATASET_CONFIGS[dataset_name]['labels']
            label_to_idx = {label: i for i, label in enumerate(label_map)}
            num_classes = len(label_map)
            
            edge_features = torch.zeros(num_nodes, num_nodes, num_classes)
            adj_mask = torch.zeros(num_nodes, num_nodes, dtype=torch.long)
            edge_index_list = []
            labels_list = []
            split_list = []
            
            for i in range(num_nodes):
                for j in range(num_nodes):
                    if i == j:
                        continue
                        
                    event1_str = events[i]
                    event2_str = events[j]
                    
                    row = pair_map.get((event1_str, event2_str))
                    
                    if row is not None:
                        text1 = str(row["event_ctx_1"]).strip()
                        text2 = str(row["event_ctx_2"]).strip()
                        label = label_to_idx[str(row["relation"]).strip()]
                        split = row["split"]
                        
                        # --- START OF CORRECTION ---
                        # Get probability distribution from the LLM
                        if args.use_routing.lower() == 'true':
                            # This block now correctly mimics the `collate_fn_pair_equal` logic
                            # for a single sample to produce one output logit.
                            enc1 = llm_tokenizer(text1, add_special_tokens=True, truncation=True, max_length=128)
                            enc2 = llm_tokenizer(text2, add_special_tokens=True, truncation=True, max_length=128)
                            ids1, ids2 = enc1["input_ids"], enc2["input_ids"]
                            mask1, mask2 = enc1["attention_mask"], enc2["attention_mask"]

                            max_len = max(len(ids1), len(ids2))
                            pad_id = llm_tokenizer.pad_token_id if llm_tokenizer.pad_token_id is not None else 0
                            
                            ids1_padded = ids1 + [pad_id] * (max_len - len(ids1))
                            ids2_padded = ids2 + [pad_id] * (max_len - len(ids2))
                            mask1_padded = mask1 + [0] * (max_len - len(mask1))
                            mask2_padded = mask2 + [0] * (max_len - len(mask2))

                            # Create a batch of size 1
                            input_ids = torch.tensor([ids1_padded]).to(device)
                            attention_mask = torch.tensor([mask1_padded]).to(device)
                            vision_input_ids = torch.tensor([ids2_padded]).to(device)
                            vision_attention_mask = torch.tensor([mask2_padded]).to(device)

                            logits = llm_model(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                vision_input_ids=vision_input_ids,
                                vision_attention_mask=vision_attention_mask
                            )
                        else:
                            # This part was already correct: combines texts into a single sequence
                            enc = llm_tokenizer(f"{text1} [SEP] {text2}", return_tensors="pt", padding=True, truncation=True, max_length=256)
                            logits = llm_model(
                                input_ids=enc['input_ids'].to(device),
                                attention_mask=enc['attention_mask'].to(device)
                            )
                        
                        # Logits will now have shape [1, num_classes].
                        # Squeezing removes the batch dimension, resulting in [num_classes].
                        probabilities = F.softmax(logits, dim=-1).squeeze().cpu()
                        # --- END OF CORRECTION ---
                        
                        # Store features
                        edge_features[i, j, :] = probabilities
                        adj_mask[i, j] = 1
                        edge_index_list.append([i, j])
                        labels_list.append(label)
                        split_list.append(split)

            if not edge_index_list:
                continue

            graph_data_list.append({
                'node_ids': torch.arange(num_nodes),
                'edge_features': edge_features,
                'adj_mask': adj_mask,
                'edge_index': torch.tensor(edge_index_list, dtype=torch.long).t(),
                'labels': torch.tensor(labels_list, dtype=torch.long),
                'split': split_list
            })

    # (The rest of the function remains the same)
    # Split graph data back into train/val/test
    train_graphs, val_graphs, test_graphs = [], [], []
    for graph in graph_data_list:
      for split_type in ['train', 'validation', 'test']:
          edge_indices_in_split = [i for i, s in enumerate(graph['split']) if s == split_type]
          if not edge_indices_in_split:
              continue
          
          new_graph = {
              'node_ids': graph['node_ids'],
              'edge_features': graph['edge_features'],
              'adj_mask': graph['adj_mask'],
              'edge_index': graph['edge_index'][:, edge_indices_in_split],
              'labels': graph['labels'][edge_indices_in_split]
          }
          if split_type == 'train':
              train_graphs.append(new_graph)
          elif split_type == 'validation':
              val_graphs.append(new_graph)
          else:
              test_graphs.append(new_graph)


    logger.info(f"Created {len(train_graphs)} train, {len(val_graphs)} val, {len(test_graphs)} test graphs.")

    # Create DataLoaders
    train_gnn_dataset = GraphDataset(train_graphs)
    val_gnn_dataset = GraphDataset(val_graphs)
    test_gnn_dataset = GraphDataset(test_graphs)
    
    train_gnn_loader = DataLoader(train_gnn_dataset, batch_size=1, shuffle=True)
    val_gnn_loader = DataLoader(val_gnn_dataset, batch_size=1, shuffle=False)
    test_gnn_loader = DataLoader(test_gnn_dataset, batch_size=1, shuffle=False)

    # === B. GNN Model Initialization ===
    logger.info("\nStep 2: Initializing GNN model...")
    gnn_model = GNN_Classifier(
        num_events=500,
        node_embed_dim=128,
        edge_feature_dim=len(DATASET_CONFIGS[dataset_name]['labels']),
        gnn_hidden_dim=256,
        num_classes=len(DATASET_CONFIGS[dataset_name]['labels']),
        num_heads=8
    ).to(device)

    # === C. GNN Training and Evaluation ===
    logger.info("\nStep 3: Training GNN model...")
    optimizer = torch.optim.AdamW(gnn_model.parameters(), lr=1e-4, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()
    best_val_f1 = 0.0

    for epoch in range(50):
        gnn_model.train()
        total_loss = 0
        for batch in train_gnn_loader:
            node_ids = batch['node_ids'].squeeze(0).to(device)
            edge_features = batch['edge_features'].squeeze(0).to(device)
            edge_index = batch['edge_index'].squeeze(0).to(device)
            adj_mask = batch['adj_mask'].squeeze(0).to(device)
            labels = batch['labels'].squeeze(0).to(device)

            if labels.numel() == 0: continue
            
            optimizer.zero_grad()
            logits = gnn_model(node_ids, edge_features, edge_index, adj_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        logger.info(f"Epoch {epoch+1}/10 - GNN Train Loss: {total_loss / len(train_gnn_loader):.4f}")

        # Validation
        gnn_model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_gnn_loader:
                node_ids = batch['node_ids'].squeeze(0).to(device)
                edge_features = batch['edge_features'].squeeze(0).to(device)
                edge_index = batch['edge_index'].squeeze(0).to(device)
                adj_mask = batch['adj_mask'].squeeze(0).to(device)
                labels = batch['labels'].squeeze(0).to(device)
                
                if labels.numel() == 0: continue

                logits = gnn_model(node_ids, edge_features, edge_index, adj_mask)
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        val_f1 = f1_score(all_labels, all_preds, average='micro')
        logger.info(f"GNN Validation F1 (micro): {val_f1:.4f}")
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(gnn_model.state_dict(), f"{args.output_dir}/best_gnn_model_{dataset_name}.pt")

    # Final Test Evaluation
    logger.info("\nStep 4: Evaluating GNN model on Test Set...")
    gnn_model.load_state_dict(torch.load(f"{args.output_dir}/best_gnn_model_{dataset_name}.pt"))
    gnn_model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_gnn_loader:
            node_ids = batch['node_ids'].squeeze(0).to(device)
            edge_features = batch['edge_features'].squeeze(0).to(device)
            edge_index = batch['edge_index'].squeeze(0).to(device)
            adj_mask = batch['adj_mask'].squeeze(0).to(device)
            labels = batch['labels'].squeeze(0).to(device)
            
            if labels.numel() == 0: continue

            logits = gnn_model(node_ids, edge_features, edge_index, adj_mask)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    report = classification_report(
        all_labels,
        all_preds,
        target_names=DATASET_CONFIGS[dataset_name]['labels'],
        digits=4
    )
    f1_micro = f1_score(all_labels, all_preds, average='micro')
    
    logger.info("="*60)
    logger.info("GNN FINAL TEST RESULTS")
    logger.info("="*60)
    logger.info(f"Micro-F1: {f1_micro:.4f}\n")
    logger.info(report)

# =====================================================================================
# END: GNN training function to be added
# =====================================================================================