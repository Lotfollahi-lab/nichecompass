type(adata.X)
sparse_mx = adata.X.tocoo()
sparse_mx.shape
values = sparse_mx.data
indices = np.vstack((sparse_mx.row, sparse_mx.col))
shape = sparse_mx.shape
a= torch.FloatTensor(values)
b= torch.IntTensor(indices)
test = torch.sparse_coo_tensor(b, a, torch.Size(shape))
print(test)